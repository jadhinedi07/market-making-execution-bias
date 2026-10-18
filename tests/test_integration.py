"""
Integration tests on SYNTHETIC files in Tardis's exact CSV format.

Synthetic data is used here only to test code paths. It is never used for results.
"""
import numpy as np
import pandas as pd
import pytest

from mmbias.accounting import cash_and_position, fills_to_arrays, inventory_pnl, spread_capture
from mmbias.analysis import summarise_day
from mmbias.events import iter_book_messages
from mmbias.simulate import AgentSpec, run_day

T0 = 1_790_000_000_000_000


def synthetic_day(tmp_path, n=4000, seed=0, trade_lag_us=1):
    """Random-walk book (5 levels a side) plus trades at or through the touch.

    Each trade is stamped `trade_lag_us` before the book message that shows its effect,
    as on real Binance data (trades are reflected in the first message strictly after
    them). trade_lag_us=0 produces timestamp ties between trades and book messages.
    """
    rng = np.random.default_rng(seed)
    tick, bb = 0.01, 100.00
    rows, trades = [], []

    def row(ts, snap, side, p, a):
        rows.append(("binance-futures", "SYN", ts, ts + 2000, "true" if snap else "false",
                     side, round(p, 2), a))

    for i in range(5):
        row(T0, True, "bid", bb - i * tick, 5.0)
        row(T0, True, "ask", bb + (i + 1) * tick, 5.0)
    ts, tid = T0, 0
    for _ in range(n):
        ts += int(rng.integers(1_000, 400_000))
        ba = round(bb + tick, 2)
        r = rng.random()
        if r < 0.25:                                       # seller trades at/through bid
            q = float(rng.integers(1, 8))
            tid += 1
            trades.append(("binance-futures", "SYN", ts - trade_lag_us, ts + 1000, tid, "sell", bb, q))
            row(ts, False, "bid", bb, max(0.0, 5.0 - q) if q < 5 else 0.0)
            if q >= 5:                                     # level swept: price ticks down
                bb = round(bb - tick, 2)
                row(ts, False, "ask", bb + tick, 4.0)
                row(ts, False, "bid", bb - 4 * tick, 5.0)
        elif r < 0.5:                                      # buyer trades at ask
            q = float(rng.integers(1, 8))
            tid += 1
            trades.append(("binance-futures", "SYN", ts - trade_lag_us, ts + 1000, tid, "buy", ba, q))
            row(ts, False, "ask", ba, max(0.0, 5.0 - q) if q < 5 else 0.0)
            if q >= 5:
                bb = round(bb + tick, 2)
                row(ts, False, "bid", bb, 4.0)
                row(ts, False, "ask", bb + 5 * tick, 5.0)
        else:                                              # size changes (adds/cancels)
            side = "bid" if rng.random() < 0.5 else "ask"
            row(ts, False, side, bb if side == "bid" else ba, float(rng.integers(1, 12)))
    bp, tp = tmp_path / "book.csv.gz", tmp_path / "trades.csv.gz"
    pd.DataFrame(rows, columns=["exchange", "symbol", "timestamp", "local_timestamp",
                                "is_snapshot", "side", "price", "amount"]).to_csv(bp, index=False)
    pd.DataFrame(trades, columns=["exchange", "symbol", "timestamp", "local_timestamp",
                                  "id", "side", "price", "amount"]).to_csv(tp, index=False)
    return bp, tp, ts


SPECS = [AgentSpec("A", "A", latency_us=50_000), AgentSpec("B", "B", "prop", 50_000),
         AgentSpec("C", "C", latency_us=50_000)]


def test_chunked_reading_matches_unchunked(tmp_path):
    bp, _, _ = synthetic_day(tmp_path)
    whole = list(iter_book_messages(bp, chunksize=10**7))
    tiny = list(iter_book_messages(bp, chunksize=7))      # messages straddle chunk borders
    assert whole == tiny
    assert len(whole) > 1000


def test_no_lookahead_truncation(tmp_path):
    """Decisions and fills before a cutoff must not change when later data is removed."""
    bp, tp, t_end = synthetic_day(tmp_path)
    cutoff = T0 + (t_end - T0) // 2
    full = run_day(bp, tp, SPECS, q=2.0, qmax=20.0, record_decisions=True)
    cut = run_day(bp, tp, SPECS, q=2.0, qmax=20.0, until_ts=cutoff, record_decisions=True)
    for a_full, a_cut in zip(full.agents, cut.agents):
        assert [d for d in a_full.decision_log if d[0] <= cutoff] == a_cut.decision_log
        assert [f for f in a_full.fills if f[0] <= cutoff] == a_cut.fills
        assert len(a_cut.fills) > 0


def test_fill_ordering_and_identity_on_full_run(tmp_path):
    bp, tp, _ = synthetic_day(tmp_path, n=6000, seed=3)
    sim = run_day(bp, tp, SPECS, q=2.0, qmax=20.0)
    by = {a.name: a for a in sim.agents}
    n = {k: sum(f[3] for f in by[k].fills) for k in "ABC"}
    assert n["A"] >= n["B"] >= n["C"] > 0          # stricter models never fill more volume
    mid_t, mid_v, mid_seq = np.array(sim.mid_t), np.array(sim.mid_v), np.array(sim.mid_seq)
    for a in sim.agents:
        f = fills_to_arrays(a.fills)
        cash, pos, _ = cash_and_position(f)
        gross = cash + pos * mid_v[-1]
        assert spread_capture(f) + inventory_pnl(f, mid_t, mid_v, mid_seq) == pytest.approx(gross, abs=1e-6)
        # The limit is checked at decision time; orders already in flight can still fill,
        # so the bound is the limit plus in-flight orders (one on this sparse synthetic day).
        assert abs(a.pos) <= 20.0 + 2.0


@pytest.mark.parametrize("trades_first", [False, True])
def test_identity_exact_with_timestamp_ties_under_either_rule(tmp_path, trades_first):
    """With trades and book messages sharing timestamps, fills and mid changes can share a
    millisecond. Ordering by event sequence keeps the decomposition exact either way."""
    bp, tp, _ = synthetic_day(tmp_path, n=4000, seed=4, trade_lag_us=0)
    sim = run_day(bp, tp, SPECS, q=2.0, qmax=20.0, trades_first_on_ties=trades_first)
    df = summarise_day(sim, "d", "SYN")
    assert (df["fill_events"] > 0).all()
    assert df["identity_error"].max() < 1e-9


def _write(tmp_path, book_rows, trade_rows):
    bp, tp = tmp_path / "b.csv", tmp_path / "t.csv"
    pd.DataFrame([("x", "S", t, t, s, side, p, a) for t, s, side, p, a in book_rows],
                 columns=["exchange", "symbol", "timestamp", "local_timestamp", "is_snapshot",
                          "side", "price", "amount"]).to_csv(bp, index=False)
    pd.DataFrame([("x", "S", t, t, i, side, p, a) for i, (t, side, p, a) in enumerate(trade_rows)],
                 columns=["exchange", "symbol", "timestamp", "local_timestamp", "id",
                          "side", "price", "amount"]).to_csv(tp, index=False)
    return bp, tp


def test_tie_rule_hand_example(tmp_path):
    """Bid 100 x5. Our queue-aware bid (q = 2) joins behind 5 at t = 10.
    At t = 1000 a book message shows 100 -> 2 AND a sell of 3 at 100 is stamped t = 1000.

    trades first (pre-registered): the trade eats 3 of the 5 ahead (2 left, no fill); the
        message's fall of 3 is all explained by it, so no cancellations: 2 ahead.
    book first (as the data shows): the fall of 3 came BEFORE this trade, so it is
        cancellations, prop rule: 5 - 3 * 5/5 = 2 ahead. Then the 3-lot arrives: 2 clear
        the queue and 1 fills us.
    """
    book = [(0, "true", "bid", 100.0, 5.0), (0, "true", "ask", 101.0, 5.0),
            (1000, "false", "bid", 100.0, 2.0)]
    trades = [(1000, "sell", 100.0, 3.0)]
    bp, tp = _write(tmp_path, book, trades)
    spec = [AgentSpec("B", "B", "prop", latency_us=10)]
    first = run_day(bp, tp, spec, q=2.0, qmax=20.0, trades_first_on_ties=True).agents[0]
    after = run_day(bp, tp, spec, q=2.0, qmax=20.0, trades_first_on_ties=False).agents[0]
    assert len(first.fills) == 0
    assert [f[3] for f in after.fills] == [1.0]
