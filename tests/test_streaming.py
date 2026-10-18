"""
Streaming: reading in chunks must give exactly what reading whole files gives.

The engine never holds a whole day in memory, so every place where a chunk boundary
could matter (a book message split across chunks, a trade merge across chunks, a quote
whose duration needs the next chunk's timestamp) is checked against a whole-file result.
Synthetic files only.
"""
import numpy as np
import pandas as pd
import pytest

from mmbias.data import quote_stats, trade_stats
from mmbias.events import iter_book_messages, iter_trades
from mmbias.simulate import AgentSpec, FillLog, run_day

from test_integration import SPECS, T0, synthetic_day


def test_trades_chunked_equal_unchunked(tmp_path):
    _, tp, _ = synthetic_day(tmp_path)
    whole = list(iter_trades(tp, chunksize=10**7))
    tiny = list(iter_trades(tp, chunksize=3))
    assert whole == tiny and len(whole) > 500
    df = pd.read_csv(tp)
    assert [t[0] for t in whole] == df["timestamp"].tolist()
    assert [t[3] for t in whole] == [1 if s == "buy" else -1 for s in df["side"]]


def test_out_of_order_trade_is_counted_and_clamped(tmp_path):
    p = tmp_path / "t.csv"
    pd.DataFrame(dict(timestamp=[10, 20, 15, 30], side=["buy", "sell", "buy", "sell"],
                      price=[1.0, 1.0, 1.0, 1.0], amount=[1.0, 2.0, 3.0, 4.0])).to_csv(p, index=False)
    for cs in (1, 2, 100):                           # the bad row on and off a boundary
        st = {}
        out = list(iter_trades(p, chunksize=cs, stats=st))
        assert [t[0] for t in out] == [10, 20, 20, 30]
        assert st["trades_out_of_order"] == 1


def test_messages_split_on_exchange_timestamp_change(tmp_path):
    # Two exchange messages (ts 100 and 101) arriving in the same local microsecond.
    p = tmp_path / "b.csv"
    pd.DataFrame(dict(timestamp=[100, 100, 101, 102], local_timestamp=[5, 5, 5, 6],
                      is_snapshot=["false"] * 4, side=["bid", "ask", "bid", "ask"],
                      price=[1.0, 2.0, 1.1, 2.1], amount=[1.0, 1.0, 1.0, 1.0])).to_csv(p, index=False)
    msgs = list(iter_book_messages(p, chunksize=2))
    assert [(m[0], len(m[2])) for m in msgs] == [(100, 2), (101, 1), (102, 1)]


def test_simulation_identical_for_any_chunk_size(tmp_path):
    bp, tp, _ = synthetic_day(tmp_path, n=3000, seed=5)
    big = run_day(bp, tp, SPECS, q=2.0, qmax=20.0, chunksize=10**7)
    small = run_day(bp, tp, SPECS, q=2.0, qmax=20.0, chunksize=17)
    for a, b in zip(big.agents, small.agents):
        assert a.fills == b.fills and len(a.fills) > 0
        assert (a.n_place, a.n_cancel, a.n_reject) == (b.n_place, b.n_cancel, b.n_reject)
    assert list(big.mid_v) == list(small.mid_v)


def test_fill_log_behaves_like_a_list_of_tuples():
    fl, ref = FillLog(), []
    for row in [(1, 1, 100.0, 2.0, 100.5, False, 0.0), (2, -1, 101.0, 1.0, 100.5, True, 1.0)]:
        fl.append(*row)
        ref.append(row)
    assert len(fl) == 2 and fl == ref and list(fl) == ref
    assert fl[-1] == ref[-1] and fl[-1][5] is True
    arr = fl.to_arrays()
    assert arr["t"].dtype == np.int64 and arr["through"].tolist() == [False, True]


def _quotes_file(tmp_path, n=5000, seed=0):
    rng = np.random.default_rng(seed)
    ts = T0 + np.cumsum(rng.integers(1, 500_000, n))
    bid = 100.0 + 0.01 * np.cumsum(rng.integers(-1, 2, n))
    spread_ticks = rng.choice([1, 1, 1, 2, 3], n)
    df = pd.DataFrame(dict(exchange="x", symbol="SYN", timestamp=ts, local_timestamp=ts + 5,
                           ask_amount=rng.integers(1, 50, n).astype(float),
                           ask_price=np.round(bid + 0.01 * spread_ticks, 2),
                           bid_price=np.round(bid, 2),
                           bid_amount=rng.integers(1, 50, n).astype(float)))
    p = tmp_path / "quotes.csv.gz"
    df.to_csv(p, index=False)
    return p, df


def test_quote_stats_chunked_equal_whole_file(tmp_path):
    p, df = _quotes_file(tmp_path)
    # Brute force, whole file in memory: the original definition.
    w = np.append(np.diff(df["timestamp"].to_numpy()), 0).astype(float)
    size = ((df["bid_amount"] + df["ask_amount"]) / 2).to_numpy()
    order = np.argsort(size, kind="stable")
    cw = np.cumsum(w[order])
    med = size[order][np.searchsorted(cw, cw[-1] / 2)]
    spr = (df["ask_price"] - df["bid_price"]).to_numpy()
    wide = w[spr > 1.5 * 0.01].sum() / w.sum()
    for cs in (7, 1000, 10**6):
        st = quote_stats(p, chunksize=cs)
        assert st["median_best_size"] == pytest.approx(med)
        assert st["tick"] == pytest.approx(0.01)
        assert st["share_time_spread_gt_1tick"] == pytest.approx(wide)


def test_trade_stats_chunked_equal_whole_file(tmp_path):
    _, tp, _ = synthetic_day(tmp_path, n=6000, seed=2)
    df = pd.read_csv(tp)
    day0 = int(df["timestamp"].iloc[0]) - int(df["timestamp"].iloc[0]) % 86_400_000_000
    minute = ((df["timestamp"] - day0) // 60_000_000).clip(0, 1439).to_numpy()
    per_min = np.bincount(minute, weights=df["amount"].to_numpy(), minlength=1440)
    for cs in (5, 999, 10**6):
        st = trade_stats(tp, chunksize=cs)
        assert st["median_trade_size"] == df["amount"].median()
        assert st["median_amount_per_min"] == np.median(per_min)
        assert st["qty_step"] == pytest.approx(1.0)
        assert st["notional"] == pytest.approx((df["price"] * df["amount"]).sum())
        assert st["n_trades"] == len(df)


def test_trade_stats_even_count_median(tmp_path):
    p = tmp_path / "t.csv"
    pd.DataFrame(dict(timestamp=[T0, T0 + 1, T0 + 2, T0 + 3], side="buy", price=1.0,
                      amount=[1.0, 2.0, 4.0, 8.0])).to_csv(p, index=False)
    assert trade_stats(p, chunksize=3)["median_trade_size"] == 3.0
