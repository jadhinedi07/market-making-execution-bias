"""
validate.py - checks of the data and of the engine's event-ordering assumption.

Every check streams the files once, as the simulator does, and none uses strategy results.

1. Reconstruction vs Tardis book_snapshot_5. The book rebuilt from incremental_book_L2 is
   compared with Tardis's own top-5 snapshot rows (best level at every row, all five
   levels on a sample). Tardis derives those rows from the same messages, so this checks
   OUR code (snapshot handling, deletions, best-price heap, message grouping), not the
   exchange feed.

2. Drift at re-snapshots. When the feed reconnects, a fresh exchange snapshot arrives.
   The incrementally rebuilt book just before it is compared with the snapshot. This is
   the independent check: errors from missed or misapplied updates accumulate in the
   incremental book and show up as differences here (so do real changes during the gap,
   which is reported alongside).

3. Trade/book alignment. The simulator assumes a trade's effect on the book appears in
   the first book message whose exchange timestamp is at or after the trade's (trades
   before book updates on equal timestamps). If that is right, the volume traded at a
   price between two messages should show up as a fall in displayed size at that price
   in the SAME message window ("lag 0"), much more than in the message before (lag -1)
   or after (lag +1). Trades whose timestamp equals the message's are tested separately,
   because the tie rule is the part of the assumption most likely to be wrong.

4. Trade prices against the reconstructed touch, under the simulator's ordering (book
   state before the message at or after the trade) and the alternative (state after it).
   A buy below the best ask, or a sell above the best bid, is inconsistent with the
   reconstructed book at that moment.
"""
import heapq

import numpy as np
import pandas as pd

from .book import OrderBook
from .events import iter_book_messages, iter_trades


def _iter_rows(path, cols, chunksize=250_000):
    for ch in pd.read_csv(path, usecols=cols, chunksize=chunksize):
        yield from ch[cols].itertuples(index=False, name=None)


def _top(side_dict, n, is_bid):
    keys = heapq.nlargest(n, side_dict) if is_bid else heapq.nsmallest(n, side_dict)
    return [(p, side_dict[p]) for p in keys]


def run_checks(book_path, trades_path, snap5_path=None, chunksize=250_000, sample_every=50,
               tick=None):
    """One streaming pass over book + trades (+ optional book_snapshot_5). Returns a dict."""
    book = OrderBook()
    started, prev_snap = False, False
    out = dict(messages=0, max_gap_ms=0.0, gaps_over_1s=0)

    # --- 1. snapshot_5 comparison
    s5 = None
    if snap5_path is not None:
        lv = [f"{s}[{i}].{f}" for i in range(5) for s in ("bids", "asks") for f in ("price", "amount")]
        s5_cols = ["timestamp", "local_timestamp"] + lv
        s5 = _iter_rows(snap5_path, s5_cols, chunksize)
        s5_row = next(s5, None)
        c1 = dict(rows=0, unmatched=0, top_checked=0, top_mismatch=0, deep_checked=0, deep_mismatch=0)

    # --- 2. resnapshot drift
    drift = []

    # --- 3. alignment: traded volume per (is_bid_level, price) in the current window
    win, win_tie = {}, {}             # trades with ts < E, and trades with ts == E
    prev_win, prev_win_tie = {}, {}
    prev_dep = {}
    match = {k: 0.0 for k in ("lag-1", "lag0", "lag+1", "tie_lag0", "tie_lag+1")}
    vol = dict(total=0.0, tie=0.0)

    # --- 4. trade prices vs touch
    c4 = dict(trades=0, no_book=0, inside_before=0, inside_after=0, through_before=0)
    pending_trades = []               # (price, aggr) of trades in this window, for the "after" test

    trades = iter_trades(trades_path, chunksize)
    nxt = next(trades, None)
    last_ts = None

    def matched(window, dep):
        return sum(min(v, dep.get(k, 0.0)) for k, v in window.items())

    for ts, lts, snap, bid, px, amt in iter_book_messages(book_path, chunksize):
        # trades up to and including this message's timestamp form its window
        bb, ba = (book.best_bid(), book.best_ask()) if started else (None, None)
        while nxt is not None and nxt[0] <= ts:
            t, p, q, a = nxt
            if a != 0 and started:
                key = (a == -1, p)                       # a sell hits bids
                d = win_tie if t == ts else win
                d[key] = d.get(key, 0.0) + q
                c4["trades"] += 1
                if bb is None or ba is None:
                    c4["no_book"] += 1
                else:
                    if (a == 1 and p < ba) or (a == -1 and p > bb):
                        c4["inside_before"] += 1
                    if (a == 1 and p > ba) or (a == -1 and p < bb):
                        c4["through_before"] += 1
                    pending_trades.append((p, a))
            nxt = next(trades, None)

        out["messages"] += 1
        gap = (ts - last_ts) / 1000 if last_ts is not None else 0.0
        out["max_gap_ms"] = max(out["max_gap_ms"], gap)
        out["gaps_over_1s"] += gap > 1000
        last_ts = ts

        dep = {}
        for s, b, p, a in zip(snap, bid, px, amt):
            if s and not prev_snap:
                if started:                              # re-snapshot: measure drift first
                    drift.append(dict(ts=ts, gap_ms=gap, before_bb=book.best_bid(), before_ba=book.best_ask(),
                                      top_before={True: _top(book.bids, 5, True),
                                                  False: _top(book.asks, 5, False)}))
                started = True
                book.reset()
            prev_snap = s
            if not started:
                continue
            old = book.apply(b, p, a)
            if not s and old > a:
                dep[(b, p)] = dep.get((b, p), 0.0) + (old - a)
        if not started:
            continue
        if drift and "after_bb" not in drift[-1]:
            d = drift[-1]
            d["after_bb"], d["after_ba"] = book.best_bid(), book.best_ask()
            same = 0
            for side in (True, False):
                after = dict(_top(book.bids if side else book.asks, 5, side))
                same += sum(1 for p, a in d["top_before"][side] if after.get(p) == a)
            d["top5_levels_identical"] = same

        # alignment bookkeeping
        tot = sum(win.values()) + sum(win_tie.values())
        vol["total"] += tot
        vol["tie"] += sum(win_tie.values())
        both = dict(win)
        for k, v in win_tie.items():
            both[k] = both.get(k, 0.0) + v
        match["lag0"] += matched(both, dep)
        match["lag-1"] += matched(both, prev_dep)         # book moved BEFORE the trades
        match["tie_lag0"] += matched(win_tie, dep)
        # the previous window's trades against THIS message: trades reflected one late
        match["lag+1"] += matched(prev_win, dep)
        match["tie_lag+1"] += matched(prev_win_tie, dep)
        prev_win, prev_win_tie, prev_dep = both, win_tie, dep
        win, win_tie = {}, {}

        bb2, ba2 = book.best_bid(), book.best_ask()
        if bb2 is not None and ba2 is not None:
            for p, a in pending_trades:
                if (a == 1 and p < ba2) or (a == -1 and p > bb2):
                    c4["inside_after"] += 1
        pending_trades = []

        # snapshot_5 comparison for rows keyed at this message
        if s5 is not None:
            while s5_row is not None and (s5_row[0], s5_row[1]) < (ts, lts):
                c1["unmatched"] += 1
                s5_row = next(s5, None)
            while s5_row is not None and (s5_row[0], s5_row[1]) == (ts, lts):
                c1["rows"] += 1
                r = s5_row[2:]
                c1["top_checked"] += 1
                if (r[0], r[1], r[2], r[3]) != (bb2, book.bids.get(bb2), ba2, book.asks.get(ba2)):
                    c1["top_mismatch"] += 1
                if c1["rows"] % sample_every == 0:
                    c1["deep_checked"] += 1
                    mine = []
                    tb, ta = _top(book.bids, 5, True), _top(book.asks, 5, False)
                    for i in range(5):
                        mine += [*(tb[i] if i < len(tb) else (np.nan, np.nan)),
                                 *(ta[i] if i < len(ta) else (np.nan, np.nan))]
                    if not np.allclose(np.array(mine, float), np.array(r, float), equal_nan=True):
                        c1["deep_mismatch"] += 1
                s5_row = next(s5, None)

    # trailing trades after the last message are not part of any window
    tv = max(vol["total"], 1e-12)
    res = dict(out)
    res.update({f"align_{k}_pct": 100 * v / tv for k, v in match.items() if not k.startswith("tie")})
    res["tie_volume_pct"] = 100 * vol["tie"] / tv
    tvt = max(vol["tie"], 1e-12)
    res["align_tie_lag0_pct"] = 100 * match["tie_lag0"] / tvt
    res["align_tie_lag+1_pct"] = 100 * match["tie_lag+1"] / tvt
    nt = max(c4["trades"] - c4["no_book"], 1)
    res.update(trades_checked=c4["trades"], inside_before_pct=100 * c4["inside_before"] / nt,
               inside_after_pct=100 * c4["inside_after"] / nt,
               through_before_pct=100 * c4["through_before"] / nt)
    if s5 is not None:
        while s5_row is not None:
            c1["unmatched"] += 1
            s5_row = next(s5, None)
        res.update({f"snap5_{k}": v for k, v in c1.items()})
    res["resnapshots"] = len(drift)
    res["drift"] = [dict(ts=d["ts"], gap_ms=d["gap_ms"], best_bid_same=d["before_bb"] == d.get("after_bb"),
                         best_ask_same=d["before_ba"] == d.get("after_ba"),
                         top5_levels_identical_of_10=d.get("top5_levels_identical"))
                    for d in drift]
    return res
