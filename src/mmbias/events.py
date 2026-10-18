"""
events.py - stream Tardis CSVs without loading whole days into memory.

A book "message" is a group of consecutive rows sharing one (timestamp, local_timestamp)
pair: one exchange WebSocket message can update several price levels at once, and the
book is only internally consistent after all of them are applied. Splitting on a change
in EITHER timestamp matters: on real Binance data two exchange messages occasionally
arrive in the same local microsecond.

Reading in chunks creates one subtlety: a message can straddle a chunk boundary. The
last (possibly incomplete) message of each chunk is carried into the next chunk, so
every message is yielded whole. tests/test_integration.py checks that chunked and
unchunked reading give identical messages.

Trades are streamed the same way (iter_trades). Memory is bounded by the chunk size,
not by the length of the day: a busy contract has several million trades per day.
"""
import numpy as np
import pandas as pd

BOOK_COLS = ["timestamp", "local_timestamp", "is_snapshot", "side", "price", "amount"]
BOOK_DTYPES = {"timestamp": "int64", "local_timestamp": "int64", "is_snapshot": "category",
               "side": "category", "price": "float64", "amount": "float64"}
TRADE_COLS = ["timestamp", "side", "price", "amount"]
TRADE_DTYPES = {"timestamp": "int64", "side": "category", "price": "float64", "amount": "float64"}


def _arrays(df):
    snap = df["is_snapshot"].isin(["true", "True", "TRUE"]).to_numpy()
    bid = (df["side"] == "bid").to_numpy()
    return (df["timestamp"].to_numpy(), df["local_timestamp"].to_numpy(), snap, bid,
            df["price"].to_numpy(), df["amount"].to_numpy())


def iter_book_messages(path, chunksize=1_000_000):
    """Yield (ts, local_ts, snap_list, bid_list, price_list, amount_list) per message."""
    carry = None
    reader = pd.read_csv(path, usecols=BOOK_COLS, dtype=BOOK_DTYPES, chunksize=chunksize)
    for chunk in reader:
        arrs = _arrays(chunk)
        if carry is not None:
            arrs = tuple(np.concatenate([c, a]) for c, a in zip(carry, arrs))
        ts, lts = arrs[0], arrs[1]
        n = len(ts)
        new = np.ones(n, dtype=bool)
        new[1:] = (lts[1:] != lts[:-1]) | (ts[1:] != ts[:-1])
        starts = np.flatnonzero(new)
        last = int(starts[-1])
        ts_l, lts_l = ts.tolist(), lts.tolist()
        snap_l, bid_l, px_l, amt_l = (a.tolist() for a in arrs[2:])
        for k in range(len(starts) - 1):
            i0, i1 = int(starts[k]), int(starts[k + 1])
            yield ts_l[i0], lts_l[i0], snap_l[i0:i1], bid_l[i0:i1], px_l[i0:i1], amt_l[i0:i1]
        carry = tuple(a[last:] for a in arrs)
    if carry is not None and len(carry[0]):
        ts, lts, snap, bid, px, amt = carry
        yield (int(ts[0]), int(lts[0]), snap.tolist(), bid.tolist(), px.tolist(), amt.tolist())


def iter_trades(path, chunksize=1_000_000, stats=None):
    """Yield (ts, price, qty, aggressor) per trade, in file order, one chunk in memory.

    aggressor: +1 buyer-initiated, -1 seller-initiated, 0 unknown.

    Tardis writes trades in arrival order, which on Binance is also exchange-timestamp
    order. A trade whose exchange timestamp is earlier than one already yielded cannot
    be moved back in time without buffering the whole file, so it is yielded at the
    latest timestamp seen so far and counted in stats["trades_out_of_order"]. The count
    is reported for every day; it is zero on all data examined.
    """
    stats = {} if stats is None else stats
    stats.setdefault("trades_out_of_order", 0)
    last = None
    for ch in pd.read_csv(path, usecols=TRADE_COLS, dtype=TRADE_DTYPES, chunksize=chunksize):
        ts = ch["timestamp"].to_numpy()
        side = ch["side"]
        aggr = np.where(side == "buy", 1, np.where(side == "sell", -1, 0)).tolist()
        if last is not None:
            ts = np.concatenate([[last], ts])
        run_max = np.maximum.accumulate(ts)
        stats["trades_out_of_order"] += int(np.sum(ts < run_max))
        if last is not None:
            run_max = run_max[1:]
        last = int(run_max[-1])
        yield from zip(run_max.tolist(), ch["price"].tolist(), ch["amount"].tolist(), aggr)
