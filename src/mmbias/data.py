"""
data.py - download Tardis CSVs with a manifest, and apply the contract-selection rule.

Every download is recorded in data/manifest.json: URL, HTTP status, bytes, SHA-256 and
UTC time. Raw data is never committed to git (see .gitignore): anyone reproducing the
study downloads the same public files with this module.
"""
import hashlib
import json
import os
import time
import urllib.request

import numpy as np
import pandas as pd

BASE = "https://datasets.tardis.dev/v1/binance-futures"
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")
MANIFEST = os.path.join(DATA_DIR, "manifest.json")


def url(dtype, date, symbol):
    y, m, d = date.split("-")
    return f"{BASE}/{dtype}/{y}/{m}/{d}/{symbol}.csv.gz"


def local_path(dtype, date, symbol):
    return os.path.join(DATA_DIR, "raw", symbol, f"{date}_{dtype}.csv.gz")


def _log(entry):
    os.makedirs(DATA_DIR, exist_ok=True)
    log = json.load(open(MANIFEST)) if os.path.exists(MANIFEST) else []
    log.append(entry)
    json.dump(log, open(MANIFEST, "w"), indent=1)


def _utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def remote_size(u):
    """HTTP status and Content-Length without downloading the body.

    datasets.tardis.dev answers HEAD requests with 404 even for files that exist, so a
    GET is opened, its headers read, and the connection closed before the body.
    """
    try:
        with urllib.request.urlopen(u, timeout=60) as r:
            return r.status, int(r.headers.get("Content-Length") or -1)
    except urllib.error.HTTPError as e:
        return e.code, -1


def download(dtype, date, symbol, retries=4):
    """Download one file (skipping if already present). Returns the path.

    The MD5 that Tardis sends in its x-md5 header is checked against the bytes received,
    as is Content-Length; a mismatch is a failed attempt. Network failures are retried
    with exponential backoff. Every attempt, failed or not, is written to the manifest.
    """
    path, u = local_path(dtype, date, symbol), url(dtype, date, symbol)
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    for attempt in range(retries + 1):
        sha, md5, n, t0 = hashlib.sha256(), hashlib.md5(), 0, time.time()
        try:
            with urllib.request.urlopen(u, timeout=600) as r, open(path + ".part", "wb") as out:
                expect_md5 = (r.headers.get("x-md5") or "").strip('"')
                expect_len = int(r.headers.get("Content-Length") or -1)
                while True:
                    block = r.read(1 << 20)
                    if not block:
                        break
                    out.write(block)
                    sha.update(block)
                    md5.update(block)
                    n += len(block)
                status = r.status
            if expect_len >= 0 and n != expect_len:
                raise IOError(f"received {n} bytes, Content-Length {expect_len}")
            if expect_md5 and md5.hexdigest() != expect_md5:
                raise IOError(f"md5 {md5.hexdigest()} != x-md5 {expect_md5}")
            os.replace(path + ".part", path)
            _log(dict(url=u, status=status, bytes=n, sha256=sha.hexdigest(), md5_verified=bool(expect_md5),
                      seconds=round(time.time() - t0, 1), utc=_utc()))
            return path
        except urllib.error.HTTPError as e:     # 4xx/5xx: the file is not there; do not retry 4xx
            _log(dict(url=u, status=e.code, bytes=n, utc=_utc()))
            if e.code < 500 or attempt == retries:
                raise
        except Exception as e:
            _log(dict(url=u, status=f"{type(e).__name__}: {e}", bytes=n, utc=_utc()))
            if attempt == retries:
                raise
        time.sleep(2 ** (attempt + 1))


# ---------------------------------------------------------------- selection rule
# Both statistics below are computed exactly but in chunks: a weighted median only needs
# the total weight per distinct value, and a day has far fewer distinct sizes, prices and
# spreads than rows. Memory therefore stays bounded by the chunk size.

def _add(acc, values, weights):
    s = pd.Series(weights, index=values).groupby(level=0).sum()
    return s if acc is None else acc.add(s, fill_value=0.0)


def weighted_median(weight_by_value):
    """Lower weighted median: smallest value whose cumulative weight reaches half."""
    s = weight_by_value.sort_index()
    cw = s.to_numpy().cumsum()
    return float(s.index[np.searchsorted(cw, cw[-1] / 2)])


def quote_stats(quotes_path, chunksize=1_000_000):
    """Time-weighted median of mean best-level size, and share of time spread > 1 tick.

    Each quote state is weighted by how long it lasted (until the next quote). The last
    row of a chunk is carried into the next one, because its duration needs the next
    timestamp.
    """
    cols = ["timestamp", "ask_amount", "ask_price", "bid_price", "bid_amount"]
    w_size = w_spread = None
    prices = set()
    carry = None
    for ch in pd.read_csv(quotes_path, usecols=cols, chunksize=chunksize):
        ch = ch.dropna()
        ts = ch["timestamp"].to_numpy(np.int64)
        size = ((ch["bid_amount"] + ch["ask_amount"]) / 2).to_numpy(np.float64)
        spr = np.round((ch["ask_price"] - ch["bid_price"]).to_numpy(np.float64), 10)
        prices.update(np.round(ch["bid_price"].to_numpy(np.float64), 10).tolist())
        if carry is not None:
            ts, size, spr = (np.concatenate([[c], a]) for c, a in zip(carry, (ts, size, spr)))
        if len(ts) < 2:
            carry = (ts[-1], size[-1], spr[-1]) if len(ts) else carry
            continue
        w = np.diff(ts).astype(float)                       # how long each state lasted
        w_size = _add(w_size, size[:-1], w)
        w_spread = _add(w_spread, spr[:-1], w)
        carry = (ts[-1], size[-1], spr[-1])
    # Tick = the most common gap between adjacent distinct prices. The smallest gap is not
    # robust: the real feed contains a handful of off-grid prices (e.g. SOLUSDT bid 208.313
    # on a 0.01 grid on 2025-10-01, 11 of 1.26M quotes).
    gaps, counts = np.unique(np.round(np.diff(np.array(sorted(prices))), 10), return_counts=True)
    tick = float(gaps[np.argmax(counts)])
    wide = float(w_spread[w_spread.index > 1.5 * tick].sum() / w_spread.sum())
    return dict(median_best_size=weighted_median(w_size), tick=tick, share_time_spread_gt_1tick=wide)


def trade_stats(trades_path, chunksize=1_000_000):
    """Median traded amount per minute, median trade size, quantity step, notional."""
    per_min = np.zeros(1440)                # UTC clock minutes of the file's day
    counts = None
    day0 = None
    notional, n = 0.0, 0
    for ch in pd.read_csv(trades_path, usecols=["timestamp", "price", "amount"], chunksize=chunksize):
        ts, px, amt = (ch[c].to_numpy() for c in ("timestamp", "price", "amount"))
        if day0 is None:
            day0 = int(ts[0]) - int(ts[0]) % 86_400_000_000
        minute = np.clip((ts - day0) // 60_000_000, 0, 1439)
        per_min += np.bincount(minute, weights=amt, minlength=1440)
        counts = _add(counts, np.round(amt.astype(np.float64), 10), np.ones(len(amt)))
        notional += float((px * amt).sum())
        n += len(amt)
    sizes = counts.sort_index()
    step = float(np.min(np.diff(sizes.index.to_numpy()))) if len(sizes) > 1 else float(sizes.index[0])
    # Exact median of trade sizes from the per-size counts (average of the two middle
    # values when n is even, as pandas' median would give).
    cum = sizes.to_numpy().cumsum()
    lo = float(sizes.index[np.searchsorted(cum, (n - 1) // 2 + 1)])
    hi = float(sizes.index[np.searchsorted(cum, n // 2 + 1)])
    return dict(median_amount_per_min=float(np.median(per_min)), median_trade_size=(lo + hi) / 2,
                qty_step=step, notional=notional, n_trades=n)


def select_contract(candidates, date, max_mb=300):
    """Apply the pre-registered rule in research_plan.md. Returns (choice, table)."""
    rows = []
    for s in candidates:
        status, nbytes = remote_size(url("incremental_book_L2", date, s))
        r = dict(symbol=s, book_status=status, book_mb=round(nbytes / 2**20, 1))
        if status == 200 and 0 < nbytes <= max_mb * 2**20:
            r.update(quote_stats(download("quotes", date, s)))
            r.update(trade_stats(download("trades", date, s)))
            r["queue_turnover_min"] = r["median_best_size"] / max(r["median_amount_per_min"], 1e-12)
        rows.append(r)
    table = pd.DataFrame(rows)
    ok = table.dropna(subset=["queue_turnover_min"]) if "queue_turnover_min" in table else table.iloc[0:0]
    if ok.empty:
        return None, table
    best = ok.sort_values(["queue_turnover_min", "notional"], ascending=[True, False]).iloc[0]
    return best["symbol"], table


def order_size(median_trade_size, step):
    """Median trade size rounded DOWN to the quantity step, at least one step."""
    return max(step, np.floor(median_trade_size / step + 1e-9) * step)
