"""
accounting.py - inventory, cash, PnL, markouts. Pure functions of fills and mid prices.

Units: quantity in the base asset (e.g. LINK), prices and cash in USDT.

The key identity (exact, checked every run):

    gross PnL = spread capture + inventory PnL

    spread capture = sum over fills of  side * (mid at fill - fill price) * qty
                     what you earn versus the mid at the instant of the fill
    inventory PnL  = sum over mid changes of  position * change in mid
                     what holding the resulting position earned or lost afterwards

Proof sketch (summation by parts): position only changes at fills, so
sum(position * d mid) = final position * final mid - sum(side * qty * mid at fill).
Adding spread capture gives final position * final mid - sum(side * qty * price),
which is exactly cash plus marked inventory.

Adverse selection lives in the inventory term: a bid filled just before the mid falls
earns the half-spread in spread capture and gives it back, or more, in inventory PnL.

Ordering: "position during a mid change" must count exactly the fills that happened
before that change. Timestamps cannot decide this when a fill and a mid change share a
millisecond, so the simulator gives every event a sequence number and the functions
below take it (mid_seq, f["seq"]) when available. Without it they fall back to times,
counting a fill at the same time as a mid change as before it.
"""
import numpy as np

FILL_FIELDS = ("t", "side", "price", "qty", "mid", "through", "supported")


def fills_to_arrays(fills):
    """Column arrays from a simulate.FillLog or a list of fill tuples."""
    if hasattr(fills, "to_arrays"):
        return fills.to_arrays()
    if not fills:
        return {k: np.array([]) for k in FILL_FIELDS}
    arr = np.array(fills, dtype=float)
    out = {k: arr[:, i] for i, k in enumerate(FILL_FIELDS)}
    out["t"] = arr[:, 0].astype(np.int64)
    out["through"] = arr[:, 5].astype(bool)
    return out


def cash_and_position(f, fee_bps=0.0):
    notional = f["price"] * f["qty"]
    cash = -np.sum(f["side"] * notional)
    fees = fee_bps * 1e-4 * np.sum(notional)
    pos = np.sum(f["side"] * f["qty"])
    return cash, pos, fees


def spread_capture(f):
    return float(np.sum(f["side"] * (f["mid"] - f["price"]) * f["qty"]))


def position_at(f, at, key="t"):
    """Position after all fills whose f[key] (time, or event sequence) is <= each value."""
    if len(f["t"]) == 0:
        return np.zeros(len(at))
    cum = np.cumsum(f["side"] * f["qty"])
    idx = np.searchsorted(f[key], at, side="right") - 1
    return np.where(idx >= 0, cum[np.clip(idx, 0, None)], 0.0)


def _order(f, mid_t, mid_seq):
    """Which keys order fills against mid changes: sequence numbers if both have them."""
    if mid_seq is not None and "seq" in f:
        return np.asarray(mid_seq), "seq"
    return np.asarray(mid_t), "t"


def inventory_pnl(f, mid_t, mid_v, mid_seq=None):
    if len(mid_v) < 2:
        return 0.0
    at, key = _order(f, mid_t, mid_seq)
    pos = position_at(f, at[1:], key)
    return float(np.sum(pos * np.diff(mid_v)))


def realised_unrealised(f, mark):
    """Average-cost method. Returns (realised, unrealised); they sum to gross PnL."""
    pos = avg = realised = 0.0
    for s, p, q in zip(f["side"], f["price"], f["qty"]):
        d = s * q
        if pos == 0 or np.sign(pos) == np.sign(d):
            new = pos + d
            avg = (avg * abs(pos) + p * q) / abs(new)
            pos = new
        else:
            close = min(abs(pos), q)
            realised += close * (p - avg) * np.sign(pos)
            pos += d
            if abs(pos) < 1e-12:
                pos, avg = 0.0, 0.0
            elif close < q:                 # flipped through zero: new position at p
                avg = p
    return realised, pos * (mark - avg)


def equity_curve(f, mid_t, mid_v, fee_bps=0.0, mid_seq=None):
    """Marked-to-mid equity at every mid change (fees deducted as incurred)."""
    if len(mid_t) == 0:
        return np.array([]), np.array([])
    at, key = _order(f, mid_t, mid_seq)
    pos = position_at(f, at, key)
    if len(f["t"]):
        cash_cum = np.cumsum(-f["side"] * f["price"] * f["qty"]
                             - fee_bps * 1e-4 * f["price"] * f["qty"])
        idx = np.searchsorted(f[key], at, side="right") - 1
        cash = np.where(idx >= 0, cash_cum[np.clip(idx, 0, None)], 0.0)
    else:
        cash = np.zeros(len(mid_t))
    return np.asarray(mid_t), cash + pos * np.asarray(mid_v)


def max_drawdown(equity):
    if len(equity) == 0:
        return 0.0
    peak = np.maximum.accumulate(equity)
    return float(np.min(equity - peak))


def markouts(f, mid_t, mid_v, horizon_us, end_ts):
    """Signed markout per fill in bps of the mid at fill. NaN beyond the end of data.

    side * (mid(t + h) - mid at fill): positive means the price moved in our favour
    after the fill, negative means adverse selection.
    """
    if len(f["t"]) == 0:
        return np.array([])
    tgt = f["t"] + horizon_us
    idx = np.searchsorted(mid_t, tgt, side="right") - 1
    m_h = np.asarray(mid_v)[np.clip(idx, 0, None)]
    mk = f["side"] * (m_h - f["mid"]) / f["mid"] * 1e4
    return np.where(tgt <= end_ts, mk, np.nan)


def weighted_mean(x, w):
    m = ~np.isnan(x)
    return float(np.sum(x[m] * w[m]) / np.sum(w[m])) if m.any() and np.sum(w[m]) > 0 else float("nan")
