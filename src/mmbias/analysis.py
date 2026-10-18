"""
analysis.py - turn one simulated day into metrics, and many days into comparisons.

Markouts are size-weighted means in basis points of the mid at fill. The statistical
unit for comparisons is the DAY: fills within one day share one market path, so they
are not independent observations.
"""
import numpy as np
import pandas as pd

from .accounting import (cash_and_position, equity_curve, fills_to_arrays, inventory_pnl,
                         markouts, max_drawdown, position_at, realised_unrealised,
                         spread_capture, weighted_mean)

HORIZONS_S = (1, 10, 60)                     # pre-registered markout horizons
CURVE_HORIZONS_S = (0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60, 120, 300)   # descriptive only
FEES_BPS = (0.0, 1.0, 2.0)


def time_weighted_abs_position(f, mid_t, end_ts):
    if len(f["t"]) == 0:
        return 0.0
    t = np.concatenate([f["t"], [end_ts]])
    pos = np.cumsum(f["side"] * f["qty"])
    dur = np.diff(t).astype(float)
    total = end_ts - mid_t[0]
    return float(np.sum(np.abs(pos) * dur) / total) if total > 0 else 0.0


def share_time_at_limit(f, mid_t, end_ts, qmax, eps=1e-9):
    """Fraction of the day with |position| at or beyond the inventory limit."""
    if len(f["t"]) == 0:
        return 0.0
    t = np.concatenate([f["t"], [end_ts]])
    pos = np.cumsum(f["side"] * f["qty"])
    dur = np.diff(t).astype(float)
    total = end_ts - mid_t[0]
    return float(np.sum(dur[np.abs(pos) >= qmax - eps]) / total) if total > 0 else 0.0


def summarise_agent(ag, sim):
    mid_t, mid_v, mid_seq = np.asarray(sim.mid_t), np.asarray(sim.mid_v), np.asarray(sim.mid_seq)
    end_ts, mid_end = sim.stats["last_ts"], mid_v[-1]
    f = fills_to_arrays(ag.fills)
    cash, pos, _ = cash_and_position(f)
    gross = cash + pos * mid_end
    sc = spread_capture(f)
    inv = inventory_pnl(f, mid_t, mid_v, mid_seq)
    real, unreal = realised_unrealised(f, mid_end) if len(f["t"]) else (0.0, 0.0)
    liq_px = sim.bb if pos > 0 else sim.ba
    _, eq = equity_curve(f, mid_t, mid_v, mid_seq=mid_seq)
    qty = f["qty"]
    out = dict(agent=ag.name, model=ag.model, qrule=ag.qrule, latency_ms=ag.lat / 1000,
               orders_placed=ag.n_place, orders_live=ag.n_live, orders_rejected=ag.n_reject,
               cancels=ag.n_cancel, orders_full=ag.n_full, orders_partial=ag.n_partial,
               fill_events=len(qty), filled_qty=float(qty.sum()),
               through_qty=float(qty[f["through"]].sum()) if len(qty) else 0.0,
               qty_filled_at_activation_ms=ag.qty_at_activation,
               filled_notional=float((f["price"] * qty).sum()),
               fill_rate=(ag.n_full + ag.n_partial) / ag.n_live if ag.n_live else float("nan"),
               gross_pnl=gross, spread_capture=sc, inventory_pnl=inv,
               identity_error=abs(sc + inv - gross),
               realised=real, unrealised=unreal,
               gross_pnl_liquidated=cash + pos * liq_px if liq_px is not None else float("nan"),
               end_position=pos, max_abs_position=ag.max_abs_pos,
               mean_abs_position=time_weighted_abs_position(f, mid_t, end_ts),
               share_time_at_limit=share_time_at_limit(f, mid_t, end_ts, ag.qmax),
               max_drawdown_gross=max_drawdown(eq),
               eff_spread_bps=weighted_mean(f["side"] * (f["mid"] - f["price"]) / f["mid"] * 1e4, qty)
               if len(qty) else float("nan"))
    for fee in FEES_BPS:
        _, _, fees = cash_and_position(f, fee)
        out[f"net_pnl_{fee:g}bp"] = gross - fees
    for h in CURVE_HORIZONS_S:
        mk = markouts(f, mid_t, mid_v, int(h * 1_000_000), end_ts)
        out[f"mkcurve_{h:g}s_bps"] = weighted_mean(mk, qty) if len(qty) else float("nan")
    for h in HORIZONS_S:
        mk = markouts(f, mid_t, mid_v, h * 1_000_000, end_ts)
        out[f"markout_{h}s_bps"] = weighted_mean(mk, qty) if len(qty) else float("nan")
        if ag.model == "A" and len(qty):
            # H3: split the optimistic agent's fills on its OWN orders.
            through = np.where(f["through"], qty, 0.0)
            supported = np.where(f["through"], 0.0, f["supported"])
            phantom = np.where(f["through"], 0.0, qty - f["supported"])
            for name, w in (("through", through), ("supported", supported), ("phantom", phantom)):
                out[f"A_{name}_qty"] = float(w.sum())
                out[f"A_{name}_markout_{h}s_bps"] = weighted_mean(mk, w)
    return out


def curves_for_day(sim, date, step_s=60):
    """Marked-to-mid gross equity and position of every agent at the end of each minute."""
    mid_t, mid_v = np.asarray(sim.mid_t), np.asarray(sim.mid_v)
    day0 = int(mid_t[0]) - int(mid_t[0]) % 86_400_000_000
    grid = day0 + np.arange(1, 86_400 // step_s + 1, dtype=np.int64) * step_s * 1_000_000
    grid = grid[grid <= sim.stats["last_ts"]]
    j = np.searchsorted(mid_t, grid, side="right") - 1
    mid_g = mid_v[np.clip(j, 0, None)]
    rows = []
    for ag in sim.agents:
        f = fills_to_arrays(ag.fills)
        _, eq = equity_curve(f, grid, mid_g)
        pos = position_at(f, grid)
        rows.append(pd.DataFrame(dict(date=date, agent=ag.name, minute=np.arange(1, len(grid) + 1),
                                      equity=eq, position=pos, mid=mid_g)))
    return pd.concat(rows, ignore_index=True)


def summarise_day(sim, date, symbol):
    rows = [summarise_agent(a, sim) for a in sim.agents]
    df = pd.DataFrame(rows)
    df.insert(0, "date", date)
    df.insert(1, "symbol", symbol)
    return df


def validation_record(sim, date):
    st = sim.stats
    states = max(st["book_states"], 1)
    rec = dict(date=date, **{k: v for k, v in st.items() if k not in ("first_ts", "last_ts")},
               crossed_pct=100 * st["crossed"] / states,
               trades_inside_pct=100 * st["trades_inside"] / max(st["trades"], 1),
               hours=(st["last_ts"] - st["first_ts"]) / 3.6e9 if st["first_ts"] else 0.0)
    rec["excluded"] = bool(rec["crossed_pct"] > 1 or rec["trades_inside_pct"] > 5 or st["resets"] > 10)
    return rec


def bootstrap_mean(x, n=10_000, seed=0):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return float(np.mean(x)) if len(x) else float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    return float(x.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired(df, metric, a, b):
    """Day-by-day difference metric(a) - metric(b), with a bootstrap over days."""
    p = df.pivot_table(index="date", columns="agent", values=metric)
    d = (p[a] - p[b]).dropna()
    mean, lo, hi = bootstrap_mean(d.to_numpy())
    return dict(metric=metric, a=a, b=b, days=len(d), mean_diff=mean, ci_low=lo, ci_high=hi,
                days_a_greater=int((d > 0).sum()))


# ---------------------------------------------------------------- across days
def pooled(df, agent, col, weight="filled_qty"):
    """Size-weighted mean of a per-day markout column over all days (descriptive)."""
    d = df[(df["agent"] == agent) & df[col].notna()]
    return float(np.sum(d[col] * d[weight]) / d[weight].sum()) if d[weight].sum() > 0 else float("nan")


def paired_diff(x, y, label):
    """Paired daily difference x - y (two aligned Series indexed by date), bootstrapped."""
    d = (x - y).dropna()
    mean, lo, hi = bootstrap_mean(d.to_numpy())
    return dict(comparison=label, days=len(d), mean_diff=mean, ci_low=lo, ci_high=hi,
                days_positive=int((d > 0).sum()))


def daily(df, agent, col):
    return df[df["agent"] == agent].set_index("date")[col]


def hypotheses(df, lat=50):
    """The three pre-registered hypotheses, evaluated exactly as research_plan.md states."""
    A, B, C = f"A_{lat}ms", f"Bprop_{lat}ms", f"C_{lat}ms"
    out = {}
    h1 = paired_diff(daily(df, A, "gross_pnl"), daily(df, B, "gross_pnl"), "gross PnL A - B")
    h1["supported"] = h1["days_positive"] > h1["days"] / 2
    out["H1"] = h1
    mk = "markout_10s_bps"
    ab = paired_diff(daily(df, A, mk), daily(df, B, mk), "10s markout A - B")
    bc = paired_diff(daily(df, B, mk), daily(df, C, mk), "10s markout B - C")
    means = {m: float(daily(df, a, mk).mean()) for m, a in (("A", A), ("B", B), ("C", C))}
    pool = {m: pooled(df, a, mk) for m, a in (("A", A), ("B", B), ("C", C))}
    both = ((daily(df, A, mk) > daily(df, B, mk)) & (daily(df, B, mk) > daily(df, C, mk))).sum()
    out["H2"] = dict(A_minus_B=ab, B_minus_C=bc, mean_of_daily=means, pooled=pool,
                     days_ordering_holds=int(both), days=ab["days"],
                     supported=means["A"] > means["B"] > means["C"])
    ph = daily(df, A, "A_phantom_markout_10s_bps")
    su = daily(df, A, "A_supported_markout_10s_bps")
    h3 = paired_diff(ph, su, "A's 10s markout: phantom - queue-supported")
    h3["phantom_mean"], h3["supported_mean"] = float(ph.mean()), float(su.mean())
    h3["supported"] = h3["mean_diff"] > 0
    out["H3"] = h3
    return out


def agent_table(df):
    """Mean over days of the main metrics, one row per agent (descriptive)."""
    g = df.groupby("agent")
    t = g[["fill_events", "filled_qty", "filled_notional", "fill_rate", "orders_placed",
           "orders_rejected", "cancels", "orders_partial", "gross_pnl", "spread_capture",
           "inventory_pnl", "net_pnl_1bp", "net_pnl_2bp", "realised", "unrealised",
           "gross_pnl_liquidated", "max_drawdown_gross", "mean_abs_position",
           "max_abs_position", "share_time_at_limit", "through_qty", "eff_spread_bps",
           "markout_1s_bps", "markout_10s_bps", "markout_60s_bps",
           "identity_error"]].mean()
    t["gross_pnl_sd"] = g["gross_pnl"].std()
    t["identity_error"] = g["identity_error"].max()
    t["days"] = g.size()
    t["gross_bps_of_notional"] = 1e4 * g["gross_pnl"].sum() / g["filled_notional"].sum()
    t["through_share"] = g["through_qty"].sum() / g["filled_qty"].sum()
    for h in (1, 10, 60):
        t[f"pooled_markout_{h}s_bps"] = [pooled(df, a, f"markout_{h}s_bps") for a in t.index]
    t["pooled_eff_spread_bps"] = [pooled(df, a, "eff_spread_bps") for a in t.index]
    t["pooled_realised_spread_10s_bps"] = t["pooled_eff_spread_bps"] + t["pooled_markout_10s_bps"]
    meta = df.groupby("agent")[["model", "qrule", "latency_ms"]].first()
    return meta.join(t).sort_values(["latency_ms", "model", "qrule"])
