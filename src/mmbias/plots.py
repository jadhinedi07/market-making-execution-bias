"""
plots.py - figures for the report. One idea per chart; each execution model keeps one
colour everywhere (A blue, B orange, C aqua: the first three slots of a palette checked
for colour-vision-deficiency separation). Text is ink-coloured, never series-coloured;
grids are hairline and recessive; every chart with two or more series has a legend.
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .analysis import CURVE_HORIZONS_S, daily, pooled

SURFACE, INK, INK2, MUTED, GRID, BASE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
COL = {"A": "#2a78d6", "B": "#eb6834", "C": "#1baf7a"}
SEQ = ["#86b6ef", "#2a78d6", "#104281"]          # ordinal blue ramp (steps 250 / 450 / 650)
LAB = {"A": "A optimistic", "B": "B queue-aware", "C": "C conservative"}
LW, MS = 1.4, 5.5                                # ~2-3 px line, >= 8 px markers at 150 dpi

plt.rcParams.update({
    "figure.figsize": (8, 4.6), "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "axes.edgecolor": BASE, "axes.labelcolor": INK2,
    "axes.titlecolor": INK, "axes.titlesize": 11, "axes.titleweight": "semibold",
    "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
    "axes.axisbelow": True, "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelcolor": INK2, "ytick.labelcolor": INK2, "font.size": 9.5,
    "legend.frameon": False, "legend.labelcolor": INK2,
    "font.family": ["DejaVu Sans"],
})


def _base(df, lat=50):
    return df[(df["latency_ms"] == lat) & ((df["model"] != "B") | (df["qrule"] == "prop"))]


def _agent(m, lat=50):
    return {"A": f"A_{lat}ms", "B": f"Bprop_{lat}ms", "C": f"C_{lat}ms"}[m]


def _zero(ax):
    ax.axhline(0, color=INK2, lw=0.8, zorder=1)


def _subtitle(ax, text):
    ax.text(0, 1.02, text, transform=ax.transAxes, color=INK2, fontsize=8.5, va="bottom")


def _save(fig, out, name):
    fig.tight_layout()
    fig.savefig(os.path.join(out, name), dpi=150)
    plt.close(fig)


def _line(ax, x, y, m, **kw):
    ax.plot(x, y, "-o", color=COL[m], lw=LW, ms=MS, mec=SURFACE, mew=1.0, label=LAB[m], **kw)


def markout_curve(df, out):
    fig, ax = plt.subplots()
    hs = list(CURVE_HORIZONS_S)
    for m in "ABC":
        y = [pooled(df, _agent(m), f"mkcurve_{h:g}s_bps") for h in hs]
        _line(ax, hs, y, m)
        ax.annotate(f"{y[-1]:.2f}", (hs[-1], y[-1]), xytext=(6, 0), textcoords="offset points",
                    va="center", color=INK2, fontsize=8.5)
    _zero(ax)
    ax.set_xscale("log")
    ax.set_xticks(hs, [f"{h:g}s" for h in hs])
    ax.minorticks_off()
    ax.set_xlabel("Time after the fill")
    ax.set_ylabel("Signed markout per unit filled (bps)")
    ax.set_title("After a fill, the price moves against the market maker", pad=18)
    _subtitle(ax, "Size-weighted over all evaluation days, 50 ms latency. Below zero = adverse selection.")
    ax.set_ylim(top=0.35)
    ax.legend(loc="upper center", ncol=3)
    _save(fig, out, "01_markout_curve.png")


def pnl_decomposition(df, out):
    base = _base(df)
    g = base.groupby("model")[["spread_capture", "inventory_pnl", "filled_notional", "net_pnl_2bp"]].mean()
    comps = [("spread capture", g["spread_capture"]), ("inventory PnL", g["inventory_pnl"]),
             ("fees at 2 bp", -2e-4 * g["filled_notional"]), ("net PnL at 2 bp", g["net_pnl_2bp"])]
    fig, ax = plt.subplots()
    x = np.arange(len(comps))
    w = 0.26
    for i, m in enumerate("ABC"):
        vals = [c[1][m] for c in comps]
        ax.bar(x + (i - 1) * w, vals, w, color=COL[m], label=LAB[m], edgecolor=SURFACE, linewidth=1.0)
    _zero(ax)
    ax.set_xticks(x, [c[0] for c in comps])
    ax.set_ylabel("Mean per day (USDT)")
    ax.set_title("Spread capture is small; inventory losses and fees dominate", pad=18)
    _subtitle(ax, "PnL = spread capture + inventory PnL - fees (exact, checked every run). 50 ms latency.")
    ax.legend(loc="lower left")
    _save(fig, out, "02_pnl_decomposition.png")


def daily_gross(df, out):
    fig, ax = plt.subplots()
    for m in "ABC":
        s = daily(df, _agent(m), "gross_pnl").sort_index()
        _line(ax, pd.to_datetime(s.index), s.values, m)
    _zero(ax)
    ax.set_ylabel("Gross PnL (USDT)")
    ax.set_title("Same strategy, same days: daily gross PnL by execution model", pad=18)
    _subtitle(ax, "AVAXUSDT perpetual, first day of each month, 50 ms latency, before fees.")
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b '%y"))
    ax.legend(loc="lower left")
    _save(fig, out, "03_daily_gross_pnl.png")


def overstatement(df, out):
    """Paired daily differences in gross PnL: how much more the looser model reports."""
    fig, ax = plt.subplots()
    pairs = [("A", "B"), ("A", "C"), ("B", "C")]
    rng = np.random.default_rng(0)
    for i, (a, b) in enumerate(pairs):
        d = (daily(df, _agent(a), "gross_pnl") - daily(df, _agent(b), "gross_pnl")).dropna()
        jitter = rng.uniform(-0.12, 0.12, len(d))
        ax.scatter(i + jitter, d.values, s=28, color=MUTED, edgecolor=SURFACE, linewidth=0.8, zorder=3)
        means = rng.choice(d.values, size=(10_000, len(d))).mean(axis=1)
        lo, hi = np.percentile(means, [2.5, 97.5])
        ax.plot([i + 0.25, i + 0.25], [lo, hi], color=INK, lw=1.4, zorder=4)
        ax.plot([i + 0.19, i + 0.31], [d.mean()] * 2, color=INK, lw=2.0, zorder=4)
        ax.annotate(f"mean {d.mean():+.1f}", (i + 0.33, d.mean()), va="center", color=INK2, fontsize=8.5)
    _zero(ax)
    ax.set_xticks(range(3), [f"{LAB[a]}\nminus {LAB[b]}" for a, b in pairs])
    ax.set_xlim(-0.5, 2.9)
    ax.set_ylabel("Difference in daily gross PnL (USDT)")
    ax.set_title("How much a looser fill model changes the answer, day by day", pad=18)
    _subtitle(ax, "Dots: evaluation days. Bar: mean with 95% bootstrap interval over days. 50 ms latency.")
    _save(fig, out, "04_paired_overstatement.png")


def phantom_split(df, out):
    a = df[df["agent"] == _agent("A")]
    fig, ax = plt.subplots()
    names = ["through", "supported", "phantom"]
    labels = ["traded through\n(every model fills)", "queue-supported\n(B would also fill)",
              "phantom\n(only A fills)"]
    w = 0.24
    for i, h in enumerate([1, 10, 60]):
        vals = []
        for n in names:
            wq = a[f"A_{n}_qty"]
            vals.append(np.nansum(a[f"A_{n}_markout_{h}s_bps"] * wq) / wq.sum() if wq.sum() else np.nan)
        ax.bar(np.arange(3) + (i - 1) * w, vals, w, color=SEQ[i], label=f"{h} s",
               edgecolor=SURFACE, linewidth=1.0)
    _zero(ax)
    ax.set_xticks(np.arange(3), labels)
    ax.set_ylabel("Signed markout (bps)")
    ax.set_title("The optimistic model's own fills, split by what justified them", pad=18)
    _subtitle(ax, "Size-weighted over all evaluation days. Same orders, three kinds of fill.")
    ax.legend(title="horizon", title_fontsize=8.5, loc="lower right")
    _save(fig, out, "05_phantom_vs_supported.png")


def sensitivity(df, out):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    lat = df[(df["model"] != "B") | (df["qrule"] == "prop")]
    g = lat.groupby(["model", "latency_ms"])["gross_pnl"].mean().unstack(0)
    for m in "ABC":
        _line(axes[0], g.index, g[m], m)
    axes[0].set_xscale("log")
    axes[0].set_xticks([10, 50, 250], ["10 ms", "50 ms", "250 ms"])
    axes[0].minorticks_off()
    _zero(axes[0])
    axes[0].set_xlabel("Order latency")
    axes[0].set_ylabel("Mean daily gross PnL (USDT)")
    axes[0].set_title("Latency")
    axes[0].legend(loc="lower right")
    q = df[(df["model"] == "B") & (df["latency_ms"] == 50)].groupby("qrule")["gross_pnl"].mean()
    order = [r for r in ("back", "prop", "front") if r in q.index]
    axes[1].bar(order, q[order], 0.5, color=COL["B"], edgecolor=SURFACE)
    for i, r in enumerate(order):
        axes[1].annotate(f"{q[r]:.1f}", (i, q[r]), xytext=(0, -12), textcoords="offset points",
                         ha="center", color=INK2, fontsize=8.5)
    _zero(axes[1])
    axes[1].set_title("Queue-aware model: where cancellations come from")
    axes[1].set_xticks(range(len(order)), ["behind us\n(back)", "spread evenly\n(prop, base)",
                                           "ahead of us\n(front)"][:len(order)])
    axes[1].set_ylabel("Mean daily gross PnL (USDT)")
    _save(fig, out, "06_sensitivity.png")


def fee_scenarios(df, out):
    fig, ax = plt.subplots()
    base = _base(df).groupby("model")[["net_pnl_0bp", "net_pnl_1bp", "net_pnl_2bp"]].mean()
    for m in "ABC":
        _line(ax, [0, 1, 2], base.loc[m].values, m)
        ax.annotate(f"{base.loc[m].values[-1]:.0f}", (2, base.loc[m].values[-1]), xytext=(6, 0),
                    textcoords="offset points", va="center", color=INK2, fontsize=8.5)
    _zero(ax)
    ax.set_xticks([0, 1, 2], ["0 bp (gross)", "1 bp", "2 bp (Binance\nregular maker)"])
    ax.set_xlim(-0.2, 2.35)
    ax.set_ylabel("Mean daily net PnL (USDT)")
    g = base.join(_base(df).groupby("model")[["filled_notional"]].mean())
    f_star = (g.loc["A", "net_pnl_0bp"] - g.loc["B", "net_pnl_0bp"]) / (g.loc["A", "filled_notional"] - g.loc["B", "filled_notional"]) * 1e4
    ax.set_title("After fees the ranking reverses: the model that fills most loses most", pad=18)
    _subtitle(ax, f"Net = gross - fee x filled notional. A and B break even at a fee of {f_star:.2f} bp. 50 ms latency.")
    ax.legend(loc="lower left")
    _save(fig, out, "07_fee_scenarios.png")


def cumulative_equity(curves, out):
    """Per-minute gross equity, each day starting flat, chained across evaluation days."""
    fig, ax = plt.subplots(figsize=(10, 4.6))
    days = sorted(curves["date"].unique())
    for m in "ABC":
        c = curves[curves["agent"] == _agent(m)].sort_values(["date", "minute"])
        offset, xs, ys = 0.0, [], []
        for k, d in enumerate(days):
            e = c[c["date"] == d]
            xs.append(k + e["minute"].to_numpy() / 1440)
            ys.append(offset + e["equity"].to_numpy())
            offset += e["equity"].to_numpy()[-1]
        ax.plot(np.concatenate(xs), np.concatenate(ys), color=COL[m], lw=LW, label=LAB[m])
        ax.annotate(f"{offset:.0f}", (len(days), offset), xytext=(5, 0), textcoords="offset points",
                    va="center", color=INK2, fontsize=8.5)
    _zero(ax)
    ax.set_xticks(np.arange(len(days)) + 0.5, [pd.Timestamp(d).strftime("%b '%y") for d in days])
    ax.set_xlim(0, len(days) + 0.6)
    for k in range(1, len(days)):
        ax.axvline(k, color=GRID, lw=0.6, zorder=0)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("Cumulative gross PnL (USDT)")
    ax.set_title(f"Cumulative gross PnL over the {len(days)} evaluation days", pad=18)
    _subtitle(ax, "Each day starts flat and is marked to mid every minute; days are chained end to end. 50 ms.")
    ax.legend(loc="lower left")
    _save(fig, out, "08_cumulative_equity.png")


def inventory_distribution(curves, out, qmax, q):
    """Empirical CDF of |position|: no binning, which would misplace the odd positions
    that partial fills create (C moves in steps of q, A and B do not)."""
    fig, ax = plt.subplots()
    ax.axvspan(qmax - q, qmax + 3.5, color=GRID, alpha=0.6, lw=0, zorder=0)
    ax.annotate("one side blocked\nby the limit", (qmax - q + 0.2, 0.04), color=INK2, fontsize=8.5)
    shares = []
    for m in "ABC":
        p = np.sort(np.abs(curves[curves["agent"] == _agent(m)]["position"].to_numpy()))
        y = np.arange(1, len(p) + 1) / len(p)
        ax.step(p, y, where="post", color=COL[m], lw=LW, label=LAB[m])
        shares.append(f"{m} {100 * np.mean(p > qmax - q + 1e-9):.0f}%")
    ax.axvline(qmax, color=INK2, lw=0.8)
    ax.annotate("limit", (qmax, 1.0), xytext=(3, -12), textcoords="offset points", color=INK2, fontsize=8.5)
    ax.set_xlim(0, qmax + 3.5)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("|Position| (AVAX), sampled at the end of every minute")
    ax.set_ylabel("Share of minutes with |position| at or below x")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_title("Inventory: the optimistic model spends twice as long pinned near its limit", pad=18)
    _subtitle(ax, f"Limit {qmax:g} AVAX, order size {q:g}, 50 ms. Minutes with one side blocked: " + ", ".join(shares) + ".")
    ax.legend(loc="upper left")
    _save(fig, out, "09_inventory_distribution.png")


def make_all(df, out, curves=None, qmax=None, q=None):
    os.makedirs(out, exist_ok=True)
    for fn in (markout_curve, pnl_decomposition, daily_gross, overstatement, phantom_split,
               sensitivity, fee_scenarios):
        fn(df, out)
    if curves is not None:
        cumulative_equity(curves, out)
        inventory_distribution(curves, out, qmax, q)
