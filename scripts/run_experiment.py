"""
run_experiment.py - the whole pipeline, in the order the research plan fixes.

  python scripts/run_experiment.py select    # selection day: choose contract and size, then freeze
  python scripts/run_experiment.py run       # every evaluation day: download, validate, simulate
  python scripts/run_experiment.py analyse   # tables and figures across days
  python scripts/run_experiment.py all

The selection stage writes results/frozen_config.json and refuses to overwrite it, so
the contract and order size cannot be changed after evaluation results exist.

For each evaluation day the run stage writes, under results/days/:
  {day}.csv               11 agents, base engine (book update first on equal timestamps)
  {day}_tiesens.csv       the same 11 agents, pre-registered tie rule (trades first)
  {day}_validation.csv    reconstruction and alignment checks, exclusion decision
  {day}_curves.csv.gz     per-minute equity and position of every agent (figures only)
Days are simulated in parallel processes; downloads happen first, one at a time, because
every download is appended to one manifest file.
"""
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))

from mmbias import data                                          # noqa: E402
from mmbias.analysis import (curves_for_day, paired, summarise_day,   # noqa: E402
                             validation_record)
from mmbias.plots import make_all                                # noqa: E402
from mmbias.simulate import AgentSpec, run_day                   # noqa: E402
from mmbias.validate import run_checks                           # noqa: E402

SELECTION_DAY = "2025-10-01"
EVAL_DAYS = [f"{y}-{m:02d}-01" for y, m in
             [(2025, 11), (2025, 12)] + [(2026, k) for k in range(1, 11)]]
CANDIDATES = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "LINKUSDT",
              "ADAUSDT", "AVAXUSDT"]
RES = os.path.join(ROOT, "results")
FROZEN = os.path.join(RES, "frozen_config.json")


def specs():
    s = []
    for lat in (10, 50, 250):
        us = lat * 1000
        s += [AgentSpec(f"A_{lat}ms", "A", "prop", us), AgentSpec(f"Bprop_{lat}ms", "B", "prop", us),
              AgentSpec(f"C_{lat}ms", "C", "prop", us)]
    s += [AgentSpec("Bback_50ms", "B", "back", 50_000), AgentSpec("Bfront_50ms", "B", "front", 50_000)]
    return s


def select():
    if os.path.exists(FROZEN):
        print("frozen_config.json already exists; selection is not re-run.")
        return json.load(open(FROZEN))
    sym, table = data.select_contract(CANDIDATES, SELECTION_DAY)
    os.makedirs(RES, exist_ok=True)
    table.to_csv(os.path.join(RES, "selection_table.csv"), index=False)
    print(table.to_string())
    if sym is None:
        raise SystemExit("No candidate satisfies the selection rule.")
    row = table[table["symbol"] == sym].iloc[0]
    q = data.order_size(row["median_trade_size"], row["qty_step"])
    cfg = dict(symbol=sym, selection_day=SELECTION_DAY, q=float(q), qmax=float(10 * q),
               tick=float(row["tick"]), frozen_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    json.dump(cfg, open(FROZEN, "w"), indent=1)
    print("FROZEN:", cfg)
    return cfg


DTYPES = ("incremental_book_L2", "trades", "book_snapshot_5")


def run_one_day(d, cfg):
    """Simulate and validate one evaluation day. Files must already be downloaded."""
    days = os.path.join(RES, "days")
    sym, q, qmax = cfg["symbol"], cfg["q"], cfg["qmax"]
    bp, tp, sp = (data.local_path(t, d, sym) for t in DTYPES)
    t0 = time.time()
    sim = run_day(bp, tp, specs(), q, qmax)
    summarise_day(sim, d, sym).to_csv(os.path.join(days, f"{d}.csv"), index=False)
    curves_for_day(sim, d).to_csv(os.path.join(days, f"{d}_curves.csv.gz"), index=False)
    v = validation_record(sim, d)
    v["seconds_base"] = round(time.time() - t0)
    del sim
    t1 = time.time()
    sim = run_day(bp, tp, specs(), q, qmax, trades_first_on_ties=True)
    summarise_day(sim, d, sym).to_csv(os.path.join(days, f"{d}_tiesens.csv"), index=False)
    v["seconds_tiesens"] = round(time.time() - t1)
    del sim
    chk = run_checks(bp, tp, sp)
    v["resnapshot_drift"] = json.dumps(chk.pop("drift"))
    v.update({f"chk_{k}": val for k, val in chk.items()})
    v["seconds_total"] = round(time.time() - t0)
    pd.DataFrame([v]).to_csv(os.path.join(days, f"{d}_validation.csv"), index=False)
    return (f"{d}: {v['rows']:,} rows, {v['trades']:,} trades, {v['seconds_total']}s, "
            f"crossed {v['crossed_pct']:.4f}%, inside {v['trades_inside_pct']:.2f}%, "
            f"resets {v['resets']}, snap5 top mismatches {v['chk_snap5_top_mismatch']}, "
            f"align lag0 {v['chk_align_lag0_pct']:.1f}%, excluded {v['excluded']}")


def run(workers=4):
    cfg = json.load(open(FROZEN))
    os.makedirs(os.path.join(RES, "days"), exist_ok=True)
    todo = [d for d in EVAL_DAYS if not os.path.exists(os.path.join(RES, "days", f"{d}_validation.csv"))]
    for d in todo:                              # sequential: one manifest file
        for t in DTYPES:
            data.download(t, d, cfg["symbol"])
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for line in ex.map(run_one_day, todo, [cfg] * len(todo)):
            print(line, flush=True)


def _load(days, suffix):
    files = sorted(f for f in os.listdir(days) if f.endswith(f"{suffix}.csv")
                   and f[:10] in EVAL_DAYS and (suffix or f == f[:10] + ".csv"))
    return pd.concat([pd.read_csv(os.path.join(days, f)) for f in files], ignore_index=True)


def _md(df, path, floatfmt=".3f"):
    with open(path, "w") as fh:
        fh.write(df.to_markdown(floatfmt=floatfmt) + "\n")


def analyse():
    from mmbias.analysis import agent_table, hypotheses, paired_diff, daily
    cfg = json.load(open(FROZEN))
    days = os.path.join(RES, "days")
    tabs = os.path.join(RES, "tables")
    os.makedirs(tabs, exist_ok=True)
    df, ts = _load(days, ""), _load(days, "_tiesens")
    val = _load(days, "_validation")
    keep = set(val.loc[~val["excluded"].astype(bool), "date"])
    excluded = sorted(set(val["date"]) - keep)
    df_all = df                                 # robustness only: excluded days kept
    df, ts = df[df["date"].isin(keep)], ts[ts["date"].isin(keep)]
    df.to_csv(os.path.join(RES, "all_days.csv"), index=False)
    ts.to_csv(os.path.join(RES, "all_days_tiesens.csv"), index=False)
    val.to_csv(os.path.join(RES, "validation.csv"), index=False)

    summary = dict(symbol=cfg["symbol"], q=cfg["q"], qmax=cfg["qmax"], days=sorted(keep),
                   excluded_days=excluded,
                   hypotheses={f"{lat}ms": hypotheses(df, lat) for lat in (50, 10, 250)},
                   hypotheses_pre_registered_tie_rule=hypotheses(ts, 50),
                   hypotheses_robustness_all_days_incl_excluded=hypotheses(df_all, 50),
                   max_identity_error=float(max(df["identity_error"].max(), ts["identity_error"].max())))

    # paired comparisons (plan: paired daily differences, bootstrap over days)
    comps = []
    for metric in ("gross_pnl", "net_pnl_1bp", "net_pnl_2bp", "spread_capture", "inventory_pnl",
                   "markout_1s_bps", "markout_10s_bps", "markout_60s_bps", "filled_qty",
                   "eff_spread_bps", "mean_abs_position", "max_drawdown_gross"):
        for lat in (10, 50, 250):
            for x, y in ((f"A_{lat}ms", f"Bprop_{lat}ms"), (f"Bprop_{lat}ms", f"C_{lat}ms"),
                         (f"A_{lat}ms", f"C_{lat}ms")):
                comps.append(dict(metric=metric, latency_ms=lat, **paired_diff(
                    daily(df, x, metric), daily(df, y, metric), f"{x} - {y}")))
        for x in ("Bback_50ms", "Bfront_50ms"):
            comps.append(dict(metric=metric, latency_ms=50, **paired_diff(
                daily(df, x, metric), daily(df, "Bprop_50ms", metric), f"{x} - Bprop_50ms")))
    comps = pd.DataFrame(comps)
    comps.to_csv(os.path.join(tabs, "paired_comparisons.csv"), index=False)

    at = agent_table(df)
    at.to_csv(os.path.join(tabs, "agent_summary.csv"))

    # all 33 configurations (11 agents x 3 fee scenarios)
    rows = []
    for ag, g in df.groupby("agent"):
        for fee in ("0", "1", "2"):
            c = g[f"net_pnl_{fee}bp"]
            rows.append(dict(agent=ag, fee_bp=int(fee), mean_daily_net=c.mean(), sd=c.std(),
                             days_positive=int((c > 0).sum()), days=len(c), total=c.sum()))
    conf = pd.DataFrame(rows)
    conf.to_csv(os.path.join(tabs, "all_33_configurations.csv"), index=False)

    # tie-rule sensitivity: base minus pre-registered rule, per agent
    m = df.merge(ts, on=["date", "agent"], suffixes=("", "_pre"))
    tie = []
    for ag, g in m.groupby("agent"):
        tie.append(dict(agent=ag, gross_base=g["gross_pnl"].mean(), gross_pre=g["gross_pnl_pre"].mean(),
                        filled_base=g["filled_qty"].mean(), filled_pre=g["filled_qty_pre"].mean(),
                        mk10_base=g["markout_10s_bps"].mean(), mk10_pre=g["markout_10s_bps_pre"].mean(),
                        max_rel_diff_filled=float((g["filled_qty"] / g["filled_qty_pre"] - 1).abs().max())))
    tie = pd.DataFrame(tie)
    tie.to_csv(os.path.join(tabs, "tie_rule_sensitivity.csv"), index=False)

    vcols = ["date", "messages", "rows", "trades", "resets", "crossed_pct", "trades_inside_pct",
             "missing_delete", "trades_out_of_order", "chk_snap5_top_checked", "chk_snap5_top_mismatch",
             "chk_snap5_deep_checked", "chk_snap5_deep_mismatch", "chk_align_lag0_pct",
             "chk_align_lag-1_pct", "chk_align_lag+1_pct", "chk_tie_volume_pct", "chk_align_tie_lag0_pct",
             "chk_align_tie_lag+1_pct", "chk_max_gap_ms", "chk_gaps_over_1s", "hours", "excluded"]
    _md(val[vcols].set_index("date"), os.path.join(tabs, "validation.md"), ".2f")

    curves = pd.concat([pd.read_csv(os.path.join(days, f"{d}_curves.csv.gz")) for d in sorted(keep)])
    # Operational inventory-limit measure: share of minutes in which the limit stops one
    # side quoting (|position| + q > limit). Sampled at minute ends from the curves; the
    # per-day column share_time_at_limit counts only |position| >= limit, which misses
    # partially filled positions such as 19.6 that still block a 2-lot order.
    blocked = curves.assign(b=curves["position"].abs() + cfg["q"] > cfg["qmax"] + 1e-9).groupby("agent")["b"].mean()
    at["share_minutes_one_side_blocked"] = blocked
    at.to_csv(os.path.join(tabs, "agent_summary.csv"))
    # market context per day (from the per-minute mid path and the trade counts)
    mid = curves[curves["agent"] == curves["agent"].iloc[0]].sort_values(["date", "minute"])
    ctx = []
    for d, g in mid.groupby("date"):
        m = g["mid"].to_numpy()
        r = np.diff(np.log(m))
        ctx.append(dict(date=d, first_mid=m[0], last_mid=m[-1], day_return_pct=100 * (m[-1] / m[0] - 1),
                        range_pct=100 * (m.max() / m.min() - 1),
                        realised_vol_1min_annualised_pct=100 * r.std() * np.sqrt(1440 * 365),
                        half_tick_bps=0.5 * 0.001 / m.mean() * 1e4,
                        trades=int(val.set_index("date").loc[d, "trades"])))
    ctx = pd.DataFrame(ctx).set_index("date")
    ctx.to_csv(os.path.join(tabs, "market_context.csv"))
    _md(ctx, os.path.join(tabs, "market_context.md"), ".3f")
    make_all(df, os.path.join(RES, "figures"), curves, cfg["qmax"], cfg["q"])
    json.dump(summary, open(os.path.join(RES, "summary.json"), "w"), indent=1, default=float)
    print(json.dumps(summary["hypotheses"]["50ms"], indent=1, default=float))
    print("excluded days:", excluded, "| max identity error:", summary["max_identity_error"])


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    if stage in ("select", "all"):
        select()
    if stage in ("run", "all"):
        run()
    if stage in ("analyse", "all"):
        analyse()
