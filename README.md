# Market-Making Execution Bias | Python Simulator

**Research question:** How do assumptions about passive limit order fills change the results of a market making backtest?

This repository contains Python source code and synthetic tests for an event-driven order book replay and execution simulator. It compares different fill assumptions on the same historical market path. It is a research exercise, **not** a live trading bot or a claim that the strategy is profitable.

A trade printing at the price of a hypothetical passive order does not guarantee the order would have executed, because other orders may have had priority. Fill modelling therefore affects both simulated turnover and the apparent profitability of a trading strategy.

## Execution assumptions

| Model | When trades print at the order's price | When trades pass through the order's price |
|---|---|---|
| **A: Optimistic** | Fill immediately up to the printed trade quantity | Fill |
| **B: Queue estimate** | Consume an estimated quantity ahead before filling | Fill |
| **C: Trade through only** | No fill | Fill |

Model B estimates queue position from displayed order book depth, trades and assumed cancellation allocation. True counterfactual queue positions **cannot be observed in level-2 data**. Model C is not guaranteed to be a lower bound on actual fills: the simulated trade-through rule and delayed cancellation can be generous.

The simulator includes order placement and cancellation latency, inventory constraints, maker fee scenarios, end-of-day valuation, markouts and day-level paired comparisons.

## Implementation

- `src/mmbias/events.py` streams trade and book events.
- `book.py` reconstructs the order book from price-level updates.
- `execution.py` implements optimistic, queue-estimated and trade-through fill rules.
- `simulate.py` simulates quotes, cancels, latency and inventory.
- `accounting.py` calculates trading P&L, fees and adverse selection measures.
- `analysis.py`, `plots.py` and `validate.py` analyse and validate the simulation.
- `scripts/run_experiment.py` provides the research pipeline.

## Run synthetic tests

Python 3.13 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q
```

The included 38 unit and integration tests use synthetic inputs, so no licensed market data are needed.

## Full historical reproduction

```bash
python scripts/run_experiment.py select
python scripts/run_experiment.py run
python scripts/run_experiment.py analyse
```

This pipeline requires separately licensed Tardis.dev order book and trade files. Exact reproduction of the private historical study additionally requires its frozen configuration and experiment outputs, which are **not included** in this repository. The public code is therefore testable with synthetic data, but the original empirical findings are not independently reproducible from this package alone.

## Research limitations

- Aggregated level-2 feeds do not reveal actual individual order priorities or hidden liquidity.
- Queue estimates and full fills on trade-through events are modelling assumptions.
- Execution latency, exchange constraints, order cancellation behaviour, fees and inventory limits affect simulated results.
- Findings from one instrument and a limited number of days do not establish general market profitability.

## Data rights and project provenance

This is a **clean source-only copy with new Git history**, separate from the private study archive. No downloaded market data, sampled historical prices, day-level result tables or data-derived figures are distributed here. Please obtain market data from its provider subject to its terms: [Tardis.dev terms](https://docs.tardis.dev/legal/terms-of-service).

The implementation used substantial AI coding assistance alongside human direction and review of the research design. The code is provided for study and methodological discussion; it has not been independently audited for live trading use.

No open-source licence is granted at this stage.
