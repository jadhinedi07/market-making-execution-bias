# Methodology summary

This is a concise account of the methodology of the historical research study. The full pre-registered protocol, dated amendments, frozen configuration and numerical results are retained in the separate private archive. This excerpt is explanatory, not a replacement for that archived record.

## Question and hypotheses

The study asks how optimistic passive order fill assumptions change simulated market making performance. It compares:

1. Gross trading P&L under optimistic versus queue-estimated fills.
2. Adverse selection after fills granted by the three execution models.
3. Post-fill price movement on the optimistic strategy's model-supported and unsupported fills.

## Replay and comparison

A simple strategy quotes at the best bid and ask subject to position limits. The same underlying price path and quote decisions are evaluated with different execution assumptions. New and cancelled orders act only after simulated latency. Maker fees are charged on executed notional; each trading day starts flat and ends with inventory marked to the final midprice.

The optimistic model fills against trades printed at its quote. The queue-estimated model first deducts an estimated volume ahead, allocating cancellations by stated rules. The trade-through model requires prices to trade beyond its quote. All models grant full remaining fills on trade-through; that assumption is not proof of real-world fillability.

## Statistical comparison

Because fills within a day share a market path, the day—not individual fills—is the comparison unit. Differences are calculated on matched days and assessed with paired resampling. Limited sample size, aggregated L2 depth, unknown queue position, batching and the strategy's simplicity make conclusions exploratory rather than guarantees about actual execution.

## Data availability

No licensed historical prices, book messages or evaluation outputs are included here. The Python implementation and synthetic tests can be reviewed without access to the private raw files.
