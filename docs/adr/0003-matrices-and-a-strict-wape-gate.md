# ADR 0003: Hold series as matrices, and gate models on a strictly lower WAPE

- **Status:** Accepted

## Context

M5 stores sales wide (one row per item and store, one column per day) and prices long (one row per
item, store and week). Most feature code melts the sales into a long table and joins prices. The
machine this was built on had about 1.6 GB of free memory during the run. Separately, "every model
has to beat the baseline to count" needs one precise rule.

## Decision

- Keep each hierarchy level as a `Panel`: a sales matrix (series x day) and one matrix per feature.
  Weekly prices go into a series x week matrix and are expanded to days by indexing with each
  day's week. The long table is produced only for export, in batches of 500 series.
- The gate is WAPE (sum of absolute errors over sum of actuals) on the holdout, per level. A model
  counts only if its WAPE is strictly lower than the seasonal-naive WAPE at that level. MASE is
  reported next to it, averaged over the series whose history varies.

## Consequences

- One store (3,049 series, 1,941 days) loads, builds all three levels and scores four models in
  about 20 s on the laptop CPU; there is no join step that multiplies memory.
- Matrices are float64 for simple, exact sums; all ten stores at once would need several
  gigabytes, so M1 runs per store. float32 or per-store processing can reduce this later.
- A tie with the baseline does not count, so a model that just reproduces seasonal naive is not
  credited.
- WAPE is dominated by high-volume series; at item level that is intended (it tracks units that
  have to be stocked), and MASE shows the per-series view.
