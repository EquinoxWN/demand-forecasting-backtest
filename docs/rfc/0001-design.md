# RFC 0001: demand-forecasting-backtest design

- **Status:** Accepted for M1 (M2 and M3 parts are still proposals)
- **Author:** EquinoxWN
- **Created:** 2026-11-11

## Problem

Stores run out of best-sellers and overstock the rest. Forecasts that drive ordering are often
judged on one optimistic number, computed on data the model has partly seen, and never compared
with the simplest possible forecast. A planner then cannot tell whether a model is worth running.

Users: data scientists and analysts who forecast retail demand and need an honest evaluation
setup, and engineers who want a tested reference for preparing the M5 (Walmart) data.

## Goals

M1 (built):

- Prepare M5 sales data with calendar, price and promotion features at item (item x store),
  store and total level, using only information available on each day.
- Score a seasonal-naive baseline first, on a holdout of the last days; any other model counts
  only if it has a lower WAPE than the baseline at that level.

Later milestones (not built):

- M2: ETS and ARIMA (statsforecast) and a global LightGBM model per hierarchy level; rolling-origin
  backtests that forecast week by week.
- M3: prediction-interval coverage checks; a reorder-point simulation that turns forecast error
  into stock-out and overstock cost.

## Non-goals

- A hosted forecasting service, a UI or a database.
- Reconciliation of forecasts across levels (each level is scored on its own in M1).
- Beating the M5 competition leaderboard.

## Critical-to-quality measures (Define)

| Measure | Target | How it is checked |
|---|---|---|
| Malformed input refused | every case in the validation test matrix raises `DataError` (exit 2 in the CLI) | `tests/test_data.py`, `tests/test_cli.py` |
| No look-ahead in features | 0 changed values up to day t when sales after t and prices of later weeks are altered | `test_features_never_look_ahead` |
| Hierarchy coherence | store sales equal the sum of their items and total equals the sum of stores, exactly | `test_levels_add_up` |
| Baseline first | the seasonal-naive row is scored first at every level; a model with equal WAPE does not count | `tests/test_baseline.py` |
| Download integrity | archive kept only over HTTPS, under 60 MB, with the pinned SHA-256 | download tests in `tests/test_data.py` |
| Runtime | one M5 store (3,049 series, 1,941 days) prepared and scored in under 60 s on a laptop CPU | `make run`, see [results](../results/m1.md) |

## Design

```
m5.zip / CSV directory
  -> data.py       calendar.csv, the sales file and sell_prices.csv, read in chunks for the
                   selected stores and validated (columns, day order, whole non-negative sales,
                   positive prices, no duplicates, SNAP column per state)
  -> features.py   series x day matrices:
                     item:  sell_price, regular_price (running max so far), discount, promo
                            (discount >= 5%), price_change (week over week), on_shelf, snap
                     store / total: summed sales, items_on_shelf, mean_discount, share_promo, snap
                     calendar: wday, month, year, event, event_type
  -> baseline.py   holdout = last h days (default 28, the M5 horizon)
                   seasonal_naive (baseline), naive_last, mean_28d, seasonal_mean_4w
                   WAPE (gate) and MASE (reported) per level; counts = WAPE < baseline WAPE
  -> cli.py        dfb download | dfb features | dfb backtest
```

- **Data source.** The Kaggle copy of M5 needs a sign-in. The code downloads the public mirror
  that Nixtla publishes on GitHub (the same files; `calendar.csv` there has no `d` column, so days
  are taken in row order, and a `d` column is checked when present). No checksum is published
  upstream, so the SHA-256 of the archive downloaded on 2026-11-11 is pinned in `data.py`.
- **Promotions.** M5 has no promotion flag. A promotion is inferred from price: the regular price
  is the highest price of that item in that store so far, and a price at least 5% under it is a
  promotion ([ADR 0002](../adr/0002-promotions-from-price-history.md)).
- **No look-ahead.** Every feature at day t uses data up to the end of t's Walmart week (prices
  are weekly) and no later sales; the property is tested by perturbing the future.
- **Memory.** Data is held as NumPy matrices (series x day) instead of a long table joined with
  pandas; one store is about 3,049 x 1,941 values per feature
  ([ADR 0003](../adr/0003-matrices-and-a-strict-wape-gate.md)).
- **Writes.** Feature tables and JSON scores are written to a temporary file and renamed into
  place; gzip output uses a fixed header time, so the same input gives the same bytes.

## Alternatives considered

| Option | Why not (yet) |
|---|---|
| Long table with pandas `melt` and `merge` for prices | Simple to write, but one store becomes 5.9 M rows before any feature, and all ten stores 59 M; the joins need several times that in memory. Matrices keep one store well under a gigabyte. |
| A dataset helper (for example `datasetsforecast`) to download and shape M5 | Saves code, but adds a dependency and skips the input validation and the no-look-ahead property this repo is meant to prove. statsforecast arrives in M2 for models, not for data. |
| Kaggle API download | Needs an account and an API token on the machine; the public mirror needs neither. |
| Rolling-origin backtest already in M1 | It is the M2 item; M1 scores one holdout of the last 28 days so the baseline exists first. |
| MASE as the gate | MASE averages per series and is undefined for series that never change; WAPE weights by volume and is defined whenever the holdout has any sales. MASE is still reported. |

## Measurement plan (Measure)

- `make test`: the test suite on synthetic M5-format data with planted weekly seasonality,
  promotion, event and SNAP effects.
- `make run`: the holdout table for store CA_1 (last 28 days of `sales_train_evaluation.csv`),
  wall time measured with a monotonic clock.
- `dfb features --level item`: rows written and time for one store.
- Later (M2/M3): rolling-origin WAPE per level against the baseline, interval coverage and
  simulated inventory cost.

## Milestones

- **M1:** M5 data prepared with calendar, price and promotion features at item, store and total
  level; seasonal-naive baseline first. Done, see [results](../results/m1.md).
- **M2:** ETS, ARIMA and a global LightGBM model per hierarchy level; rolling-origin backtests.
- **M3:** interval coverage; reorder-point simulation with stock-out and overstock cost.

## Risks and open questions

- The mirror could change or disappear; the pinned checksum makes a change visible, and the
  loader also reads a directory with the original Kaggle files.
- Inferring promotions from price misses promotions that do not change the shelf price and marks
  a permanent price cut as a lasting promotion (ADR 0002).
- All ten stores at once need several gigabytes as float64 matrices; M1 was run on one store.
  Smaller dtypes or per-store processing are options for M2.
