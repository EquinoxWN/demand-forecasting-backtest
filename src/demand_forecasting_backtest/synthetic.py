"""Small M5-format data with planted weekly seasonality, promotions, events and SNAP days."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from .data import SALES_FILE
from .files import atomic_text

START = "2011-01-29"  # a Saturday, as in M5 (wday 1 = Saturday)
DEPTS = ("FOODS_1", "HOUSEHOLD_1", "HOBBIES_1")
# Planted multipliers the tests check: weekend peak, promotion and event lifts.
WEEKDAY_FACTOR = np.array([1.5, 1.4, 0.8, 0.8, 0.8, 0.9, 1.1])  # Saturday .. Friday
PROMO_DISCOUNT = 0.3
PROMO_LIFT = 2.0  # demand multiplier per unit of discount, so a 30% cut gives 1.6x
EVENT_LIFT = 1.4
SNAP_LIFT = 1.15


def write_synthetic_m5(
    dest: Path,
    *,
    stores: Sequence[str] = ("CA_1", "TX_1"),
    n_items: int = 12,
    n_days: int = 364,
    seed: int = 0,
    d_column: bool = True,
) -> Path:
    """Write calendar.csv, sell_prices.csv and the sales file into dest."""
    if n_items < 1 or n_days < 14 or not stores:
        raise ValueError("need at least one store, one item and 14 days")
    rng = np.random.default_rng(seed)
    dates = pd.date_range(START, periods=n_days, freq="D")
    day = np.arange(n_days)
    wday = day % 7 + 1
    week = 11101 + day // 7
    event = np.zeros(n_days, dtype=bool)
    event[45::91] = True
    calendar = pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "wm_yr_wk": week,
            "weekday": dates.day_name(),
            "wday": wday,
            "month": dates.month,
            "year": dates.year,
        }
    )
    if d_column:
        calendar["d"] = [f"d_{i}" for i in range(1, n_days + 1)]
    calendar["event_name_1"] = np.where(event, "Holiday", "NA")
    calendar["event_type_1"] = np.where(event, "National", "NA")
    calendar["event_name_2"] = "NA"
    calendar["event_type_2"] = "NA"
    states = sorted({s.split("_")[0] for s in stores})
    snap_days = np.asarray(dates.day <= 10)
    for state in states:
        calendar[f"snap_{state}"] = snap_days.astype(int)

    n_weeks = int(week[-1] - week[0] + 1)
    id_rows: list[dict[str, str]] = []
    sales_rows: list[np.ndarray] = []
    price_rows: list[pd.DataFrame] = []
    for store in stores:
        state = store.split("_")[0]
        for k in range(n_items):
            dept = DEPTS[k % len(DEPTS)]
            item = f"{dept}_{k + 1:03d}"
            base = rng.uniform(0.5, 6.0)
            regular = round(float(rng.uniform(1.0, 20.0)), 2)
            release_week = int(rng.integers(0, n_weeks // 4)) if k % 4 == 3 else 0
            promo = rng.random(n_weeks) < 0.12
            weekly_price = np.where(promo, round(regular * (1 - PROMO_DISCOUNT), 2), regular)
            discount = np.where(promo, PROMO_DISCOUNT, 0.0)[week - week[0]]
            lam = (
                base
                * WEEKDAY_FACTOR[wday - 1]
                * (1 + PROMO_LIFT * discount)
                * np.where(event, EVENT_LIFT, 1.0)
                * np.where(snap_days, SNAP_LIFT, 1.0)
            )
            sold = rng.poisson(lam)
            sold[week - week[0] < release_week] = 0
            id_rows.append(
                {
                    "item_id": item,
                    "dept_id": dept,
                    "cat_id": dept.split("_")[0],
                    "store_id": store,
                    "state_id": state,
                }
            )
            sales_rows.append(sold)
            on = np.arange(release_week, n_weeks)
            price_rows.append(
                pd.DataFrame(
                    {
                        "store_id": store,
                        "item_id": item,
                        "wm_yr_wk": week[0] + on,
                        "sell_price": weekly_price[on],
                    }
                )
            )
    sales = pd.DataFrame(id_rows)
    sales = pd.concat(
        [
            sales,
            pd.DataFrame(np.stack(sales_rows), columns=[f"d_{i}" for i in range(1, n_days + 1)]),
        ],
        axis=1,
    )
    dest.mkdir(parents=True, exist_ok=True)
    for name, frame in (
        ("calendar.csv", calendar),
        ("sell_prices.csv", pd.concat(price_rows, ignore_index=True)),
        (SALES_FILE, sales),
    ):
        with atomic_text(dest / name) as fh:
            frame.to_csv(fh, index=False)
    return dest
