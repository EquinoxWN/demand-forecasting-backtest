"""Calendar, price and promotion features at item, store and total level, using only the past."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt
import pandas as pd

from .data import M5Data

Level = Literal["item", "store", "total"]
LEVELS: tuple[Level, ...] = ("item", "store", "total")
# A price at least 5% under the item's highest price so far counts as a promotion.
PROMO_THRESHOLD = 0.05
Matrix = npt.NDArray[np.float64]


@dataclass(frozen=True)
class Panel:
    """Daily sales and per-day features of every series at one hierarchy level."""

    level: Level
    keys: pd.DataFrame
    calendar: pd.DataFrame
    sales: Matrix
    features: dict[str, Matrix]

    @property
    def n_series(self) -> int:
        """Number of series."""
        return int(self.sales.shape[0])

    @property
    def n_days(self) -> int:
        """Number of days."""
        return int(self.sales.shape[1])


def calendar_features(calendar: pd.DataFrame) -> pd.DataFrame:
    """Weekday, month, year and event columns, one row per day."""
    event_1 = calendar["event_name_1"].astype(str)
    event_2 = calendar["event_name_2"].astype(str)
    return pd.DataFrame(
        {
            "date": calendar["date"],
            "wday": calendar["wday"],
            "month": calendar["month"],
            "year": calendar["year"],
            "event": ((event_1 != "") | (event_2 != "")).astype(np.int8),
            "event_type": calendar["event_type_1"].astype(str),
        }
    )


def price_matrices(data: M5Data) -> dict[str, Matrix]:
    """Per item and day: price, regular price, discount, on-shelf flag and weekly price change."""
    weeks = np.sort(data.calendar["wm_yr_wk"].unique())
    week_pos = {int(w): i for i, w in enumerate(weeks)}
    day_week = np.array([week_pos[int(w)] for w in data.calendar["wm_yr_wk"]], dtype=np.intp)
    series_pos = {
        (s, i): n
        for n, (s, i) in enumerate(zip(data.ids["store_id"], data.ids["item_id"], strict=True))
    }
    weekly = np.full((len(data.ids), len(weeks)), np.nan)
    for store, item, week, price in data.prices.itertuples(index=False, name=None):
        row = series_pos.get((store, item))
        col = week_pos.get(int(week))
        if row is not None and col is not None:
            weekly[row, col] = float(price)
    # Running maximum over weeks so far: the regular price never looks ahead.
    regular = np.fmax.accumulate(weekly, axis=1)
    previous = np.concatenate([np.full((weekly.shape[0], 1), np.nan), weekly[:, :-1]], axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        discount = np.where(np.isnan(weekly), 0.0, 1.0 - weekly / regular)
        change = np.where(np.isnan(weekly) | np.isnan(previous), 0.0, weekly / previous - 1.0)
    on_shelf = (~np.isnan(weekly)).astype(np.float64)
    return {
        "sell_price": weekly[:, day_week],
        "regular_price": regular[:, day_week],
        "discount": discount[:, day_week],
        "promo": (discount >= PROMO_THRESHOLD).astype(np.float64)[:, day_week],
        "price_change": change[:, day_week],
        "on_shelf": on_shelf[:, day_week],
    }


def _snap(data: M5Data, states: pd.Series) -> Matrix:
    """SNAP flag of each series' state per day."""
    return np.stack([data.calendar[f"snap_{s}"].to_numpy(dtype=np.float64) for s in states])


def _aggregate(
    groups: npt.NDArray[np.intp],
    n_groups: int,
    item: dict[str, Matrix],
    sales: Matrix,
    snap: Matrix,
) -> tuple[Matrix, dict[str, Matrix]]:
    """Sum sales and summarise item price features over groups of series."""

    def total(m: Matrix) -> Matrix:
        out = np.zeros((n_groups, m.shape[1]))
        np.add.at(out, groups, m)
        return out

    shelf = total(item["on_shelf"])
    counts = total(np.ones_like(snap))
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_discount = np.where(shelf > 0, total(item["discount"] * item["on_shelf"]) / shelf, 0.0)
        share_promo = np.where(shelf > 0, total(item["promo"] * item["on_shelf"]) / shelf, 0.0)
    feats = {
        "items_on_shelf": shelf,
        "mean_discount": mean_discount,
        "share_promo": share_promo,
        "snap": total(snap) / counts,
    }
    return total(sales), feats


def build_panels(data: M5Data) -> dict[Level, Panel]:
    """Item-store, store and total panels with their features."""
    sales = data.sales.astype(np.float64)
    item_feats = price_matrices(data)
    snap = _snap(data, data.ids["state_id"])
    item = Panel("item", data.ids.copy(), data.calendar, sales, {**item_feats, "snap": snap})

    store_ids = sorted(set(data.ids["store_id"]))
    store_index = {s: i for i, s in enumerate(store_ids)}
    groups = np.array([store_index[s] for s in data.ids["store_id"]], dtype=np.intp)
    store_sales, store_feats = _aggregate(groups, len(store_ids), item_feats, sales, snap)
    states = [data.ids.loc[data.ids["store_id"] == s, "state_id"].iloc[0] for s in store_ids]
    store_keys = pd.DataFrame({"store_id": store_ids, "state_id": states})
    store = Panel("store", store_keys, data.calendar, store_sales, store_feats)

    zeros = np.zeros(len(data.ids), dtype=np.intp)
    total_sales, total_feats = _aggregate(zeros, 1, item_feats, sales, snap)
    total_keys = pd.DataFrame({"total_id": ["Total"]})
    total = Panel("total", total_keys, data.calendar, total_sales, total_feats)
    return {"item": item, "store": store, "total": total}


def iter_frames(panel: Panel, batch: int = 500) -> Iterator[pd.DataFrame]:
    """Long feature table (one row per series and day) in batches of series."""
    if batch < 1:
        raise ValueError("batch must be at least 1")
    cal = calendar_features(panel.calendar)
    for start in range(0, panel.n_series, batch):
        stop = min(start + batch, panel.n_series)
        n = stop - start
        rows = np.repeat(np.arange(start, stop), panel.n_days)
        frame = panel.keys.iloc[rows].reset_index(drop=True)
        for col in cal.columns:
            frame[col] = np.tile(cal[col].to_numpy(), n)
        frame["sales"] = panel.sales[start:stop].reshape(-1)
        for name, values in panel.features.items():
            frame[name] = values[start:stop].reshape(-1)
        yield frame
