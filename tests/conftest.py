"""Shared fixtures: small M5-format data sets and hand-built inputs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from demand_forecasting_backtest.data import M5Data
from demand_forecasting_backtest.synthetic import write_synthetic_m5


@pytest.fixture
def m5_dir(tmp_path: Path) -> Path:
    """Synthetic M5 files for two stores in two states."""
    return write_synthetic_m5(tmp_path / "m5", stores=("CA_1", "TX_1"), n_items=8, n_days=140)


@pytest.fixture
def tiny() -> M5Data:
    """One CA store, two items, three weeks; item B appears in week two."""
    days = 21
    calendar = pd.DataFrame(
        {
            "date": pd.date_range("2011-01-29", periods=days, freq="D"),
            "wm_yr_wk": np.repeat([11101, 11102, 11103], 7),
            "wday": np.arange(days) % 7 + 1,
            "month": 1,
            "year": 2011,
            "event_name_1": ["Holiday" if d == 3 else "" for d in range(days)],
            "event_type_1": ["National" if d == 3 else "" for d in range(days)],
            "event_name_2": "",
            "event_type_2": "",
            "snap_CA": np.array([1 if d < 5 else 0 for d in range(days)], dtype=np.int8),
        }
    )
    ids = pd.DataFrame(
        {
            "item_id": ["A", "B"],
            "dept_id": ["D", "D"],
            "cat_id": ["C", "C"],
            "store_id": ["CA_1", "CA_1"],
            "state_id": ["CA", "CA"],
        }
    )
    sales = np.vstack([np.arange(days) % 5, np.r_[np.zeros(7), np.full(14, 2)]]).astype(np.int16)
    prices = pd.DataFrame(
        {
            "store_id": ["CA_1"] * 5,
            "item_id": ["A", "A", "A", "B", "B"],
            "wm_yr_wk": [11101, 11102, 11103, 11102, 11103],
            "sell_price": [10.0, 8.0, 10.0, 4.0, 5.0],
        }
    )
    return M5Data(ids=ids, sales=sales, calendar=calendar, prices=prices)
