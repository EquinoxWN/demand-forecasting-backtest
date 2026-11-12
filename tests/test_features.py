"""Feature values, the no-look-ahead property and the hierarchy sums."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from demand_forecasting_backtest.data import M5Data, load_m5
from demand_forecasting_backtest.features import (
    build_panels,
    calendar_features,
    iter_frames,
    price_matrices,
)
from demand_forecasting_backtest.synthetic import PROMO_DISCOUNT, write_synthetic_m5


def test_price_features_on_a_worked_example(tiny: M5Data) -> None:
    f = price_matrices(tiny)
    week = np.repeat([0, 1, 2], 7)
    a = {k: v[0][::7] for k, v in f.items()}
    np.testing.assert_allclose(a["sell_price"], [10, 8, 10])
    np.testing.assert_allclose(a["regular_price"], [10, 10, 10])
    np.testing.assert_allclose(a["discount"], [0.0, 0.2, 0.0])
    np.testing.assert_allclose(a["promo"], [0, 1, 0])
    np.testing.assert_allclose(a["price_change"], [0.0, -0.2, 0.25])
    b = {k: v[1][::7] for k, v in f.items()}
    np.testing.assert_allclose(b["on_shelf"], [0, 1, 1])
    assert np.isnan(b["sell_price"][0])
    np.testing.assert_allclose(b["regular_price"][1:], [4, 5])
    np.testing.assert_allclose(b["discount"], [0, 0, 0])
    np.testing.assert_allclose(b["price_change"], [0.0, 0.0, 0.25])
    # Every day of a week carries that week's price.
    np.testing.assert_allclose(f["sell_price"][0], np.array([10, 8, 10])[week])


def test_calendar_features_flag_events(tiny: M5Data) -> None:
    cal = calendar_features(tiny.calendar)
    assert cal["event"].tolist() == [1 if d == 3 else 0 for d in range(21)]
    assert cal["event_type"].iloc[3] == "National"
    assert cal["wday"].iloc[0] == 1


def test_features_never_look_ahead(m5_dir: Path) -> None:
    data = load_m5(m5_dir, ["CA_1", "TX_1"])
    base = build_panels(data)
    rng = np.random.default_rng(1)
    for cut in rng.integers(7, data.sales.shape[1] - 7, size=5):
        cut_week = int(data.calendar["wm_yr_wk"].iloc[cut])
        prices = data.prices.copy()
        future = prices["wm_yr_wk"] > cut_week
        prices.loc[future, "sell_price"] = prices.loc[future, "sell_price"] * 3.0
        sales = data.sales.copy()
        sales[:, cut + 1 :] = 999
        changed = build_panels(replace(data, prices=prices, sales=sales))
        last_day = int(np.flatnonzero(data.calendar["wm_yr_wk"].to_numpy() == cut_week)[-1])
        for level, panel in base.items():
            other = changed[level]
            np.testing.assert_array_equal(panel.sales[:, : cut + 1], other.sales[:, : cut + 1])
            for name, values in panel.features.items():
                np.testing.assert_array_equal(
                    values[:, : last_day + 1], other.features[name][:, : last_day + 1], err_msg=name
                )


def test_levels_add_up(m5_dir: Path) -> None:
    data = load_m5(m5_dir, ["CA_1", "TX_1"])
    panels = build_panels(data)
    item, store, total = panels["item"], panels["store"], panels["total"]
    assert (item.n_series, store.n_series, total.n_series) == (16, 2, 1)
    for i, sid in enumerate(store.keys["store_id"]):
        mask = (item.keys["store_id"] == sid).to_numpy()
        np.testing.assert_array_equal(store.sales[i], item.sales[mask].sum(axis=0))
        np.testing.assert_array_equal(
            store.features["items_on_shelf"][i], item.features["on_shelf"][mask].sum(axis=0)
        )
    np.testing.assert_array_equal(total.sales[0], store.sales.sum(axis=0))
    np.testing.assert_array_equal(total.sales[0], data.sales.sum(axis=0))
    assert store.keys["state_id"].tolist() == ["CA", "TX"]


def test_store_and_total_summaries_on_a_worked_example(tiny: M5Data) -> None:
    panels = build_panels(tiny)
    store = panels["store"].features
    # Week 2: A at 20% off, B on shelf at full price.
    assert store["items_on_shelf"][0][7] == 2
    assert store["mean_discount"][0][7] == pytest.approx(0.1)
    assert store["share_promo"][0][7] == pytest.approx(0.5)
    assert store["share_promo"][0][0] == 0.0
    assert store["snap"][0].tolist() == [1.0] * 5 + [0.0] * 16


def test_total_snap_is_the_share_of_series_in_snap_states(m5_dir: Path) -> None:
    cal_path = m5_dir / "calendar.csv"
    text = cal_path.read_text().splitlines()
    header = text[0].split(",")
    tx = header.index("snap_TX")
    rows = [line.split(",") for line in text[1:]]
    for r in rows:
        r[tx] = "0"
    cal_path.write_text("\n".join([text[0], *(",".join(r) for r in rows)]) + "\n")
    panels = build_panels(load_m5(m5_dir, ["CA_1", "TX_1"]))
    snap_ca = panels["item"].features["snap"][0]
    assert set(np.unique(panels["total"].features["snap"][0])) <= {0.0, 0.5}
    np.testing.assert_allclose(panels["total"].features["snap"][0], snap_ca / 2)


def test_planted_promotion_lift_shows_in_the_features(tmp_path: Path) -> None:
    src = write_synthetic_m5(tmp_path / "s", stores=("CA_1",), n_items=30, n_days=364, seed=3)
    item = build_panels(load_m5(src, ["CA_1"]))["item"]
    shelf = item.features["on_shelf"] == 1
    promo = item.features["promo"] == 1
    assert np.allclose(item.features["discount"][promo], PROMO_DISCOUNT, atol=0.01)
    ratio = item.sales[shelf & promo].mean() / item.sales[shelf & ~promo].mean()
    assert ratio > 1.3  # planted 1.6x
    assert item.sales[~shelf].sum() == 0  # nothing sells before release


def test_feature_table_matches_the_matrices(tiny: M5Data) -> None:
    panel = build_panels(tiny)["item"]
    frames = list(iter_frames(panel, batch=1))
    assert len(frames) == 2
    frame = frames[1]
    assert len(frame) == 21
    assert (frame["item_id"] == "B").all()
    np.testing.assert_array_equal(frame["sales"], panel.sales[1])
    np.testing.assert_array_equal(frame["on_shelf"], panel.features["on_shelf"][1])
    assert frame["date"].iloc[0].strftime("%Y-%m-%d") == "2011-01-29"
    assert {"wday", "event", "snap", "discount", "promo"} <= set(frame.columns)
    with pytest.raises(ValueError, match="batch"):
        next(iter_frames(panel, batch=0))


def test_a_small_price_dip_is_a_discount_but_not_a_promotion(tiny: M5Data) -> None:
    prices = tiny.prices.copy()
    prices.loc[1, "sell_price"] = 9.7  # item A, week 2: 3% under its regular price
    f = price_matrices(replace(tiny, prices=prices))
    np.testing.assert_allclose(f["discount"][0][::7], [0.0, 0.03, 0.0])
    np.testing.assert_array_equal(f["promo"][0][::7], [0, 0, 0])


def test_promotion_share_counts_only_items_on_the_shelf(tiny: M5Data) -> None:
    prices = tiny.prices[~((tiny.prices["item_id"] == "B") & (tiny.prices["wm_yr_wk"] == 11102))]
    store = build_panels(replace(tiny, prices=prices))["store"].features
    # Week 2: only A is on the shelf, and it is on promotion.
    assert store["items_on_shelf"][0][7] == 1
    assert store["share_promo"][0][7] == 1.0
    assert store["mean_discount"][0][7] == pytest.approx(0.2)
