"""Seasonal-naive baseline, challengers, metrics and the beat-the-baseline gate."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from demand_forecasting_backtest.baseline import (
    BASELINE,
    MODELS,
    Matrix,
    beats,
    evaluate,
    holdout,
    mase,
    naive_last,
    seasonal_mean,
    seasonal_naive,
    wape,
    window_mean,
)
from demand_forecasting_backtest.data import M5Data, load_m5
from demand_forecasting_backtest.features import build_panels
from demand_forecasting_backtest.synthetic import write_synthetic_m5


def test_seasonal_naive_repeats_the_last_week_in_weekday_order() -> None:
    history = np.arange(1, 15, dtype=float)[None, :]  # days 1..14
    fc = seasonal_naive(history, 10)
    np.testing.assert_array_equal(fc[0], [8, 9, 10, 11, 12, 13, 14, 8, 9, 10])
    # Forecast day 14 + j has the same weekday as the history day it copies.
    full = np.arange(1, 25, dtype=float)
    assert all((full[14 + j] - fc[0][j]) % 7 == 0 for j in range(10))


def test_seasonal_naive_with_another_season() -> None:
    fc = seasonal_naive(np.array([[1.0, 2, 3, 4, 5]]), 3, season=2)
    np.testing.assert_array_equal(fc, [[4, 5, 4]])


def test_challengers_on_worked_examples() -> None:
    h = np.array([[0.0] * 21 + [1, 2, 3, 4, 5, 6, 7]])
    np.testing.assert_array_equal(naive_last(h, 3), [[7, 7, 7]])
    np.testing.assert_allclose(window_mean(h, 2), [[1.0, 1.0]])
    np.testing.assert_allclose(seasonal_mean(h, 8), [[0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 0.25]])


@pytest.mark.parametrize(
    ("history", "horizon", "message"),
    [
        (np.ones((2, 10)), 0, "horizon"),
        (np.ones((2, 6)), 3, "at least 7"),
        (np.ones(10), 3, "2-D"),
        (np.ones((0, 10)), 3, "non-empty"),
        (np.array([[1.0] * 9 + [np.nan]]), 3, "missing"),
    ],
)
def test_forecasters_refuse_bad_input(history: Matrix, horizon: int, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        seasonal_naive(history, horizon)


def test_wape_on_a_worked_example_and_its_limits() -> None:
    actual = np.array([[10.0, 0.0], [5.0, 5.0]])
    forecast = np.array([[8.0, 1.0], [5.0, 9.0]])
    assert wape(actual, forecast) == pytest.approx((2 + 1 + 0 + 4) / 20)
    with pytest.raises(ValueError, match="undefined"):
        wape(np.zeros((1, 3)), np.ones((1, 3)))
    with pytest.raises(ValueError, match="shape"):
        wape(actual, forecast[:, :1])
    with pytest.raises(ValueError, match="finite"):
        wape(actual, np.full((2, 2), np.inf))


def test_mase_scales_each_series_and_skips_flat_ones() -> None:
    history = np.array([[0.0, 2, 0, 2], [3.0, 3, 3, 3]])
    actual = np.array([[1.0, 1.0], [3.0, 3.0]])
    forecast = np.array([[0.0, 3.0], [9.0, 9.0]])
    value, used = mase(actual, forecast, history, season=1)
    # Series 1: MAE 1.5 / scale 2 = 0.75; series 2 is flat and left out.
    assert value == pytest.approx(0.75)
    assert used == 1
    flat_value, flat_used = mase(actual, forecast, np.ones((2, 4)), season=1)
    assert np.isnan(flat_value)
    assert flat_used == 0
    with pytest.raises(ValueError, match="one row per series"):
        mase(actual, forecast, history[:1], season=1)


def test_a_model_counts_only_when_strictly_better() -> None:
    assert beats(0.29, 0.30)
    assert not beats(0.30, 0.30)
    assert not beats(0.31, 0.30)


def test_the_holdout_is_the_last_horizon_days(tiny: M5Data) -> None:
    panel = build_panels(tiny)["total"]
    history, actual = holdout(panel, 7)
    assert history.shape == (1, 14)
    assert actual.shape == (1, 7)
    np.testing.assert_array_equal(actual, panel.sales[:, -7:])
    for bad in (0, 21):
        with pytest.raises(ValueError, match="horizon"):
            holdout(panel, bad)


def test_evaluate_scores_the_baseline_first_and_flags_winners(m5_dir: Path) -> None:
    panels = build_panels(load_m5(m5_dir, ["CA_1", "TX_1"]))
    scores = evaluate(panels, 28)
    assert [s.level for s in scores] == ["item"] * 4 + ["store"] * 4 + ["total"] * 4
    for level in ("item", "store", "total"):
        rows = [s for s in scores if s.level == level]
        assert rows[0].model == BASELINE
        for s in rows[1:]:
            assert s.counts == (s.wape < rows[0].wape)


def test_planted_weekly_pattern_seasonal_naive_beats_last_value(tmp_path: Path) -> None:
    src = write_synthetic_m5(tmp_path / "s", stores=("CA_1",), n_items=40, n_days=364, seed=5)
    scores = {(s.level, s.model): s for s in evaluate(build_panels(load_m5(src, ["CA_1"])), 28)}
    for level in ("store", "total"):
        assert scores[(level, BASELINE)].wape < scores[(level, "naive_last")].wape
        assert not scores[(level, "naive_last")].counts


def test_a_perfectly_weekly_series_has_zero_baseline_error(tiny: M5Data) -> None:
    weekly = np.tile(np.array([5, 1, 1, 1, 1, 2, 9], dtype=np.int16), 3)
    data = M5Data(tiny.ids.iloc[:1], weekly[None, :], tiny.calendar, tiny.prices)
    scores = evaluate(build_panels(data), 7, {BASELINE: seasonal_naive})
    assert scores[0].model == BASELINE
    assert scores[0].wape == 0.0


def test_forecasts_never_see_the_holdout(m5_dir: Path) -> None:
    seen: list[Matrix] = []

    def spy(history: Matrix, horizon: int) -> Matrix:
        seen.append(history.copy())
        return seasonal_naive(history, horizon)

    panels = build_panels(load_m5(m5_dir, ["CA_1"]))
    evaluate(panels, 28, {BASELINE: spy})
    for history, panel in zip(seen, panels.values(), strict=True):
        assert history.shape[1] == panel.n_days - 28
        np.testing.assert_array_equal(history, panel.sales[:, :-28])


def test_evaluate_needs_the_baseline(m5_dir: Path) -> None:
    panels = build_panels(load_m5(m5_dir, ["CA_1"]))
    with pytest.raises(ValueError, match="baseline"):
        evaluate(panels, 28, {"naive_last": MODELS["naive_last"]})


def test_mase_uses_the_seasonal_lag_for_its_scale() -> None:
    history = np.array([[1.0, 2, 3, 4, 5, 6, 7, 2, 3, 4, 5, 6, 7, 8]])
    # Lag-7 differences are all 1, so the scale is 1 (a lag-1 scale would be 17/13).
    value, used = mase(np.array([[3.0]]), np.array([[1.0]]), history)
    assert value == pytest.approx(2.0)
    assert used == 1
