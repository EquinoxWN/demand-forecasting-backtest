"""Seasonal-naive baseline, simple challengers, error metrics and the beat-the-baseline gate."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from .features import LEVELS, Level, Panel

Matrix = npt.NDArray[np.float64]
Forecaster = Callable[[Matrix, int], Matrix]
SEASON = 7
BASELINE = "seasonal_naive"


def _check(history: Matrix, horizon: int, min_history: int) -> None:
    """Validate a history matrix and a horizon."""
    if history.ndim != 2 or history.shape[0] == 0:
        raise ValueError("history must be a non-empty 2-D array (series x days)")
    if horizon < 1:
        raise ValueError("horizon must be at least 1")
    if history.shape[1] < min_history:
        raise ValueError(f"need at least {min_history} days of history, got {history.shape[1]}")
    if not np.isfinite(history).all():
        raise ValueError("history has missing or infinite values")


def seasonal_naive(history: Matrix, horizon: int, season: int = SEASON) -> Matrix:
    """Repeat the last full season: next Monday's forecast is last Monday's sales."""
    if season < 1:
        raise ValueError("season must be at least 1")
    _check(history, horizon, season)
    last = history[:, -season:]
    reps = -(-horizon // season)
    return np.tile(last, reps)[:, :horizon]


def naive_last(history: Matrix, horizon: int) -> Matrix:
    """Repeat the last observed day."""
    _check(history, horizon, 1)
    return np.repeat(history[:, -1:], horizon, axis=1)


def window_mean(history: Matrix, horizon: int, window: int = 28) -> Matrix:
    """Flat forecast at the mean of the last window days."""
    _check(history, horizon, window)
    return np.repeat(history[:, -window:].mean(axis=1, keepdims=True), horizon, axis=1)


def seasonal_mean(history: Matrix, horizon: int, weeks: int = 4, season: int = SEASON) -> Matrix:
    """Mean of the same weekday over the last few weeks, repeated."""
    _check(history, horizon, weeks * season)
    block = history[:, -weeks * season :].reshape(history.shape[0], weeks, season).mean(axis=1)
    return np.tile(block, -(-horizon // season))[:, :horizon]


MODELS: Mapping[str, Forecaster] = {
    BASELINE: seasonal_naive,
    "naive_last": naive_last,
    "mean_28d": window_mean,
    "seasonal_mean_4w": seasonal_mean,
}


def _same_shape(actual: Matrix, forecast: Matrix) -> None:
    """Fail unless actual and forecast are finite arrays of one shape."""
    if actual.shape != forecast.shape or actual.size == 0:
        raise ValueError(f"shape mismatch or empty: {actual.shape} vs {forecast.shape}")
    if not (np.isfinite(actual).all() and np.isfinite(forecast).all()):
        raise ValueError("actual and forecast must be finite")


def wape(actual: Matrix, forecast: Matrix) -> float:
    """Weighted absolute percentage error: sum |error| / sum |actual|."""
    _same_shape(actual, forecast)
    denom = float(np.abs(actual).sum())
    if denom == 0.0:
        raise ValueError("WAPE is undefined when every actual value is zero")
    return float(np.abs(actual - forecast).sum()) / denom


def mase(
    actual: Matrix, forecast: Matrix, history: Matrix, season: int = SEASON
) -> tuple[float, int]:
    """Mean over series of MAE scaled by the in-sample seasonal-naive MAE; series count used.

    Series whose history never changes from one season to the next have no scale and are left
    out; when none is left the result is NaN with a count of 0.
    """
    _same_shape(actual, forecast)
    if history.shape[0] != actual.shape[0] or history.shape[1] <= season:
        raise ValueError("history must have one row per series and more than one season")
    scale = np.abs(history[:, season:] - history[:, :-season]).mean(axis=1)
    usable = scale > 0
    if not usable.any():
        return float("nan"), 0
    mae = np.abs(actual - forecast).mean(axis=1)
    return float((mae[usable] / scale[usable]).mean()), int(usable.sum())


@dataclass(frozen=True)
class Score:
    """Holdout error of one model at one level."""

    level: str
    model: str
    wape: float
    mase: float
    mase_series: int
    counts: bool


def beats(candidate_wape: float, baseline_wape: float) -> bool:
    """A model counts only if its WAPE is strictly lower than the baseline's."""
    return candidate_wape < baseline_wape


def holdout(panel: Panel, horizon: int) -> tuple[Matrix, Matrix]:
    """Split each series into history and the last horizon days."""
    if horizon < 1 or horizon >= panel.n_days:
        raise ValueError(f"horizon must be between 1 and {panel.n_days - 1}")
    return panel.sales[:, :-horizon], panel.sales[:, -horizon:]


def evaluate(
    panels: Mapping[Level, Panel], horizon: int, models: Mapping[str, Forecaster] = MODELS
) -> list[Score]:
    """Score every model per level on the final horizon days, baseline first."""
    if BASELINE not in models:
        raise ValueError(f"the {BASELINE} baseline must be among the models")
    order = [BASELINE, *[m for m in models if m != BASELINE]]
    scores: list[Score] = []
    for level in [lv for lv in LEVELS if lv in panels]:
        history, actual = holdout(panels[level], horizon)
        base_wape = 0.0
        for name in order:
            forecast = models[name](history, horizon)
            err = wape(actual, forecast)
            scaled, used = mase(actual, forecast, history)
            if name == BASELINE:
                base_wape = err
            ok = name == BASELINE or beats(err, base_wape)
            scores.append(Score(level, name, err, scaled, used, ok))
    return scores
