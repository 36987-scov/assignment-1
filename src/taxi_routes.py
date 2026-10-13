"""Taxi route revenue, pricing and fleet planning (Mini-Project 5).

Contents
--------
Route
    Daily passenger counts and fare, with revenue and descriptive statistics.
equilibrium, excess_demand
    Supply-demand equilibrium as a 2 x 2 linear system (scipy.linalg.solve).
Forecaster (abstract) and subclasses
    MovingAverageForecaster, SESForecaster, LinearTrendForecaster,
    SeasonalNaiveForecaster (extension). All make one-step-ahead forecasts.
backtest, tune_alpha
    Rolling-origin (walk-forward) evaluation and grid search for alpha.
vehicles_needed
    Fleet size from forecast demand, vehicle capacity and a safety buffer.
simulate_weekly
    Extension: seeded daily passenger counts with a weekly pattern.

This hierarchy is separate from the population Forecaster in Mini-Project 1:
those models forecast annual levels several years ahead; these make
one-day-ahead forecasts of daily counts.
"""

from __future__ import annotations

import math
import statistics
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import linalg

__all__ = [
    "Route",
    "equilibrium",
    "excess_demand",
    "Forecaster",
    "MovingAverageForecaster",
    "SESForecaster",
    "LinearTrendForecaster",
    "SeasonalNaiveForecaster",
    "backtest",
    "tune_alpha",
    "vehicles_needed",
    "simulate_weekly",
    "WEEKDAYS",
]

WEEKDAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")


def _counts(values: ArrayLike, label: str = "passengers") -> NDArray[np.float64]:
    """Validate a non-empty 1-D array of finite, non-negative numbers."""
    try:
        arr = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric.") from exc
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError(f"{label} must be a non-empty 1-D sequence.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{label} must be finite.")
    if np.any(arr < 0):
        raise ValueError(f"{label} must not be negative.")
    return arr


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------

class Route:
    """One taxi route: daily passenger counts and a flat fare (UGX per passenger)."""

    def __init__(self, name: str, passengers: ArrayLike, fare_ugx: float, source: str) -> None:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("name must be a non-empty string.")
        if not math.isfinite(fare_ugx) or fare_ugx <= 0:
            raise ValueError("fare_ugx must be a positive number.")
        self.name: str = name.strip()
        self.passengers: NDArray[np.float64] = _counts(passengers)
        self.fare_ugx: float = float(fare_ugx)
        self.source: str = source

    def __repr__(self) -> str:
        return f"Route({self.name!r}, days={len(self)}, fare=UGX {self.fare_ugx:,.0f})"

    def __len__(self) -> int:
        return int(self.passengers.size)

    def daily_revenue(self) -> NDArray[np.float64]:
        """Revenue for each day in UGX (passengers x fare)."""
        return self.passengers * self.fare_ugx

    def total_revenue(self) -> float:
        """Total revenue over all days in UGX."""
        return float(self.daily_revenue().sum())

    def stats(self) -> dict[str, float]:
        """Mean, sample variance and sample SD of daily passengers (``statistics``).

        Raises
        ------
        ValueError
            If fewer than 2 days (sample variance undefined).
        """
        if len(self) < 2:
            raise ValueError("At least 2 days are needed for variance.")
        data = self.passengers.tolist()
        return {"mean": statistics.mean(data), "variance": statistics.variance(data),
                "stdev": statistics.stdev(data)}


# ---------------------------------------------------------------------------
# Supply-demand equilibrium
# ---------------------------------------------------------------------------

def equilibrium(
    demand: tuple[float, float] = (120.0, -0.02),
    supply: tuple[float, float] = (10.0, 0.03),
) -> tuple[float, float]:
    """Solve Qd = a_d + b_d P and Qs = a_s + b_s P for the equilibrium (P, Q).

    Rearranged as ``Q - b P = a`` for both lines, i.e.
    ``[[1, -b_d], [1, -b_s]] @ [Q, P] = [a_d, a_s]``, solved with scipy.

    Raises
    ------
    ValueError
        If the slopes are equal (parallel lines: no unique equilibrium).
    """
    (a_d, b_d), (a_s, b_s) = demand, supply
    A = np.array([[1.0, -b_d], [1.0, -b_s]])
    if math.isclose(b_d, b_s):
        raise ValueError("Demand and supply have the same slope: no unique equilibrium.")
    q, p = linalg.solve(A, np.array([a_d, a_s], dtype=float))
    return float(p), float(q)


def excess_demand(
    price: float,
    demand: tuple[float, float] = (120.0, -0.02),
    supply: tuple[float, float] = (10.0, 0.03),
) -> float:
    """Qd(P) - Qs(P): positive means more riders than seats offered at that price."""
    return (demand[0] + demand[1] * price) - (supply[0] + supply[1] * price)


# ---------------------------------------------------------------------------
# Forecasters
# ---------------------------------------------------------------------------

class Forecaster(ABC):
    """Abstract one-step-ahead forecaster for daily counts.

    Subclasses set ``min_history`` and implement ``_fit`` and ``_predict``;
    validation happens once, here.
    """

    min_history: int = 1

    def __init__(self) -> None:
        self._history: NDArray[np.float64] | None = None

    @property
    def name(self) -> str:
        """Readable name for tables and legends."""
        return type(self).__name__.replace("Forecaster", "")

    def fit(self, history: ArrayLike) -> Forecaster:
        """Learn from past days and return ``self``.

        Raises
        ------
        ValueError
            If the history is invalid or shorter than ``min_history``.
        """
        h = _counts(history, "history")
        if h.size < self.min_history:
            raise ValueError(f"{self.name} needs at least {self.min_history} days of history, got {h.size}.")
        self._history = h
        self._fit(h)
        return self

    def predict(self) -> float:
        """Forecast the next day.

        Raises
        ------
        RuntimeError
            If called before ``fit``.
        """
        if self._history is None:
            raise RuntimeError(f"{self.name}: call fit() before predict().")
        return float(self._predict())

    @abstractmethod
    def _fit(self, history: NDArray[np.float64]) -> None:
        """Learn parameters from validated history."""

    @abstractmethod
    def _predict(self) -> float:
        """Return the next-day forecast."""


class MovingAverageForecaster(Forecaster):
    """Mean of the last ``window`` days (the original 3-day method)."""

    def __init__(self, window: int = 3) -> None:
        super().__init__()
        if isinstance(window, bool) or not isinstance(window, int) or window < 1:
            raise ValueError("window must be a positive integer.")
        self.window = window
        self.min_history = window

    @property
    def name(self) -> str:
        return f"MA({self.window})"

    def _fit(self, history: NDArray[np.float64]) -> None:
        self._level = float(history[-self.window:].mean())

    def _predict(self) -> float:
        return self._level


class SESForecaster(Forecaster):
    """Simple exponential smoothing: s = alpha * x + (1 - alpha) * s_previous.

    The level starts at the first observation (stated assumption). With
    ``alpha = 1`` the forecast is the last value (naive forecast).
    """

    def __init__(self, alpha: float = 0.5) -> None:
        super().__init__()
        if not 0 < alpha <= 1:
            raise ValueError("alpha must be in (0, 1].")
        self.alpha = float(alpha)

    @property
    def name(self) -> str:
        return f"SES(alpha={self.alpha:.2f})"

    def _fit(self, history: NDArray[np.float64]) -> None:
        level = history[0]
        for x in history[1:]:
            level = self.alpha * x + (1 - self.alpha) * level
        self._level = float(level)

    def _predict(self) -> float:
        return self._level


class LinearTrendForecaster(Forecaster):
    """Least-squares line through all past days (``np.polyfit``), extended one day."""

    min_history = 2

    def _fit(self, history: NDArray[np.float64]) -> None:
        t = np.arange(history.size)
        self.slope_, self.intercept_ = (float(v) for v in np.polyfit(t, history, deg=1))
        self._next_t = history.size

    def _predict(self) -> float:
        return self.intercept_ + self.slope_ * self._next_t


class SeasonalNaiveForecaster(Forecaster):
    """Tomorrow = the same weekday last week (value ``season`` days ago)."""

    def __init__(self, season: int = 7) -> None:
        super().__init__()
        if isinstance(season, bool) or not isinstance(season, int) or season < 1:
            raise ValueError("season must be a positive integer.")
        self.season = season
        self.min_history = season

    def _fit(self, history: NDArray[np.float64]) -> None:
        self._value = float(history[-self.season])

    def _predict(self) -> float:
        return self._value


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def backtest(
    series: ArrayLike,
    make_model: Callable[[], Forecaster],
    start_day: int = 4,
    end_day: int | None = None,
) -> dict[str, object]:
    """Rolling-origin (walk-forward) one-step-ahead evaluation.

    For each target day t (1-based) from ``start_day`` to ``end_day``, a fresh
    model is fitted on days 1..t-1 only and forecasts day t, so no future
    information leaks into a forecast.

    Returns
    -------
    dict with ``days`` (1-based), ``forecasts``, ``actuals``, ``errors`` and ``mae``.
    """
    x = _counts(series, "series")
    end = x.size if end_day is None else end_day
    if not 2 <= start_day <= end <= x.size:
        raise ValueError(f"Need 2 <= start_day <= end_day <= {x.size}.")
    days = np.arange(start_day, end + 1)
    forecasts = np.array([make_model().fit(x[: d - 1]).predict() for d in days])
    actuals = x[days - 1]
    errors = actuals - forecasts
    return {"days": days, "forecasts": forecasts, "actuals": actuals, "errors": errors,
            "mae": float(np.mean(np.abs(errors)))}


def tune_alpha(
    series: ArrayLike,
    grid: Sequence[float] | None = None,
    start_day: int = 4,
) -> tuple[float, dict[float, float]]:
    """Grid-search SES alpha by backtest MAE; ties go to the smaller alpha.

    The default grid is 0.01, 0.02, ..., 1.00. Note: scoring alpha on the same
    days used to choose it makes the winning MAE optimistic.
    """
    grid = [round(a, 2) for a in np.arange(0.01, 1.0001, 0.01)] if grid is None else list(grid)
    if not grid:
        raise ValueError("grid must not be empty.")
    maes = {float(a): backtest(series, lambda a=a: SESForecaster(a), start_day)["mae"] for a in grid}
    best = min(maes, key=lambda a: (round(maes[a], 12), a))
    return best, maes


# ---------------------------------------------------------------------------
# Fleet planning
# ---------------------------------------------------------------------------

def vehicles_needed(
    passengers: float,
    trips_per_day: int = 8,
    seats: int = 14,
    buffer: float = 0.15,
) -> int:
    """Vehicles to deploy: ceil(passengers x (1 + buffer) / (trips x seats)), at least 1.

    Rounding up guarantees enough seats; a minimum of 1 keeps the route served.

    Raises
    ------
    ValueError
        If passengers is negative or the capacity parameters are not positive.
    """
    if not math.isfinite(passengers) or passengers < 0:
        raise ValueError("passengers must be a finite, non-negative number.")
    if trips_per_day < 1 or seats < 1 or buffer < 0:
        raise ValueError("trips_per_day and seats must be >= 1 and buffer >= 0.")
    capacity = trips_per_day * seats
    return max(1, math.ceil(round(passengers * (1 + buffer) / capacity, 9)))


# ---------------------------------------------------------------------------
# Extension: simulated weekly pattern
# ---------------------------------------------------------------------------

def simulate_weekly(
    base: float,
    days: int = 60,
    seed: int = 2026,
    friday_factor: float = 1.30,
    sunday_factor: float = 0.70,
    noise_sd: float = 0.08,
) -> NDArray[np.float64]:
    """Seeded daily passenger counts with a weekly pattern (day 1 = Monday).

    ``count = round(base * weekday_factor * (1 + N(0, noise_sd)))``, floored at 0,
    with Friday and Sunday factors as given and 1.0 on other days.
    """
    if base <= 0 or days < 1 or noise_sd < 0:
        raise ValueError("Need base > 0, days >= 1 and noise_sd >= 0.")
    rng = np.random.default_rng(seed)
    factors = np.array([{"Fri": friday_factor, "Sun": sunday_factor}.get(WEEKDAYS[d % 7], 1.0) for d in range(days)])
    noise = 1 + rng.normal(0, noise_sd, size=days)
    return np.maximum(0, np.round(base * factors * noise))
