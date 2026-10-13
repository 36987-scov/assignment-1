"""Lake Victoria fish stock and export risk model (Mini-Project 3).

Units and timing conventions (used everywhere in this module)
-------------------------------------------------------------
* Time step: 1 week. ``r`` and ``h`` are per-week rates.
* Stock and harvest: tonnes. Price: UGX per kg. Revenue: UGX.
* In week t (t = 0 .. weeks-1) the harvest is ``H[t] = h * N[t]`` tonnes, taken
  from the stock at the start of the week, and

      N[t+1] = N[t] + r * N[t] * (1 - N[t] / K) - h * N[t]

  so a 52-week run has 52 harvests and 53 stock values (N[0] .. N[52]).
* Weekly revenue: ``R[t] = H[t] * 1000 kg/t * P[t]``. Annual revenue is the sum
  of 52 weekly revenues.

Contents
--------
fibonacci (re-used from src.population), FishStock, StockPath, PriceModel,
weekly_revenue, monte_carlo_annual_revenue, value_at_risk, RiskAssessor.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from src.population import fibonacci  # tested there; re-used rather than duplicated

__all__ = [
    "fibonacci",
    "FishStock",
    "StockPath",
    "PriceModel",
    "weekly_revenue",
    "monte_carlo_annual_revenue",
    "value_at_risk",
    "RiskAssessor",
    "KG_PER_TONNE",
    "WEEKS_PER_YEAR",
]

KG_PER_TONNE = 1000.0
WEEKS_PER_YEAR = 52


# ---------------------------------------------------------------------------
# Stock dynamics
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StockPath:
    """Result of a stock simulation.

    Attributes
    ----------
    stock_tonnes:
        Stock at the start of each week plus the final stock (weeks + 1 values).
    harvest_tonnes:
        Tonnes harvested in each week (weeks values).
    h:
        Harvest rate used (per week; 0 in closed weeks).
    """

    stock_tonnes: NDArray[np.float64]
    harvest_tonnes: NDArray[np.float64]
    h: float

    @property
    def final_stock(self) -> float:
        """Stock after the last week, in tonnes."""
        return float(self.stock_tonnes[-1])

    @property
    def total_harvest(self) -> float:
        """Total tonnes harvested over the run."""
        return float(self.harvest_tonnes.sum())


class FishStock:
    """Discrete logistic fish stock with proportional harvesting.

    Parameters
    ----------
    r:
        Intrinsic growth rate per week (> 0).
    K:
        Carrying capacity in tonnes (> 0).
    n0:
        Initial stock in tonnes (>= 0).
    """

    def __init__(self, r: float = 0.4, K: float = 10_000.0, n0: float = 4_000.0) -> None:
        for label, value in (("r", r), ("K", K)):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{label} must be a positive finite number.")
        if not np.isfinite(n0) or n0 < 0:
            raise ValueError("n0 must be a non-negative finite number.")
        self.r, self.K, self.n0 = float(r), float(K), float(n0)

    def __repr__(self) -> str:
        return f"FishStock(r={self.r}, K={self.K:,.0f} t, n0={self.n0:,.0f} t)"

    # -- theory --------------------------------------------------------------

    @property
    def msy(self) -> float:
        """Maximum sustainable yield rK/4, in tonnes per week."""
        return self.r * self.K / 4.0

    @property
    def msy_harvest_rate(self) -> float:
        """Harvest rate h = r/2 that keeps the stock at K/2, where yield = MSY."""
        return self.r / 2.0

    def equilibrium(self, h: float) -> float:
        """Non-zero steady state K(1 - h/r); 0 if h >= r (stock cannot persist)."""
        return max(0.0, self.K * (1.0 - h / self.r))

    def sustainable_yield(self, h: float) -> float:
        """Weekly harvest at equilibrium, h * N*, in tonnes."""
        return h * self.equilibrium(h)

    # -- simulation ------------------------------------------------------------

    def simulate(
        self,
        weeks: int = WEEKS_PER_YEAR,
        h: float = 0.10,
        closed_weeks: Iterable[int] = (),
    ) -> StockPath:
        """Simulate ``weeks`` weeks of growth and harvesting.

        Parameters
        ----------
        weeks:
            Number of weekly steps (>= 1).
        h:
            Proportion of the stock harvested per week, 0 <= h <= 1.
        closed_weeks:
            Week-of-year numbers (0..51) with no harvesting, repeated every
            52 weeks (closed season).

        Raises
        ------
        ValueError
            For invalid ``weeks``, ``h`` or closed-week numbers.
        """
        if isinstance(weeks, bool) or not isinstance(weeks, (int, np.integer)) or weeks < 1:
            raise ValueError("weeks must be a positive integer.")
        if not 0.0 <= h <= 1.0:
            raise ValueError("h must be between 0 and 1.")
        closed = set(int(w) for w in closed_weeks)
        if any(not 0 <= w < WEEKS_PER_YEAR for w in closed):
            raise ValueError("closed_weeks must be week-of-year numbers 0..51.")

        stock = np.empty(weeks + 1)
        harvest = np.empty(weeks)
        stock[0] = self.n0
        for t in range(weeks):
            n = stock[t]
            rate = 0.0 if (t % WEEKS_PER_YEAR) in closed else h
            harvest[t] = rate * n
            stock[t + 1] = max(0.0, n + self.r * n * (1.0 - n / self.K) - harvest[t])
        return StockPath(stock, harvest, float(h))


# ---------------------------------------------------------------------------
# Price model
# ---------------------------------------------------------------------------

class PriceModel:
    """Bounded random walk for the weekly fish price (UGX/kg).

    ``P[0] = start`` and ``P[t+1] = P[t] + e[t]`` with ``e ~ N(0, step_sd)``.
    A step that crosses a bound is reflected back inside (``2*bound - P``), so the
    price never sticks at a bound the way clipping would make it.

    Parameters
    ----------
    start, lower, upper:
        Starting price and bounds, ``lower < start < upper``.
    step_sd:
        Standard deviation of the weekly change (>= 0).
    """

    def __init__(
        self,
        start: float = 12_000.0,
        step_sd: float = 300.0,
        lower: float = 9_000.0,
        upper: float = 16_000.0,
    ) -> None:
        if not lower < start < upper:
            raise ValueError("Bounds must satisfy lower < start < upper.")
        if step_sd < 0:
            raise ValueError("step_sd must not be negative.")
        if step_sd >= upper - lower:
            raise ValueError("step_sd must be smaller than the width of the price band.")
        self.start, self.step_sd = float(start), float(step_sd)
        self.lower, self.upper = float(lower), float(upper)

    def __repr__(self) -> str:
        return (f"PriceModel(start={self.start:,.0f}, step_sd={self.step_sd:,.0f}, "
                f"bounds=[{self.lower:,.0f}, {self.upper:,.0f}] UGX/kg)")

    def _reflect(self, p: NDArray[np.float64]) -> NDArray[np.float64]:
        """Mirror values back into [lower, upper]; repeats for very large steps."""
        while True:
            above, below = p > self.upper, p < self.lower
            if not (above.any() or below.any()):
                return p
            p = np.where(above, 2 * self.upper - p, p)
            p = np.where(below, 2 * self.lower - p, p)

    def simulate_paths(self, n_paths: int, weeks: int, rng: np.random.Generator) -> NDArray[np.float64]:
        """Simulate ``n_paths`` price paths of ``weeks`` values each (first value = start)."""
        if n_paths < 1 or weeks < 1:
            raise ValueError("n_paths and weeks must be at least 1.")
        prices = np.empty((n_paths, weeks))
        prices[:, 0] = self.start
        steps = rng.normal(0.0, self.step_sd, size=(n_paths, weeks - 1))
        for t in range(1, weeks):
            prices[:, t] = self._reflect(prices[:, t - 1] + steps[:, t - 1])
        return prices

    def simulate(self, weeks: int, rng: np.random.Generator) -> NDArray[np.float64]:
        """Simulate one price path of ``weeks`` values."""
        return self.simulate_paths(1, weeks, rng)[0]


# ---------------------------------------------------------------------------
# Revenue and risk
# ---------------------------------------------------------------------------

def weekly_revenue(harvest_tonnes: ArrayLike, price_ugx_per_kg: ArrayLike) -> NDArray[np.float64]:
    """Weekly revenue in UGX: tonnes x 1,000 kg/t x UGX/kg.

    ``price_ugx_per_kg`` may be one path (weeks,) or many paths (n_paths, weeks).

    Raises
    ------
    ValueError
        If shapes do not match or any value is negative or non-finite.
    """
    H = np.asarray(harvest_tonnes, dtype=float)
    P = np.asarray(price_ugx_per_kg, dtype=float)
    if H.ndim != 1 or P.shape[-1] != H.size:
        raise ValueError(f"harvest {H.shape} and price {P.shape} do not line up week by week.")
    for label, arr in (("harvest_tonnes", H), ("price_ugx_per_kg", P)):
        if not np.all(np.isfinite(arr)) or np.any(arr < 0):
            raise ValueError(f"{label} must be finite and non-negative.")
    return H * KG_PER_TONNE * P


def monte_carlo_annual_revenue(
    path: StockPath,
    price_model: PriceModel,
    n_paths: int = 1000,
    seed: int = 2026,
) -> NDArray[np.float64]:
    """Annual revenue (UGX) for ``n_paths`` simulated price paths.

    The stock path is deterministic, so the spread of the result is price risk
    only. Each element is the sum of the path's weekly revenues.
    """
    if n_paths < 1000:
        raise ValueError("Use at least 1,000 price paths for the Monte Carlo.")
    rng = np.random.default_rng(seed)
    prices = price_model.simulate_paths(n_paths, path.harvest_tonnes.size, rng)
    return weekly_revenue(path.harvest_tonnes, prices).sum(axis=1)


def value_at_risk(samples: ArrayLike, level: float = 0.05) -> float:
    """5 % Value-at-Risk reported as the 5th percentile of the outcomes.

    "In 1 year out of 20 we expect annual revenue at or below this value."
    """
    x = np.asarray(samples, dtype=float)
    if x.ndim != 1 or x.size == 0 or not np.all(np.isfinite(x)):
        raise ValueError("samples must be a non-empty 1-D array of finite numbers.")
    if not 0.0 < level < 1.0:
        raise ValueError("level must be between 0 and 1.")
    return float(np.percentile(x, 100 * level))


class RiskAssessor:
    """Classify revenue risk by the coefficient of variation (CV = SD / mean).

    CV is unitless and scale-free, unlike variance (UGX squared). The default
    thresholds are derived, not arbitrary: under approximate normality the
    1-in-20 bad year is about mean - 1.645 SD, a shortfall of 1.645 x CV. If a
    shortfall up to 10 % is tolerable (Low) and up to 20 % is uncomfortable but
    manageable (Moderate), the thresholds are 0.10/1.645 and 0.20/1.645.
    The 10 % / 20 % tolerances are stated assumptions.
    """

    def __init__(self, low_cv: float = 0.10 / 1.645, high_cv: float = 0.20 / 1.645) -> None:
        if not 0 < low_cv < high_cv:
            raise ValueError("Thresholds must satisfy 0 < low_cv < high_cv.")
        self.low_cv, self.high_cv = float(low_cv), float(high_cv)

    @classmethod
    def from_shortfall_tolerance(cls, low: float = 0.10, high: float = 0.20, z: float = 1.645) -> RiskAssessor:
        """Build thresholds from tolerable 1-in-20 shortfalls (fractions of the mean)."""
        if z <= 0:
            raise ValueError("z must be positive.")
        return cls(low / z, high / z)

    def __repr__(self) -> str:
        return f"RiskAssessor(Low < {self.low_cv:.2%} <= Moderate < {self.high_cv:.2%} <= High)"

    def classify(self, cv: float) -> str:
        """Return ``"Low"``, ``"Moderate"`` or ``"High"``."""
        if not np.isfinite(cv) or cv < 0:
            raise ValueError("cv must be a non-negative finite number.")
        if cv < self.low_cv:
            return "Low"
        if cv < self.high_cv:
            return "Moderate"
        return "High"

    def assess(self, annual_revenues: ArrayLike, level: float = 0.05) -> dict[str, float | str]:
        """Mean, SD, CV, risk class and VaR of simulated annual revenues.

        Raises
        ------
        ValueError
            If fewer than 2 samples or the mean is not positive (CV undefined).
        """
        x = np.asarray(annual_revenues, dtype=float)
        if x.ndim != 1 or x.size < 2:
            raise ValueError("Need at least 2 simulated revenues.")
        mean = float(x.mean())
        if mean <= 0:
            raise ValueError("CV is undefined when mean revenue is not positive.")
        sd = float(x.std(ddof=1))
        cv = sd / mean
        return {"mean": mean, "sd": sd, "cv": cv, "risk": self.classify(cv),
                "var": value_at_risk(x, level)}
