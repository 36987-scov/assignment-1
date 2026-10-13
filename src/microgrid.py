"""Solar micro-grid dispatch planning (Mini-Project 2).

Each day the energy drawn from solar (x) and batteries (y), in kWh, must satisfy

    3x + 2y = D1   (daytime load, kWh)
    4x +  y = D2   (critical-equipment load, kWh)

Contents
--------
MicroGrid
    Generic n x n dispatch model (default: the 2 x 2 system above) with
    determinant, condition number, single-day, loop and vectorised solvers.
HybridMicroGrid
    Extension: adds a diesel generator (z) and a third constraint (3 x 3).
flag_infeasible, solve_nonnegative, dispatch_plan
    Detect physically impossible (negative) allocations and replace them with
    the closest non-negative allocation (scipy.optimize.nnls), reporting the
    unmet demand.
parse_demand, read_demand
    Validated interactive input; the input function is injectable so tests and
    notebooks can run without a person typing.
generate_demand_csv, load_demand_csv
    Seeded illustrative 30-day demand data with a weekly pattern.
daily_cost
    Energy cost in UGX.
"""

from __future__ import annotations

import csv
import math
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import linalg
from scipy.optimize import nnls

__all__ = [
    "MicroGrid",
    "HybridMicroGrid",
    "flag_infeasible",
    "solve_nonnegative",
    "dispatch_plan",
    "parse_demand",
    "read_demand",
    "generate_demand_csv",
    "load_demand_csv",
    "daily_cost",
    "WEEKDAYS",
]

WEEKDAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")


# ---------------------------------------------------------------------------
# Dispatch models
# ---------------------------------------------------------------------------

class MicroGrid:
    """Linear dispatch model ``A @ allocation = demand``.

    Parameters
    ----------
    coefficients:
        Square coefficient matrix ``A``. Defaults to ``[[3, 2], [4, 1]]``.
    source_names:
        One name per energy source (column of ``A``).

    Raises
    ------
    ValueError
        If ``A`` is not square, not finite, the names do not match its size,
        or ``A`` is singular (its rows are linearly dependent).
    """

    DEFAULT_COEFFICIENTS: tuple[tuple[float, ...], ...] = ((3, 2), (4, 1))
    DEFAULT_SOURCES: tuple[str, ...] = ("solar", "battery")

    def __init__(
        self,
        coefficients: ArrayLike | None = None,
        source_names: Sequence[str] | None = None,
    ) -> None:
        A = np.asarray(
            self.DEFAULT_COEFFICIENTS if coefficients is None else coefficients, dtype=float
        )
        if A.ndim != 2 or A.shape[0] != A.shape[1] or A.size == 0:
            raise ValueError(f"coefficients must be a non-empty square matrix, got shape {A.shape}.")
        if not np.all(np.isfinite(A)):
            raise ValueError("coefficients must be finite numbers.")
        names = tuple(self.DEFAULT_SOURCES if source_names is None else source_names)
        if len(names) != A.shape[0]:
            raise ValueError(f"Expected {A.shape[0]} source names, got {len(names)}.")
        rank = int(np.linalg.matrix_rank(A))
        if rank < A.shape[0]:
            raise ValueError(
                f"The system is singular (rank {rank} < {A.shape[0]}): at least one equation "
                "is a linear combination of the others, so there is either no solution or "
                "infinitely many. Replace the dependent equation with independent information."
            )
        self.A: NDArray[np.float64] = A
        self.source_names: tuple[str, ...] = names
        self.determinant: float = float(np.linalg.det(A))
        self.condition_number: float = float(np.linalg.cond(A))

    @property
    def n_sources(self) -> int:
        """Number of energy sources (unknowns)."""
        return int(self.A.shape[0])

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(sources={self.source_names}, "
            f"det={self.determinant:.4g}, cond={self.condition_number:.4g})"
        )

    # -- validation ----------------------------------------------------------

    def _validate_demands(self, demands: ArrayLike) -> NDArray[np.float64]:
        """Return demands as a (days, n_sources) array of finite, non-negative floats."""
        D = np.asarray(demands, dtype=float)
        if D.ndim == 1:
            D = D.reshape(1, -1)
        if D.ndim != 2 or D.shape[1] != self.n_sources:
            raise ValueError(
                f"demands must have shape (days, {self.n_sources}), got {np.shape(demands)}."
            )
        if D.shape[0] == 0:
            raise ValueError("demands must not be empty.")
        if not np.all(np.isfinite(D)):
            raise ValueError("demands must be finite numbers.")
        if np.any(D < 0):
            raise ValueError("demands must not be negative.")
        return D

    # -- solvers ---------------------------------------------------------------

    def solve_day(self, *demands: float) -> NDArray[np.float64]:
        """Exact allocation for one day, e.g. ``solve_day(d1, d2)``.

        The result may contain negative values: this method reports the exact
        mathematical solution. Use ``flag_infeasible`` / ``dispatch_plan`` to
        detect and handle physically impossible days.
        """
        D = self._validate_demands(demands)
        return linalg.solve(self.A, D[0])

    def solve_loop(self, demands: ArrayLike) -> NDArray[np.float64]:
        """Solve each day separately in a Python loop. Returns (days, n_sources)."""
        D = self._validate_demands(demands)
        return np.array([linalg.solve(self.A, row) for row in D])

    def solve_vectorised(self, demands: ArrayLike) -> NDArray[np.float64]:
        """Solve all days in one call with an (n_sources x days) right-hand side."""
        D = self._validate_demands(demands)
        return linalg.solve(self.A, D.T).T


class HybridMicroGrid(MicroGrid):
    """Extension: solar (x), battery (y) and diesel (z) with three constraints.

    Default system (coefficients on z and the third equation are our stated
    assumptions; the assignment leaves the third constraint to us)::

        3x + 2y + 1z = D1   daytime load
        4x + 1y + 2z = D2   critical-equipment load
         x +  y +  z = D3   total metered energy

    Determinant = -4, so the default system has exactly one solution.
    Everything else (validation, solvers, singularity check) is inherited.
    """

    DEFAULT_COEFFICIENTS = ((3, 2, 1), (4, 1, 2), (1, 1, 1))
    DEFAULT_SOURCES = ("solar", "battery", "diesel")


# ---------------------------------------------------------------------------
# Feasibility handling
# ---------------------------------------------------------------------------

def flag_infeasible(solutions: ArrayLike, tol: float = 1e-9) -> NDArray[np.bool_]:
    """True for each day (row) with any allocation below ``-tol``.

    ``tol`` prevents floating-point round-off such as -1e-15 being flagged.
    """
    S = np.asarray(solutions, dtype=float)
    if S.ndim == 1:
        S = S.reshape(1, -1)
    return np.any(S < -tol, axis=1)


def solve_nonnegative(
    grid: MicroGrid, demand: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Closest non-negative allocation for one day (``scipy.optimize.nnls``).

    NNLS minimises ``||A @ allocation - demand||`` subject to allocation >= 0.
    When the exact solution is negative, no non-negative allocation satisfies
    every equation, so some demand is left unmet.

    Returns
    -------
    (allocation, unmet)
        ``unmet = demand - A @ allocation`` per constraint, in kWh: positive
        means under-supplied, negative means over-supplied.
    """
    D = grid._validate_demands(demand)[0]
    allocation, _ = nnls(grid.A, D)
    return allocation, D - grid.A @ allocation


def dispatch_plan(grid: MicroGrid, demands: ArrayLike, tol: float = 1e-9) -> dict[str, NDArray]:
    """Exact solutions plus a physically feasible plan for every day.

    Feasible days keep their exact solution. Infeasible days are replaced by
    the NNLS allocation and their unmet demand is recorded.

    Returns
    -------
    dict with keys
        ``exact`` (days, n), ``infeasible`` (days,) bool, ``allocation``
        (days, n) non-negative, ``unmet`` (days, n) kWh (zero on feasible days).
    """
    D = grid._validate_demands(demands)
    exact = grid.solve_vectorised(D)
    infeasible = flag_infeasible(exact, tol)
    allocation = exact.copy()
    unmet = np.zeros_like(D)
    for i in np.flatnonzero(infeasible):
        allocation[i], unmet[i] = solve_nonnegative(grid, D[i])
    return {"exact": exact, "infeasible": infeasible, "allocation": allocation, "unmet": unmet}


# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------

def parse_demand(text: str) -> float:
    """Convert user text to a demand in kWh.

    Raises
    ------
    ValueError
        With a human-readable reason if the text is empty, not a number,
        not finite, or negative.
    """
    stripped = (text or "").strip()
    if not stripped:
        raise ValueError("Empty input: please type a number of kWh.")
    try:
        value = float(stripped)
    except ValueError:
        raise ValueError(f"'{stripped}' is not a number.") from None
    if not math.isfinite(value):
        raise ValueError("Demand must be a finite number.")
    if value < 0:
        raise ValueError("Demand cannot be negative.")
    return value


def read_demand(
    prompt: str,
    input_fn: Callable[[str], str] = input,
    print_fn: Callable[[str], None] = print,
    max_attempts: int | None = None,
) -> float:
    """Ask for a demand until a valid value is entered.

    ``input_fn`` and ``print_fn`` default to the built-ins but can be replaced
    (tests, scripted notebook demos). ``max_attempts`` limits retries.

    Raises
    ------
    ValueError
        If ``max_attempts`` invalid entries are made.
    """
    attempts = 0
    while True:
        attempts += 1
        try:
            return parse_demand(input_fn(prompt))
        except ValueError as err:
            print_fn(f"  Invalid input - {err} Try again.")
            if max_attempts is not None and attempts >= max_attempts:
                raise ValueError(f"No valid input after {max_attempts} attempts.") from err


# ---------------------------------------------------------------------------
# CSV data
# ---------------------------------------------------------------------------

CSV_COLUMNS = ("day", "weekday", "d1_kwh", "d2_kwh")


def generate_demand_csv(
    path: str | Path,
    days: int = 30,
    seed: int = 2026,
    d1_base: float = 80.0,
    d2_base: float = 70.0,
    d1_weekend_factor: float = 0.80,
    d2_weekend_factor: float = 0.95,
    noise_sd: float = 0.06,
) -> NDArray[np.float64]:
    """Write illustrative daily demand with a weekly pattern and seeded noise.

    Day 1 is a Monday. Weekend days are scaled by the weekend factors, and
    each value gets independent multiplicative noise ``1 + N(0, noise_sd)``.
    Values are rounded to 0.1 kWh. Returns the (days, 2) demand array written.
    """
    if days < 1:
        raise ValueError("days must be at least 1.")
    rng = np.random.default_rng(seed)
    rows, demands = [], []
    for day in range(days):
        weekday = WEEKDAYS[day % 7]
        weekend = weekday in ("Sat", "Sun")
        d1 = d1_base * (d1_weekend_factor if weekend else 1.0) * (1 + rng.normal(0, noise_sd))
        d2 = d2_base * (d2_weekend_factor if weekend else 1.0) * (1 + rng.normal(0, noise_sd))
        d1, d2 = round(max(d1, 0.0), 1), round(max(d2, 0.0), 1)
        rows.append((day + 1, weekday, d1, d2))
        demands.append((d1, d2))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(CSV_COLUMNS)
        writer.writerows(rows)
    return np.array(demands)


def load_demand_csv(path: str | Path) -> NDArray[np.float64]:
    """Read ``d1_kwh`` and ``d2_kwh`` columns into a (days, 2) array.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If columns are missing, the file has no data rows, or a value is
        not a non-negative number.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Demand file not found: {path}")
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        missing = {"d1_kwh", "d2_kwh"} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV is missing column(s): {sorted(missing)}")
        demands = []
        for line, row in enumerate(reader, start=2):
            try:
                demands.append((parse_demand(row["d1_kwh"]), parse_demand(row["d2_kwh"])))
            except ValueError as err:
                raise ValueError(f"Line {line}: {err}") from None
    if not demands:
        raise ValueError("CSV contains no data rows.")
    return np.array(demands)


# ---------------------------------------------------------------------------
# Cost
# ---------------------------------------------------------------------------

def daily_cost(
    allocations: ArrayLike,
    prices: Sequence[float] = (150.0, 450.0),
) -> NDArray[np.float64]:
    """Daily energy cost in UGX: ``allocations @ prices``.

    Parameters
    ----------
    allocations:
        (days, n_sources) kWh, which must be non-negative (use the feasible plan).
    prices:
        UGX per kWh for each source. Default solar 150, battery 450 (illustrative).

    Raises
    ------
    ValueError
        If shapes do not match or allocations/prices are negative.
    """
    X = np.asarray(allocations, dtype=float)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    p = np.asarray(prices, dtype=float)
    if X.shape[1] != p.size:
        raise ValueError(f"{X.shape[1]} sources but {p.size} prices.")
    if np.any(p < 0):
        raise ValueError("prices must not be negative.")
    if np.any(X < -1e-9):
        raise ValueError("allocations must be non-negative; cost the feasible plan, not raw solutions.")
    return np.clip(X, 0, None) @ p
