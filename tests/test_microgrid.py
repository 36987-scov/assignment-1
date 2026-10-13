"""Tests for src/microgrid.py (Mini-Project 2).

Expected values come from the closed-form solution of the 2 x 2 system:
    x = (2*D2 - D1) / 5,   y = (4*D1 - 3*D2) / 5
"""

import numpy as np
import pytest

from src.microgrid import (
    HybridMicroGrid,
    MicroGrid,
    daily_cost,
    dispatch_plan,
    flag_infeasible,
    generate_demand_csv,
    load_demand_csv,
    parse_demand,
    read_demand,
    solve_nonnegative,
)


def closed_form(d1: float, d2: float) -> tuple[float, float]:
    return (2 * d2 - d1) / 5, (4 * d1 - 3 * d2) / 5


@pytest.fixture
def grid() -> MicroGrid:
    return MicroGrid()


# ---------------------------------------------------------------- well-posedness

def test_determinant_and_condition_number(grid):
    assert grid.determinant == pytest.approx(-5.0)       # 3*1 - 2*4
    assert grid.condition_number == pytest.approx(3 + 2 * np.sqrt(2))  # ~5.83


def test_singular_matrix_rejected():
    with pytest.raises(ValueError, match="singular"):
        MicroGrid([[1, 2], [2, 4]])                       # row 2 = 2 * row 1


@pytest.mark.parametrize("bad", [[[1, 2, 3], [4, 5, 6]], [[1, np.nan], [0, 1]], []])
def test_invalid_matrix_rejected(bad):
    with pytest.raises(ValueError):
        MicroGrid(bad)


# ---------------------------------------------------------------- solving

def test_solve_day_matches_closed_form(grid):
    x, y = grid.solve_day(80, 70)
    ex, ey = closed_form(80, 70)                          # (12, 22)
    assert (x, y) == pytest.approx((ex, ey))
    assert (x, y) == pytest.approx((12.0, 22.0))


def test_zero_demand_gives_zero_allocation(grid):
    np.testing.assert_allclose(grid.solve_day(0, 0), [0, 0])


@pytest.mark.parametrize("d1, d2", [(-1, 10), (np.inf, 10), (np.nan, 10)])
def test_solve_day_rejects_bad_demand(grid, d1, d2):
    with pytest.raises(ValueError):
        grid.solve_day(d1, d2)


def test_solve_day_wrong_number_of_demands(grid):
    with pytest.raises(ValueError, match="shape"):
        grid.solve_day(80)


def test_loop_and_vectorised_agree(grid):
    rng = np.random.default_rng(0)
    demands = rng.uniform(50, 100, size=(30, 2))
    loop, vec = grid.solve_loop(demands), grid.solve_vectorised(demands)
    assert vec.shape == (30, 2)
    np.testing.assert_allclose(loop, vec)
    np.testing.assert_allclose(vec[:, 0], (2 * demands[:, 1] - demands[:, 0]) / 5)


def test_empty_demands_rejected(grid):
    with pytest.raises(ValueError, match="empty"):
        grid.solve_vectorised(np.empty((0, 2)))


# ---------------------------------------------------------------- feasibility

def test_flag_infeasible_with_tolerance():
    sols = np.array([[1.0, 2.0], [-1e-15, 3.0], [-0.5, 4.0]])
    np.testing.assert_array_equal(flag_infeasible(sols), [False, False, True])


def test_nnls_on_infeasible_day(grid):
    # D2/D1 = 1.75 > 4/3, so the exact battery share is negative
    exact = grid.solve_day(40, 70)
    assert exact[1] < 0                                   # y = (160 - 210)/5 = -10
    allocation, unmet = solve_nonnegative(grid, [40, 70])
    assert np.all(allocation >= 0)
    # residual identity: demand = A @ allocation + unmet
    np.testing.assert_allclose(grid.A @ allocation + unmet, [40, 70])


def test_dispatch_plan_keeps_feasible_days_exact(grid):
    plan = dispatch_plan(grid, [[80, 70], [40, 70]])
    np.testing.assert_array_equal(plan["infeasible"], [False, True])
    np.testing.assert_allclose(plan["allocation"][0], [12, 22])
    np.testing.assert_allclose(plan["unmet"][0], [0, 0])
    assert np.all(plan["allocation"] >= 0)


# ---------------------------------------------------------------- input

@pytest.mark.parametrize("text, expected", [("80", 80.0), (" 12.5 ", 12.5), ("0", 0.0)])
def test_parse_demand_valid(text, expected):
    assert parse_demand(text) == expected


@pytest.mark.parametrize("text, reason", [("", "Empty"), ("   ", "Empty"), ("abc", "not a number"),
                                          ("-5", "negative"), ("inf", "finite"), ("nan", "finite")])
def test_parse_demand_invalid(text, reason):
    with pytest.raises(ValueError, match=reason):
        parse_demand(text)


def test_read_demand_reprompts_until_valid():
    answers = iter(["abc", "", "-5", "80"])
    messages = []
    value = read_demand("D1: ", input_fn=lambda _: next(answers), print_fn=messages.append)
    assert value == 80.0
    assert len(messages) == 3                             # one message per rejected entry


def test_read_demand_gives_up_after_max_attempts():
    with pytest.raises(ValueError, match="2 attempts"):
        read_demand("D1: ", input_fn=lambda _: "x", print_fn=lambda _: None, max_attempts=2)


# ---------------------------------------------------------------- CSV

def test_generate_and_load_csv_round_trip(tmp_path):
    path = tmp_path / "demand.csv"
    written = generate_demand_csv(path, days=30, seed=7)
    loaded = load_demand_csv(path)
    assert loaded.shape == (30, 2)
    np.testing.assert_allclose(loaded, written)


def test_generate_csv_is_reproducible(tmp_path):
    a = generate_demand_csv(tmp_path / "a.csv", seed=1)
    b = generate_demand_csv(tmp_path / "b.csv", seed=1)
    c = generate_demand_csv(tmp_path / "c.csv", seed=2)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_generate_csv_has_weekly_pattern(tmp_path):
    d = generate_demand_csv(tmp_path / "d.csv", days=28, seed=3, noise_sd=0.0)
    weekday_d1, weekend_d1 = d[0, 0], d[5, 0]             # Monday vs Saturday
    assert weekday_d1 == pytest.approx(80.0)
    assert weekend_d1 == pytest.approx(64.0)              # 80 * 0.80


def test_load_csv_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_demand_csv(tmp_path / "missing.csv")
    bad_cols = tmp_path / "cols.csv"
    bad_cols.write_text("day,d1_kwh\n1,80\n")
    with pytest.raises(ValueError, match="missing"):
        load_demand_csv(bad_cols)
    bad_value = tmp_path / "value.csv"
    bad_value.write_text("day,weekday,d1_kwh,d2_kwh\n1,Mon,80,-3\n")
    with pytest.raises(ValueError, match="Line 2"):
        load_demand_csv(bad_value)
    empty = tmp_path / "empty.csv"
    empty.write_text("day,weekday,d1_kwh,d2_kwh\n")
    with pytest.raises(ValueError, match="no data"):
        load_demand_csv(empty)


# ---------------------------------------------------------------- cost

def test_daily_cost_hand_value():
    # 12 kWh solar * 150 + 22 kWh battery * 450 = 1,800 + 9,900 = 11,700 UGX
    np.testing.assert_allclose(daily_cost([[12, 22]]), [11_700])


def test_daily_cost_rejects_negative_allocation():
    with pytest.raises(ValueError, match="non-negative"):
        daily_cost([[12, -1]])


# ---------------------------------------------------------------- extension

def test_hybrid_solves_3x3():
    h = HybridMicroGrid()
    assert h.determinant == pytest.approx(-4.0)
    # chosen so the answer is x=10, y=20, z=5
    np.testing.assert_allclose(h.solve_day(75, 70, 35), [10, 20, 5])
    assert isinstance(h, MicroGrid)


def test_hybrid_dependent_third_equation_rejected():
    dependent = [[3, 2, 1], [4, 1, 2], [7, 3, 3]]          # row 3 = row 1 + row 2
    with pytest.raises(ValueError, match="singular"):
        HybridMicroGrid(dependent)
