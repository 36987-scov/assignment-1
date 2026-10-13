"""Tests for src/taxi_routes.py (Mini-Project 5). Expected values are hand calculations."""

import numpy as np
import pytest

from src.taxi_routes import (
    LinearTrendForecaster,
    MovingAverageForecaster,
    Route,
    SeasonalNaiveForecaster,
    SESForecaster,
    backtest,
    equilibrium,
    excess_demand,
    simulate_weekly,
    tune_alpha,
    vehicles_needed,
)

NTINDA = [35, 40, 42, 50, 55, 60, 48, 52, 47, 45]


# ---------------------------------------------------------------- Route

def test_route_revenue_hand_values():
    r = Route("Kampala-Ntinda", NTINDA, 2000, "PDF")
    assert r.daily_revenue()[0] == 70_000                    # 35 x 2,000
    assert r.total_revenue() == 948_000                      # 474 passengers x 2,000
    assert len(r) == 10


def test_route_stats_match_numpy():
    s = Route("Ntinda", NTINDA, 2000, "PDF").stats()
    assert s["mean"] == pytest.approx(47.4)
    assert s["variance"] == pytest.approx(np.var(NTINDA, ddof=1))
    assert s["stdev"] == pytest.approx(np.std(NTINDA, ddof=1))


@pytest.mark.parametrize("pax, fare", [([], 2000), ([10, -1], 2000), ([10, 12], 0), ([10, np.nan], 2000)])
def test_route_rejects_bad_input(pax, fare):
    with pytest.raises(ValueError):
        Route("X", pax, fare, "test")


# ---------------------------------------------------------------- equilibrium

def test_equilibrium_hand_values():
    p, q = equilibrium()
    assert p == pytest.approx(2200.0)                        # 110 / 0.05
    assert q == pytest.approx(76.0)                          # 120 - 0.02 * 2200
    assert excess_demand(2000) == pytest.approx(10.0)        # 80 demanded - 70 supplied
    assert excess_demand(p) == pytest.approx(0.0)


def test_equilibrium_parallel_lines_rejected():
    with pytest.raises(ValueError, match="same slope"):
        equilibrium(demand=(120, 0.03), supply=(10, 0.03))


# ---------------------------------------------------------------- forecasters

def test_moving_average_hand_value():
    assert MovingAverageForecaster(3).fit([35, 40, 42, 50]).predict() == pytest.approx(44.0)  # (40+42+50)/3


def test_ses_hand_value_and_alpha_one_is_naive():
    # level: 10 -> 0.5*20 + 0.5*10 = 15 -> 0.5*30 + 0.5*15 = 22.5
    assert SESForecaster(0.5).fit([10, 20, 30]).predict() == pytest.approx(22.5)
    assert SESForecaster(1.0).fit(NTINDA).predict() == NTINDA[-1]


def test_linear_trend_exact_line():
    assert LinearTrendForecaster().fit([10, 12, 14, 16]).predict() == pytest.approx(18.0)


def test_seasonal_naive_uses_same_weekday_last_week():
    history = list(range(1, 15))                             # 14 days: 1..14
    assert SeasonalNaiveForecaster(7).fit(history).predict() == 8   # day 15 <- day 8


def test_forecaster_errors():
    with pytest.raises(ValueError, match="at least 3"):
        MovingAverageForecaster(3).fit([1, 2])
    with pytest.raises(RuntimeError):
        SESForecaster(0.3).predict()
    for bad in (0, 1.5, -0.1):
        with pytest.raises(ValueError):
            SESForecaster(bad)
    with pytest.raises(ValueError):
        MovingAverageForecaster(0)


# ---------------------------------------------------------------- backtest and tuning

def test_backtest_days_4_to_10():
    result = backtest(NTINDA, lambda: MovingAverageForecaster(3), start_day=4)
    assert list(result["days"]) == [4, 5, 6, 7, 8, 9, 10]
    assert result["forecasts"][0] == pytest.approx((35 + 40 + 42) / 3)       # day 4 from days 1-3
    assert result["actuals"][0] == 50


def test_backtest_has_no_look_ahead():
    changed = NTINDA.copy()
    changed[-1] = 999                                        # alter only day 10
    a = backtest(NTINDA, LinearTrendForecaster, start_day=4)
    b = backtest(changed, LinearTrendForecaster, start_day=4)
    np.testing.assert_allclose(a["forecasts"], b["forecasts"])   # forecasts never see the target day


def test_backtest_invalid_range():
    with pytest.raises(ValueError):
        backtest(NTINDA, lambda: MovingAverageForecaster(3), start_day=1)
    with pytest.raises(ValueError):
        backtest(NTINDA, lambda: MovingAverageForecaster(3), start_day=4, end_day=11)


def test_tune_alpha_grid_and_ties():
    best, maes = tune_alpha(NTINDA)
    assert len(maes) == 100 and min(maes) == 0.01 and max(maes) == 1.0
    assert maes[best] == min(maes.values())
    flat_best, _ = tune_alpha([5] * 10, grid=[0.3, 0.1, 0.9])  # every alpha gives MAE 0
    assert flat_best == 0.1                                  # tie -> smallest alpha


# ---------------------------------------------------------------- fleet

@pytest.mark.parametrize("pax, expected", [(45, 1), (97, 1), (98, 2), (0, 1), (224, 3)])
def test_vehicles_needed(pax, expected):
    # capacity 8 x 14 = 112 per vehicle; 97 x 1.15 = 111.55 -> 1, 98 x 1.15 = 112.7 -> 2
    assert vehicles_needed(pax) == expected


def test_vehicles_needed_rejects_bad_input():
    with pytest.raises(ValueError):
        vehicles_needed(-1)
    with pytest.raises(ValueError):
        vehicles_needed(50, seats=0)


# ---------------------------------------------------------------- simulation

def test_simulate_weekly_reproducible_and_patterned():
    a, b = simulate_weekly(50, seed=1), simulate_weekly(50, seed=1)
    np.testing.assert_array_equal(a, b)
    assert a.size == 60 and np.all(a >= 0) and np.all(a == np.round(a))
    fridays, sundays = a[4::7], a[6::7]                      # day 1 = Monday
    assert fridays.mean() > sundays.mean()
    flat = simulate_weekly(100, days=7, noise_sd=0)
    np.testing.assert_array_equal(flat, [100, 100, 100, 100, 130, 100, 70])
