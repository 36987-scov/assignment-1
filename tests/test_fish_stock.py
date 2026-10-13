"""Tests for src/fish_stock.py (Mini-Project 3). Expected values are hand calculations."""

import numpy as np
import pytest

from src.fish_stock import (
    FishStock,
    PriceModel,
    RiskAssessor,
    fibonacci,
    monte_carlo_annual_revenue,
    value_at_risk,
    weekly_revenue,
)


# ---------------------------------------------------------------- Fibonacci baseline

def test_fibonacci_15():
    f = fibonacci(15)
    assert list(f[:7]) == [1, 1, 2, 3, 5, 8, 13]
    assert f[-1] == 610


# ---------------------------------------------------------------- FishStock

def test_one_step_hand_value():
    # N1 = 4000 + 0.4*4000*(1 - 0.4) - 0.1*4000 = 4000 + 960 - 400 = 4560
    path = FishStock().simulate(weeks=1, h=0.10)
    assert path.stock_tonnes[1] == pytest.approx(4560.0)
    assert path.harvest_tonnes[0] == pytest.approx(400.0)


def test_shapes_52_weeks():
    path = FishStock().simulate(52, h=0.10)
    assert path.stock_tonnes.shape == (53,)
    assert path.harvest_tonnes.shape == (52,)


def test_fixed_points():
    assert FishStock(n0=10_000).simulate(20, h=0.0).final_stock == pytest.approx(10_000)  # N = K
    assert FishStock(n0=0).simulate(20, h=0.1).final_stock == 0.0                         # extinct stays 0


@pytest.mark.parametrize("h, expected", [(0.05, 8750), (0.10, 7500), (0.20, 5000), (0.30, 2500)])
def test_converges_to_equilibrium(h, expected):
    stock = FishStock()
    assert stock.equilibrium(h) == pytest.approx(expected)          # K(1 - h/r)
    assert stock.simulate(400, h).final_stock == pytest.approx(expected, rel=1e-6)


def test_msy_theory():
    stock = FishStock()
    assert stock.msy == pytest.approx(1000.0)                       # rK/4 = 0.4*10000/4
    assert stock.msy_harvest_rate == pytest.approx(0.2)
    assert stock.sustainable_yield(0.2) == pytest.approx(stock.msy)
    assert stock.sustainable_yield(0.1) < stock.msy and stock.sustainable_yield(0.3) < stock.msy
    assert stock.equilibrium(0.5) == 0.0                            # h >= r: no positive equilibrium


def test_stock_never_negative_under_full_harvest():
    path = FishStock().simulate(10, h=1.0)
    assert np.all(path.stock_tonnes >= 0)


def test_closed_season_has_no_harvest():
    closed = range(10, 18)                                          # 8 weeks
    path = FishStock().simulate(104, h=0.1, closed_weeks=closed)
    for year in (0, 1):
        assert np.all(path.harvest_tonnes[year * 52 + 10: year * 52 + 18] == 0)
    assert path.harvest_tonnes[9] > 0


@pytest.mark.parametrize("kwargs", [{"r": 0}, {"K": -1}, {"n0": -5}, {"r": float("nan")}])
def test_invalid_stock_parameters(kwargs):
    with pytest.raises(ValueError):
        FishStock(**kwargs)


@pytest.mark.parametrize("kwargs", [{"weeks": 0}, {"h": -0.1}, {"h": 1.5}, {"closed_weeks": [52]}])
def test_invalid_simulation_arguments(kwargs):
    with pytest.raises(ValueError):
        FishStock().simulate(**kwargs)


# ---------------------------------------------------------------- PriceModel

def test_price_paths_reproducible_and_bounded():
    model = PriceModel()
    a = model.simulate_paths(500, 52, np.random.default_rng(1))
    b = model.simulate_paths(500, 52, np.random.default_rng(1))
    np.testing.assert_array_equal(a, b)
    assert np.all(a[:, 0] == 12_000)
    assert a.min() >= 9_000 and a.max() <= 16_000


def test_price_reflection_with_large_steps():
    model = PriceModel(step_sd=5_000)                               # forces many reflections
    p = model.simulate_paths(200, 52, np.random.default_rng(3))
    assert p.min() >= 9_000 and p.max() <= 16_000


def test_zero_volatility_price_is_constant():
    p = PriceModel(step_sd=0).simulate(10, np.random.default_rng(0))
    np.testing.assert_array_equal(p, np.full(10, 12_000.0))


@pytest.mark.parametrize("kwargs", [{"start": 20_000}, {"lower": 13_000}, {"step_sd": -1}])
def test_invalid_price_model(kwargs):
    with pytest.raises(ValueError):
        PriceModel(**kwargs)


# ---------------------------------------------------------------- revenue

def test_weekly_revenue_units():
    # 10 t * 1000 kg/t * 12,000 UGX/kg = 120,000,000 UGX
    np.testing.assert_allclose(weekly_revenue([10.0], [12_000.0]), [120_000_000.0])


def test_weekly_revenue_rejects_bad_input():
    with pytest.raises(ValueError):
        weekly_revenue([1, 2], [12_000])                            # length mismatch
    with pytest.raises(ValueError):
        weekly_revenue([-1.0], [12_000.0])                          # negative harvest


def test_monte_carlo_constant_price_matches_hand_value():
    path = FishStock().simulate(52, h=0.1)
    revenues = monte_carlo_annual_revenue(path, PriceModel(step_sd=0), n_paths=1000)
    expected = path.total_harvest * 1000 * 12_000
    np.testing.assert_allclose(revenues, expected)


def test_monte_carlo_requires_1000_paths():
    with pytest.raises(ValueError, match="1,000"):
        monte_carlo_annual_revenue(FishStock().simulate(52), PriceModel(), n_paths=999)


# ---------------------------------------------------------------- VaR and risk

def test_value_at_risk_known_sample():
    assert value_at_risk(np.arange(1, 101)) == pytest.approx(5.95)  # 5th percentile of 1..100
    with pytest.raises(ValueError):
        value_at_risk([])


def test_risk_thresholds_derived_from_tolerance():
    ra = RiskAssessor.from_shortfall_tolerance(0.10, 0.20, 1.645)
    assert ra.low_cv == pytest.approx(0.0608, abs=1e-4)
    assert ra.high_cv == pytest.approx(0.1216, abs=1e-4)


@pytest.mark.parametrize("cv, label", [(0.0, "Low"), (0.06, "Low"), (0.061, "Moderate"),
                                       (0.12, "Moderate"), (0.1215, "Moderate"), (0.1216, "High")])
def test_risk_classification_boundaries(cv, label):
    assert RiskAssessor().classify(cv) == label


def test_assess_and_invalid_cases():
    out = RiskAssessor().assess([90.0, 100.0, 110.0])
    assert out["mean"] == pytest.approx(100.0)
    assert out["cv"] == pytest.approx(0.10)                         # SD (n-1) = 10
    assert out["risk"] == "Moderate"
    with pytest.raises(ValueError):
        RiskAssessor().assess([0.0, 0.0])                           # mean 0: CV undefined
    with pytest.raises(ValueError):
        RiskAssessor(0.2, 0.1)                                      # thresholds out of order
