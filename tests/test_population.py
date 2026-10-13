"""Tests for src/population.py (Mini-Project 1).

Run from the project root with:  pytest
Expected values are hand calculations, not outputs copied from the code.
"""

from pathlib import Path
import statistics

import numpy as np
import pytest

from src.population import (
    CAGRForecaster,
    DistrictPopulation,
    FibonacciRatioForecaster,
    Forecaster,
    LinearTrendForecaster,
    additional_classrooms,
    bootstrap_intervals,
    evaluate,
    load_ubos_district,
    mae,
    mape,
    rmse,
    select_best,
)

UBOS_FILE = Path(__file__).resolve().parents[1] / "data" / (
    "Revised_Subcounty_population_2015_2030_146_Districts.xlsx"
)
YEARS = list(range(2015, 2025))
KAMPALA = [1200, 1250, 1300, 1350, 1420, 1500, 1580, 1650, 1720, 1800]


@pytest.fixture
def kampala() -> DistrictPopulation:
    return DistrictPopulation("Kampala", YEARS, KAMPALA, "Illustrative (assignment PDF)")


# ---------------------------------------------------------------- DistrictPopulation

def test_len_and_repr(kampala):
    assert len(kampala) == 10
    text = repr(kampala)
    assert "Kampala" in text and "2015" in text and "2024" in text


@pytest.mark.parametrize(
    "years, pops, message",
    [
        ([], [], "empty"),
        ([2015, 2016], [100], "equal length"),
        ([2015, 2016], [100, -5], "negative"),
        ([2016, 2015], [100, 110], "increasing"),
        ([2015, 2016], [100, float("nan")], "NaN"),
        ([2015.5, 2016], [100, 110], "whole numbers"),
    ],
)
def test_invalid_inputs_rejected(years, pops, message):
    with pytest.raises(ValueError, match=message):
        DistrictPopulation("X", years, pops, "test")


def test_zero_population_is_valid_but_growth_undefined():
    d = DistrictPopulation("Empty", [2015, 2016], [0, 10], "test")
    with pytest.raises(ValueError, match="zero"):
        d.growth_rates()
    with pytest.raises(ValueError, match="zero"):
        d.cagr()


def test_summary_statistics_match_numpy_with_ddof(kampala):
    s = kampala.summary()
    # statistics.variance is the sample variance -> NumPy ddof=1
    assert s["statistics"]["sample_variance"] == pytest.approx(s["numpy"]["sample_variance"])
    assert s["statistics"]["population_variance"] == pytest.approx(s["numpy"]["population_variance"])
    # population / sample = (n - 1) / n = 0.9 for n = 10
    ratio = s["numpy"]["population_variance"] / s["numpy"]["sample_variance"]
    assert ratio == pytest.approx(0.9)
    assert s["statistics"]["mean"] == pytest.approx(1477.0)  # 14770 / 10
    assert s["statistics"]["median"] == pytest.approx(1460.0)  # (1420 + 1500) / 2


def test_growth_rates_and_cagr_hand_values():
    d = DistrictPopulation("T", [2020, 2021, 2022], [100, 110, 121], "test")
    np.testing.assert_allclose(d.growth_rates(), [0.10, 0.10])
    assert d.cagr() == pytest.approx(0.10)  # (121/100)^(1/2) - 1, two intervals


def test_cagr_uses_n_minus_one_intervals(kampala):
    # 1800 / 1200 = 1.5 over 9 intervals, not 10
    assert kampala.cagr() == pytest.approx(1.5 ** (1 / 9) - 1)


def test_split_train_test(kampala):
    train, test = kampala.split(2021)
    assert list(train.years) == list(range(2015, 2022))
    assert list(test.years) == [2022, 2023, 2024]
    with pytest.raises(ValueError):
        kampala.split(2030)  # empty test set


# ---------------------------------------------------------------- Forecasters

def test_forecaster_is_abstract():
    with pytest.raises(TypeError):
        Forecaster()  # type: ignore[abstract]


def test_linear_recovers_exact_line():
    years = [2015, 2016, 2017, 2018]
    values = [10, 13, 16, 19]  # slope 3
    model = LinearTrendForecaster().fit(years, values)
    assert model.slope_ == pytest.approx(3.0)
    np.testing.assert_allclose(model.predict(2), [22, 25])
    np.testing.assert_allclose(model.fitted_values(), values)


def test_cagr_forecaster_compounds():
    model = CAGRForecaster().fit([2020, 2021, 2022], [100, 110, 121])
    assert model.growth_rate_ == pytest.approx(0.10)
    np.testing.assert_allclose(model.predict(2), [133.1, 146.41])


def test_fibonacci_ratios_and_forecast():
    model = FibonacciRatioForecaster()
    np.testing.assert_allclose(model.ratios(4), [1.0, 2.0, 1.5, 5 / 3])
    model.fit([2020, 2021], [50, 100])
    # 100 * 1, 100 * 1 * 2, 100 * 1 * 2 * 1.5
    np.testing.assert_allclose(model.predict(3), [100, 200, 300])
    assert model.fitted_values() is None


def test_predict_before_fit_and_bad_horizon():
    model = LinearTrendForecaster()
    with pytest.raises(RuntimeError):
        model.predict(3)
    model.fit([2015, 2016], [1, 2])
    for bad in (0, -1, 2.5):
        with pytest.raises(ValueError):
            model.predict(bad)  # type: ignore[arg-type]


def test_fit_rejects_non_consecutive_years():
    with pytest.raises(ValueError, match="consecutive"):
        LinearTrendForecaster().fit([2015, 2017], [1, 2])


# ---------------------------------------------------------------- Metrics and selection

def test_metrics_hand_values():
    actual, pred = [100, 200], [110, 180]  # errors -10, +20
    assert mae(actual, pred) == pytest.approx(15.0)
    assert rmse(actual, pred) == pytest.approx(np.sqrt((100 + 400) / 2))
    assert mape(actual, pred) == pytest.approx(10.0)  # (10% + 10%) / 2


def test_mape_rejects_zero_and_shape_mismatch():
    with pytest.raises(ValueError, match="zero"):
        mape([0, 10], [1, 10])
    with pytest.raises(ValueError, match="shape"):
        mae([1, 2], [1, 2, 3])


def test_evaluate_and_select_best(kampala):
    rows = evaluate(kampala, [LinearTrendForecaster(), CAGRForecaster(), FibonacciRatioForecaster()])
    assert [r["model"] for r in rows] == ["LinearTrend", "CAGR", "FibonacciRatio"]
    assert all(len(r["predictions"]) == 3 for r in rows)
    assert select_best(rows) != "FibonacciRatio"


def test_select_best_tie_break_on_mape():
    rows = [
        {"model": "A", "RMSE": 5.0, "MAPE": 2.0},
        {"model": "B", "RMSE": 5.0, "MAPE": 1.0},
    ]
    assert select_best(rows) == "B"


# ---------------------------------------------------------------- Planning

def test_additional_classrooms_hand_value():
    # 0.18 * 100 thousand * 1000 / 53 = 339.6 -> 340
    assert additional_classrooms(1000, 1100) == 340


def test_additional_classrooms_edge_cases():
    assert additional_classrooms(500, 500) == 0  # no growth
    assert additional_classrooms(500, 450) == 0  # decline -> 0, never negative
    # exactly 53 pupils must be 1 classroom, not 2 (floating-point guard)
    assert additional_classrooms(0, 53 / 0.18 / 1000) == 1
    with pytest.raises(ValueError):
        additional_classrooms(-1, 10)
    with pytest.raises(ValueError):
        additional_classrooms(1, 10, school_age_share=0)


# ---------------------------------------------------------------- Bootstrap (extension)

def test_bootstrap_is_reproducible_and_ordered(kampala):
    a = bootstrap_intervals(LinearTrendForecaster, kampala.years, kampala.populations, 5, n_boot=200, seed=1)
    b = bootstrap_intervals(LinearTrendForecaster, kampala.years, kampala.populations, 5, n_boot=200, seed=1)
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x, y)
    point, lower, upper = a
    assert np.all(lower <= upper)
    assert np.all((lower <= point) & (point <= upper))


def test_bootstrap_rejects_fibonacci(kampala):
    with pytest.raises(ValueError, match="no in-sample fit"):
        bootstrap_intervals(FibonacciRatioForecaster, kampala.years, kampala.populations, 3)


# ---------------------------------------------------------------- UBOS loader

def test_load_ubos_district_values():
    d = load_ubos_district(UBOS_FILE, " kasese ")  # case and spaces ignored
    assert d.name == "Kasese"
    assert len(d) == 10
    assert d.populations[0] == pytest.approx(707.6)  # 707,600 persons -> thousands
    assert d.populations[-1] == pytest.approx(860.3)


def test_load_ubos_district_missing():
    with pytest.raises(ValueError, match="not found"):
        load_ubos_district(UBOS_FILE, "Atlantis")
    with pytest.raises(FileNotFoundError):
        load_ubos_district("no_such_file.xlsx", "Kasese")
