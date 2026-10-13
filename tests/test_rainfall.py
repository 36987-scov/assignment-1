"""Tests for src/rainfall.py (Mini-Project 4). All tests run offline."""

import numpy as np
import pytest
from scipy.spatial.distance import cosine as scipy_cosine_distance

from src.rainfall import (
    CropRule,
    Region,
    cosine_similarity,
    daily_to_monthly,
    detect_seasons,
    euclidean,
    load_power_csv,
    pearson,
    similarity_matrices,
)

KAMPALA = [120, 140, 180, 200, 220, 180, 90, 70, 60, 100, 110, 130]
MBARARA = [70, 85, 120, 140, 90, 25, 20, 55, 100, 125, 120, 90]


@pytest.fixture
def kampala() -> Region:
    return Region("Kampala", KAMPALA, "Illustrative")


# ---------------------------------------------------------------- Region

def test_region_summary_hand_values(kampala):
    assert kampala.annual_total() == 1600
    assert kampala.mean() == pytest.approx(1600 / 12)
    assert kampala.wettest_month() == ("May", 220)
    assert kampala.driest_month() == ("Sep", 60)
    assert len(kampala) == 12


def test_region_cv(kampala):
    assert kampala.cv() == pytest.approx(np.std(KAMPALA, ddof=1) / np.mean(KAMPALA))
    with pytest.raises(ValueError, match="undefined"):
        Region("Desert", [0] * 12, "test").cv()


@pytest.mark.parametrize("bad", [[100] * 11, [100] * 13, [100] * 11 + [-1], [100] * 11 + [float("nan")], []])
def test_region_rejects_bad_rainfall(bad):
    with pytest.raises(ValueError):
        Region("X", bad, "test")


# ---------------------------------------------------------------- CropRule

def test_from_seasonal_derivation():
    maize = CropRule.from_seasonal("maize", 500, 800, 125, 180, "FAO")
    assert maize.low_mm == pytest.approx(500 / (180 / 30.4375))      # ~84.5 mm
    assert maize.high_mm == pytest.approx(800 / (125 / 30.4375))     # ~194.8 mm


def test_classify_boundaries():
    rule = CropRule("maize", 80, 200, "test")
    assert rule.classify(79.9) == "Drought risk"
    assert rule.classify(80) == "Good for maize"                     # bounds are inclusive
    assert rule.classify(200) == "Good for maize"
    assert rule.classify(200.1) == "Excess rain (waterlogging risk)"
    assert rule.classify(0) == "Drought risk"                        # zero rain edge case


def test_classify_region_and_codes(kampala):
    rule = CropRule("beans", 83, 160, "test")
    labels = rule.classify_region(kampala)
    assert len(labels) == 12 and labels[0] == "Good for beans"     # Jan 120 mm
    assert labels[4] == "Excess rain (waterlogging risk)"          # May 220 mm
    assert labels[8] == "Drought risk"                             # Sep 60 mm
    assert list(rule.codes_for_region(kampala))[:5] == [0, 0, 1, 1, 1]


@pytest.mark.parametrize("low, high", [(100, 100), (150, 100), (-1, 50), (0, float("inf"))])
def test_crop_rule_rejects_bad_band(low, high):
    with pytest.raises(ValueError):
        CropRule("maize", low, high, "test")


def test_from_seasonal_rejects_unordered_inputs():
    with pytest.raises(ValueError):
        CropRule.from_seasonal("maize", 800, 500, 125, 180, "test")
    with pytest.raises(ValueError):
        CropRule.from_seasonal("maize", 500, 800, 180, 125, "test")


# ---------------------------------------------------------------- similarity

def test_cosine_matches_scipy():
    expected = 1 - scipy_cosine_distance(KAMPALA, MBARARA)           # scipy returns a DISTANCE
    assert cosine_similarity(KAMPALA, MBARARA) == pytest.approx(expected)


def test_cosine_is_scale_invariant_but_euclidean_is_not():
    a = np.array(KAMPALA, dtype=float)
    assert cosine_similarity(a, 2 * a) == pytest.approx(1.0)
    assert euclidean(a, 2 * a) == pytest.approx(np.linalg.norm(a))


def test_cosine_zero_vector_raises():
    with pytest.raises(ValueError, match="zero vector"):
        cosine_similarity([0] * 12, KAMPALA)


def test_pearson_matches_numpy():
    assert pearson(KAMPALA, MBARARA) == pytest.approx(np.corrcoef(KAMPALA, MBARARA)[0, 1])
    with pytest.raises(ValueError):
        pearson([5] * 12, KAMPALA)


def test_similarity_matrices_properties(kampala):
    regions = [kampala, Region("Mbarara", MBARARA, "Illustrative")]
    m = similarity_matrices(regions)
    for key in ("cosine", "pearson", "euclidean"):
        np.testing.assert_allclose(m[key], m[key].T)                # symmetric
    np.testing.assert_allclose(np.diag(m["cosine"]), 1.0)
    np.testing.assert_allclose(np.diag(m["euclidean"]), 0.0)
    with pytest.raises(ValueError):
        similarity_matrices([kampala])


# ---------------------------------------------------------------- season detection

def test_detect_one_and_two_peaks():
    one = Region("One", [10, 20, 60, 120, 200, 260, 200, 120, 60, 20, 10, 5], "synthetic")
    two = Region("Two", [20, 60, 150, 200, 120, 40, 30, 80, 160, 190, 90, 30], "synthetic")
    assert detect_seasons(one)["regime"] == "unimodal"
    result = detect_seasons(two)
    assert result["regime"] == "bimodal"
    assert result["months"] == ["Apr", "Oct"]


def test_circular_wrapping_finds_december_peak():
    # rainfall peaks in December; a non-circular search cannot see the last element
    dec_peak = Region("DecPeak", [150, 100, 60, 40, 30, 20, 20, 30, 40, 60, 120, 200], "synthetic")
    assert detect_seasons(dec_peak, circular=True)["months"] == ["Dec"]
    assert detect_seasons(dec_peak, circular=False)["regime"] == "no clear peak"


def test_threshold_controls_regime():
    gulu = Region("Gulu", [8, 25, 75, 160, 190, 145, 170, 215, 175, 150, 60, 15], "Illustrative")
    assert detect_seasons(gulu, prominence_mm=10)["regime"] == "bimodal"     # May peak has prominence 45
    assert detect_seasons(gulu, prominence_mm=50)["regime"] == "unimodal"
    with pytest.raises(ValueError):
        detect_seasons(gulu, prominence_mm=-1)


# ---------------------------------------------------------------- NASA POWER helpers (offline)

def _fake_month(year, month, days, mm_per_day):
    return {f"{year}{month:02d}{d:02d}": mm_per_day for d in range(1, days + 1)}


def test_daily_to_monthly_sums_days():
    daily = {**_fake_month(2023, 2, 28, 2.0), **_fake_month(2024, 2, 29, 1.0)}   # leap year
    monthly = daily_to_monthly(daily)
    assert monthly[(2023, 2)] == pytest.approx(56.0)               # 28 days x 2 mm/day
    assert monthly[(2024, 2)] == pytest.approx(29.0)


def test_daily_to_monthly_rejects_fill_and_gaps():
    with pytest.raises(ValueError, match="Missing"):
        daily_to_monthly({**_fake_month(2023, 1, 30, 1.0), "20230131": -999.0})
    with pytest.raises(ValueError, match="expected 31"):
        daily_to_monthly(_fake_month(2023, 1, 30, 1.0))            # January needs 31 days


def test_load_power_csv(tmp_path):
    path = tmp_path / "power.csv"
    lines = ["region,year,month,rain_mm"] + [f"Gulu,{y},{m},{m * 10}" for y in (2015, 2016) for m in range(1, 13)]
    path.write_text("\n".join(lines) + "\n")
    years, rain = load_power_csv(path)["Gulu"]
    assert list(years) == [2015, 2016]
    assert rain.shape == (2, 12) and rain[0, 11] == 120
    incomplete = tmp_path / "bad.csv"
    incomplete.write_text("region,year,month,rain_mm\nGulu,2015,1,10\n")
    with pytest.raises(ValueError, match="12 months"):
        load_power_csv(incomplete)
    with pytest.raises(FileNotFoundError):
        load_power_csv(tmp_path / "missing.csv")
