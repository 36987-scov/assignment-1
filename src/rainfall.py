"""Rainfall pattern and crop suitability analysis (Mini-Project 4).

Contents
--------
Region
    Twelve monthly rainfall totals (mm) with summary methods.
CropRule
    A monthly rainfall band for a crop, derived from seasonal crop water needs.
cosine_similarity, pearson, euclidean, similarity_matrices
    Pairwise comparison of rainfall regimes.
detect_seasons
    Rainy-season peaks with scipy.signal.find_peaks, treating the year as circular.
fetch_power_daily, daily_to_monthly, download_power_csv, load_power_csv
    Extension: real monthly rainfall from NASA POWER (PRECTOTCORR, mm/day),
    summed to mm/month. Run ``python -m src.rainfall`` once to download.
"""

from __future__ import annotations

import calendar
import csv
import json
import math
import urllib.parse
import urllib.request
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.signal import find_peaks

__all__ = [
    "MONTHS",
    "DAYS_PER_MONTH",
    "Region",
    "CropRule",
    "cosine_similarity",
    "pearson",
    "euclidean",
    "similarity_matrices",
    "detect_seasons",
    "fetch_power_daily",
    "daily_to_monthly",
    "download_power_csv",
    "load_power_csv",
    "POWER_LOCATIONS",
]

MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
DAYS_PER_MONTH = 365.25 / 12  # 30.44: average month length used to convert seasonal needs


def _monthly_array(values: ArrayLike, label: str = "rainfall") -> NDArray[np.float64]:
    """Validate 12 finite, non-negative monthly values."""
    try:
        arr = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric.") from exc
    if arr.shape != (12,):
        raise ValueError(f"{label} must have exactly 12 monthly values, got shape {arr.shape}.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{label} must not contain NaN or infinite values.")
    if np.any(arr < 0):
        raise ValueError(f"{label} must not be negative.")
    return arr


# ---------------------------------------------------------------------------
# Region
# ---------------------------------------------------------------------------

class Region:
    """Monthly rainfall (mm, Jan-Dec) for one region.

    Raises
    ------
    ValueError
        If the name is empty or the rainfall is not 12 finite, non-negative numbers.
    """

    def __init__(self, name: str, rainfall_mm: ArrayLike, source: str) -> None:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("name must be a non-empty string.")
        self.name: str = name.strip()
        self.rainfall: NDArray[np.float64] = _monthly_array(rainfall_mm)
        self.source: str = source

    def __repr__(self) -> str:
        return f"Region(name={self.name!r}, annual={self.annual_total():.0f} mm, source={self.source!r})"

    def __len__(self) -> int:
        return 12

    def annual_total(self) -> float:
        """Total annual rainfall in mm."""
        return float(self.rainfall.sum())

    def mean(self) -> float:
        """Mean monthly rainfall in mm."""
        return float(self.rainfall.mean())

    def wettest_month(self) -> tuple[str, float]:
        """(month, mm) of the wettest month (first one if tied)."""
        i = int(np.argmax(self.rainfall))
        return MONTHS[i], float(self.rainfall[i])

    def driest_month(self) -> tuple[str, float]:
        """(month, mm) of the driest month (first one if tied)."""
        i = int(np.argmin(self.rainfall))
        return MONTHS[i], float(self.rainfall[i])

    def cv(self) -> float:
        """Coefficient of variation across months: sample SD / mean (unitless).

        Raises
        ------
        ValueError
            If the mean is zero (no rain at all), where CV is undefined.
        """
        m = self.rainfall.mean()
        if m == 0:
            raise ValueError("CV is undefined for a region with no rainfall.")
        return float(self.rainfall.std(ddof=1) / m)


# ---------------------------------------------------------------------------
# Crop rules
# ---------------------------------------------------------------------------

DROUGHT = "Drought risk"
EXCESS = "Excess rain (waterlogging risk)"


class CropRule:
    """A suitable monthly rainfall band [low_mm, high_mm] for one crop.

    Below the band is "Drought risk"; above it is "Excess rain (waterlogging
    risk)". Rain above a crop's water need does not by itself cause
    waterlogging (that depends on soil and drainage), so the upper label is a
    warning rather than a certainty.
    """

    def __init__(self, crop: str, low_mm: float, high_mm: float, source: str) -> None:
        if not isinstance(crop, str) or not crop.strip():
            raise ValueError("crop must be a non-empty string.")
        if not (math.isfinite(low_mm) and math.isfinite(high_mm)) or not 0 <= low_mm < high_mm:
            raise ValueError("Need finite bounds with 0 <= low_mm < high_mm.")
        self.crop, self.low_mm, self.high_mm, self.source = crop.strip(), float(low_mm), float(high_mm), source

    @classmethod
    def from_seasonal(
        cls,
        crop: str,
        need_low_mm: float,
        need_high_mm: float,
        season_days_short: float,
        season_days_long: float,
        source: str,
    ) -> CropRule:
        """Derive a monthly band from a seasonal water need and season length.

        Stated assumption: the need is spread evenly over the season, so
        ``low = need_low / (long season in months)`` (the least water per month)
        and ``high = need_high / (short season in months)`` (the most).

        Raises
        ------
        ValueError
            If the needs or season lengths are not positive and ordered.
        """
        if not 0 < need_low_mm <= need_high_mm:
            raise ValueError("Need 0 < need_low_mm <= need_high_mm.")
        if not 0 < season_days_short <= season_days_long:
            raise ValueError("Need 0 < season_days_short <= season_days_long.")
        low = need_low_mm / (season_days_long / DAYS_PER_MONTH)
        high = need_high_mm / (season_days_short / DAYS_PER_MONTH)
        return cls(crop, low, high, source)

    def __repr__(self) -> str:
        return f"CropRule({self.crop!r}, good {self.low_mm:.1f}-{self.high_mm:.1f} mm/month)"

    def code(self, mm: float) -> int:
        """-1 drought risk, 0 good, +1 excess rain."""
        if not math.isfinite(mm) or mm < 0:
            raise ValueError("Rainfall must be a finite, non-negative number.")
        if mm < self.low_mm:
            return -1
        if mm > self.high_mm:
            return 1
        return 0

    def classify(self, mm: float) -> str:
        """Human-readable label for one month's rainfall."""
        return {-1: DROUGHT, 0: f"Good for {self.crop}", 1: EXCESS}[self.code(mm)]

    def classify_region(self, region: Region) -> list[str]:
        """Labels for all 12 months of a region."""
        return [self.classify(float(v)) for v in region.rainfall]

    def codes_for_region(self, region: Region) -> NDArray[np.int64]:
        """Codes (-1, 0, +1) for all 12 months, for heatmaps."""
        return np.array([self.code(float(v)) for v in region.rainfall])


# ---------------------------------------------------------------------------
# Similarity and distance
# ---------------------------------------------------------------------------

def _pair(a: ArrayLike, b: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    x, y = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if x.ndim != 1 or x.shape != y.shape or x.size == 0:
        raise ValueError("Vectors must be non-empty, 1-D and the same length.")
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
        raise ValueError("Vectors must be finite.")
    return x, y


def cosine_similarity(a: ArrayLike, b: ArrayLike) -> float:
    """Cosine of the angle between two vectors: a.b / (|a| |b|).

    Measures shape only: ``cosine_similarity(a, 2 * a) == 1``.

    Raises
    ------
    ValueError
        If either vector is all zeros (the angle is undefined).
    """
    x, y = _pair(a, b)
    nx, ny = np.linalg.norm(x), np.linalg.norm(y)
    if nx == 0 or ny == 0:
        raise ValueError("Cosine similarity is undefined for a zero vector.")
    return float(np.dot(x, y) / (nx * ny))


def pearson(a: ArrayLike, b: ArrayLike) -> float:
    """Pearson correlation: cosine similarity of the mean-centred vectors.

    Raises
    ------
    ValueError
        If either vector is constant (zero variance).
    """
    x, y = _pair(a, b)
    xc, yc = x - x.mean(), y - y.mean()
    if not np.any(xc) or not np.any(yc):
        raise ValueError("Pearson correlation is undefined for a constant vector.")
    return cosine_similarity(xc, yc)


def euclidean(a: ArrayLike, b: ArrayLike) -> float:
    """Euclidean distance in the data's units (mm)."""
    x, y = _pair(a, b)
    return float(np.linalg.norm(x - y))


def similarity_matrices(regions: Sequence[Region]) -> dict[str, NDArray[np.float64] | list[str]]:
    """Cosine, Pearson and Euclidean matrices for every pair of regions."""
    if len(regions) < 2:
        raise ValueError("Need at least 2 regions to compare.")
    n = len(regions)
    out: dict[str, NDArray[np.float64] | list[str]] = {"names": [r.name for r in regions]}
    for key, fn in (("cosine", cosine_similarity), ("pearson", pearson), ("euclidean", euclidean)):
        m = np.empty((n, n))
        for i in range(n):
            for j in range(n):
                m[i, j] = fn(regions[i].rainfall, regions[j].rainfall)
        out[key] = m
    return out


# ---------------------------------------------------------------------------
# Season detection
# ---------------------------------------------------------------------------

def detect_seasons(region: Region, prominence_mm: float = 50.0, circular: bool = True) -> dict[str, object]:
    """Find rainy-season peaks and classify the regime.

    With ``circular=True`` the year is wrapped (Oct-Dec placed before January
    and Jan-Mar after December), so a peak in December or January can be found;
    ``find_peaks`` never reports the first or last element of an array.

    Returns
    -------
    dict
        ``months`` (peak month names), ``indices`` (0-11), ``prominences`` (mm)
        and ``regime`` ("unimodal", "bimodal", "multimodal" or "no clear peak").
    """
    if prominence_mm < 0:
        raise ValueError("prominence_mm must not be negative.")
    x = region.rainfall
    pad = 3 if circular else 0
    series = np.concatenate([x[-pad:], x, x[:pad]]) if pad else x
    peaks, props = find_peaks(series, prominence=prominence_mm)
    keep = (peaks >= pad) & (peaks < pad + 12)
    idx = (peaks[keep] - pad) % 12
    prom = props["prominences"][keep]
    regime = {0: "no clear peak", 1: "unimodal", 2: "bimodal"}.get(len(idx), "multimodal")
    return {"months": [MONTHS[i] for i in idx], "indices": idx, "prominences": prom, "regime": regime}


# ---------------------------------------------------------------------------
# Extension: NASA POWER real data
# ---------------------------------------------------------------------------

POWER_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
POWER_LOCATIONS: dict[str, tuple[float, float]] = {   # (latitude, longitude), approximate town centres
    "Kampala": (0.35, 32.58),
    "Gulu": (2.77, 32.30),
    "Mbarara": (-0.61, 30.65),
}


def fetch_power_daily(lat: float, lon: float, start_year: int, end_year: int, timeout: float = 60.0) -> dict[str, float]:
    """Download daily corrected precipitation (PRECTOTCORR, mm/day) from NASA POWER.

    Returns ``{"YYYYMMDD": mm_per_day}``. Needs an internet connection.
    """
    query = urllib.parse.urlencode({
        "parameters": "PRECTOTCORR",
        "community": "AG",
        "latitude": lat,
        "longitude": lon,
        "start": f"{start_year}0101",
        "end": f"{end_year}1231",
        "format": "JSON",
    })
    with urllib.request.urlopen(f"{POWER_URL}?{query}", timeout=timeout) as response:
        payload = json.load(response)
    try:
        return {k: float(v) for k, v in payload["properties"]["parameter"]["PRECTOTCORR"].items()}
    except KeyError as exc:
        raise ValueError(f"Unexpected NASA POWER response: {str(payload)[:300]}") from exc


def daily_to_monthly(daily: Mapping[str, float], fill_value: float = -999.0) -> dict[tuple[int, int], float]:
    """Sum daily mm/day values into monthly totals (mm/month).

    Each daily value is the depth of rain on that day, so a month's total is
    the sum of its days.

    Raises
    ------
    ValueError
        If any value equals the fill value (missing data) or is negative, or a
        month has fewer days than the calendar requires.
    """
    totals: dict[tuple[int, int], float] = defaultdict(float)
    counts: dict[tuple[int, int], int] = defaultdict(int)
    for key, value in daily.items():
        if value == fill_value:
            raise ValueError(f"Missing value (fill {fill_value}) on {key}.")
        if value < 0:
            raise ValueError(f"Negative rainfall on {key}.")
        ym = (int(key[:4]), int(key[4:6]))
        totals[ym] += value
        counts[ym] += 1
    for (y, m), n in counts.items():
        expected = calendar.monthrange(y, m)[1]  # days in that calendar month (handles leap years)
        if n != expected:
            raise ValueError(f"{y}-{m:02d} has {n} daily values, expected {expected}.")
    return dict(sorted(totals.items()))


def download_power_csv(
    path: str | Path,
    locations: Mapping[str, tuple[float, float]] = POWER_LOCATIONS,
    start_year: int = 2015,
    end_year: int = 2024,
) -> Path:
    """Download each location, aggregate to months and write ``region,year,month,rain_mm``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, (lat, lon) in locations.items():
        print(f"Downloading {name} ({lat}, {lon}) {start_year}-{end_year} ...")
        monthly = daily_to_monthly(fetch_power_daily(lat, lon, start_year, end_year))
        rows += [(name, y, m, round(v, 2)) for (y, m), v in monthly.items()]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(("region", "year", "month", "rain_mm"))
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {path}")
    return path


def load_power_csv(path: str | Path) -> dict[str, tuple[NDArray[np.int64], NDArray[np.float64]]]:
    """Read the CSV into ``{region: (years, rain[years, 12])}``.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If columns are missing or a region lacks a complete set of 12 months for a year.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python -m src.rainfall` to download it.")
    data: dict[str, dict[int, dict[int, float]]] = defaultdict(lambda: defaultdict(dict))
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        missing = {"region", "year", "month", "rain_mm"} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV is missing column(s): {sorted(missing)}")
        for row in reader:
            data[row["region"]][int(row["year"])][int(row["month"])] = float(row["rain_mm"])
    out = {}
    for region, years in data.items():
        ys = sorted(years)
        for y in ys:
            if sorted(years[y]) != list(range(1, 13)):
                raise ValueError(f"{region} {y} does not have all 12 months.")
        out[region] = (np.array(ys), np.array([[years[y][m] for m in range(1, 13)] for y in ys]))
    if not out:
        raise ValueError("CSV contains no data rows.")
    return out


if __name__ == "__main__":
    target = Path(__file__).resolve().parents[1] / "data" / "nasa_power_monthly_rainfall.csv"
    download_power_csv(target)
