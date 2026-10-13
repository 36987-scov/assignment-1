# OOP with Python — Assignment 1, Advent 2026

> **Student:** OJORU LINDYWAY SCOVIA.  s26M19/043
> **Programme:** MSDS · **Course:** OOP with Python

## Mini-projects

| # | Project | Notebook | Module | Tests |
|---|---|---|---|---|
| 1 | UBOS District Population Forecaster | `notebooks/project1_population.ipynb` | `src/population.py` | 28 |
| 2 | Solar Micro Grid Dispatch Planner | `notebooks/project2_microgrid.ipynb` | `src/microgrid.py` | 35 |
| 3 | Lake Victoria Fish Stock & Export Risk Model | `notebooks/project3_fish_stock.ipynb` | `src/fish_stock.py` | 38 |
| 4 | Rainfall Pattern & Crop Suitability Analyser | `notebooks/project4_rainfall.ipynb` | `src/rainfall.py` | 26 |
| 5 | Taxi Route Revenue, Pricing & Fleet Planner | `notebooks/project5_taxi_routes.ipynb` | `src/taxi_routes.py` | 24 |

All reusable logic is in `src/`; the notebooks import it and contain presentation and interpretation only. Each notebook ends with a Findings & Limitations section.

## Requirements

* **Python 3.14** (developed and tested with 3.14.8 on Windows). Python 3.15 was avoided because, at the time of writing, it was very new and SciPy wheels for it could not be relied on.
* Packages, pinned in `requirements.txt`: `numpy`, `scipy`, `matplotlib`, `pytest`, `notebook` and `openpyxl` (used only to read the UBOS Excel workbook in Project 1). `statistics`, `timeit`, `csv`, `json` and `urllib` are from the standard library.
* No black-box forecasting libraries are used: all forecasting, smoothing and risk methods are implemented in `src/`.

## Setup

Windows (Command Prompt), from the folder where you want the project:

```cmd
git clone https://github.com/36987-scov/assignment-1.git
cd assignment-1
py -3.14 -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

macOS / Linux:

```bash
git clone https://github.com/36987-scov/assignment-1.git
cd assignment-1
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Running

**Tests** (from the project root, with the virtual environment active):

```cmd
pytest
```

Expected result: `151 passed`. `pytest.ini` puts the project root on the import path and limits collection to `tests/`.

**Notebooks** — interactively:

```cmd
jupyter notebook
```

then open a notebook from `notebooks/` and choose *Kernel → Restart & Run All*. Each notebook's first cell adds the project root to `sys.path` so that `src` can be imported.

Or headlessly, from a fresh kernel (this is how each committed notebook was executed):

```cmd
jupyter nbconvert --to notebook --execute --inplace notebooks\project1_population.ipynb
```

All notebooks run offline. Project 2 regenerates `data/microgrid_demand.csv` from a fixed seed when run. The NASA POWER data for Project 4 is already committed; to re-download it (internet required; overwrites the file):

```cmd
python -m src.rainfall
```

## Project structure

```
assignment-1/
├── data/
│   ├── Revised_Subcounty_population_2015_2030_146_Districts.xlsx   # UBOS source workbook (P1)
│   ├── microgrid_demand.csv                                         # seeded illustrative demand (P2)
│   └── nasa_power_monthly_rainfall.csv                              # NASA POWER 2015–2024 (P4 extension)
├── notebooks/        # one notebook per mini-project
├── src/              # reusable modules (one per mini-project)
├── tests/            # pytest suites (one per module)
├── pytest.ini
├── requirements.txt
└── README.md
```

## Data and sources

| Data | Used in | Nature |
|---|---|---|
| Kampala, Wakiso, Gulu populations 2015–2024 | P1 | Illustrative (assignment PDF) |
| Arua City and Kasese populations 2015–2024 | P1 | UBOS (2022) *Revised Subcounty Population Projections 2015 to 2030 for 146 Districts* (district total rows, persons converted to thousands). These are **projections**, not census counts. |
| 30 days of micro-grid demand | P2 | Synthetic, generated with a fixed seed (weekly pattern + noise; parameters stated in the notebook) |
| Fish stock, price and harvest parameters | P3 | Assignment parameters; price-walk step and bounds handling are stated assumptions |
| Kampala, Gulu, Mbarara monthly rainfall | P4 | Illustrative (assignment PDF) |
| Monthly rainfall 2015–2024 | P4 extension | NASA POWER, daily `PRECTOTCORR` (MERRA-2 corrected precipitation, mm/day), AG community, summed to months. Gridded model/satellite estimates, not gauge records. |
| Ntinda passenger counts | P5 | Original question; Entebbe and Mukono are illustrative |
| 60 days of passenger counts | P5 extension | Synthetic, fixed seeds |

**References**

* Uganda Bureau of Statistics (2022). *Revised Subcounty Population Projections 2015 to 2030 for 146 Districts* [dataset]. Last updated 2 June 2022. https://www.ubos.org/explore-statistics/20
* FAO (1991). *Water harvesting: A manual for the design and construction of water harvesting schemes for plant production*, Chapter 2, Tables 2, 6 and 7 (crop water needs and growing periods for maize, beans and sorghum). https://www.fao.org/4/u3160e/u3160e04.htm
* FAO Land & Water. *Crop information: Maize*. https://www.fao.org/land-water/databases-and-software/crop-information/maize/
* Kisembe, J. et al. (2018). Evaluation of rainfall simulations over Uganda in CORDEX regional climate models. *Theoretical and Applied Climatology*. https://doi.org/10.1007/s00704-018-2643-x
* FAO AQUASTAT. *Country profile – Uganda*. https://openknowledge.fao.org
* Phillips, J. and McIntyre, B. (2000). ENSO and interannual rainfall variability in Uganda: implications for agricultural management. https://www.giss.nasa.gov/pubs/abs/ph09000j.html
* NASA Langley Research Center, POWER Project. https://power.larc.nasa.gov (data accessed October 2026)

## Key findings

**P1 — Population.** Wakiso grows fastest in relative terms (CAGR ≈ 6.5 %); Kampala and Gulu have identical CAGRs (≈ 4.6 %) because both grow by a factor of 1.5. On the 2022–2024 hold-out, constant-percentage (CAGR) growth forecast the illustrative districts best, while a linear trend forecast the two UBOS districts best. Estimated additional classrooms by 2029: Wakiso 2,088, Kampala 1,545, Gulu 412, Kasese 292, Arua City 156 (lower bounds; current capacity assumed adequate). The Fibonacci-ratio model missed by 78–92 % (MAPE) and implies roughly 62 % growth per period.

**P2 — Micro-grid.** The system is well-posed (det = −5, condition number = 3 + 2√2 ≈ 5.83). Days are feasible only when 0.5 ≤ D₂/D₁ ≤ 1.333; none of the 30 seeded days was infeasible, and infeasible days are handled with NNLS plus a report of unmet demand. Battery use is the more volatile source (CV ≈ 0.37 vs 0.15 for solar); the 30-day cost is UGX 304,125, about 80 % of it from batteries. The vectorised solve matched the loop exactly and was many times faster (machine-dependent).

**P3 — Fish stock.** h = 0.20 is the MSY harvest rate (stock held at K/2 = 5,000 t, yield 1,000 t/week = rK/4). h = 0.30 halves the stock and earns less than h = 0.20. A raw variance threshold of 50,000 is meaningless because revenue variance is in UGX² (≈ 10¹⁸ here); the coefficient of variation is used instead. Price risk is Moderate at every harvest rate, so a CV-only rule cannot detect over-harvesting. An 8-week closed season raised 5-year revenue only when the stock was over-harvested (+7 % at h = 0.30; −9 % to −15 % otherwise).

**P4 — Rainfall.** Cosine similarity rates all regions as fairly similar (0.75–0.89), but Pearson correlation shows their wet and dry months barely align (−0.17 to 0.21). With a 50 mm prominence rule, peak detection matched the known zones for Gulu (unimodal) and Mbarara (bimodal) but misclassified the illustrative Kampala series; on ten years of NASA POWER data the same rule matched all three. The advisory note (Gulu, 193 words) recommends planting maize with the established April rains.

**P5 — Taxi routes.** Kampala–Entebbe earns 57 % of revenue. The Ntinda fare of UGX 2,000 is below the equilibrium of UGX 2,200, leaving excess demand of 10 passengers per trip-hour. Simple exponential smoothing with α ≈ 1 (effectively "tomorrow = today") had the lowest walk-forward MAE on every route, an optimistic result because α was tuned on the scoring days. Day-11 forecasts need one vehicle per route even with a 15 % buffer, which suggests the counts cover only part of each route's demand. In the 60-day extension, seasonal naive cut MA(3)'s error by about 29–56 %.

## Limitations (summary)

Several datasets are illustrative or synthetic, so those results describe the methods rather than real places. Series are short (10 years, 10 days, 12 months), so model rankings rest on very few errors. Monthly crop thresholds assume water use is spread evenly over a season, and the 50 mm peak threshold was fixed after inspecting the data. Project-specific limitations are discussed at the end of each notebook.

## Testing and reproducibility

* **151 pytest tests** across five files, covering hand-calculated values, edge cases (empty, zero, negative, NaN, singular or ill-conditioned systems, boundary values) and invalid inputs. Network-dependent code in Project 4 is tested offline with synthetic data.
* All randomness uses `numpy.random.default_rng(seed)` with fixed seeds (2026 unless stated); no global random state is used.
* Every committed notebook was executed top to bottom from a fresh kernel with `jupyter nbconvert --execute`.

## AI use declaration

I used AI to assist with explanations and debugging of code. The resulting implementation was tested, and the I reviewed the code and its outputs.
