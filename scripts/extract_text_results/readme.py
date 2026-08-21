"""Write outputs/text_results/README.md."""
import common as c

TEXT = """# Bow Valley — textual results

Every numeric result from the nine notebooks in `notebooks/bow_valley/`, written
out as text. No plots. Regenerate with:

    /home/py/mambaforge/envs/research-gwmo/bin/python scripts/extract_text_results/run_all.py

The analysis code is copied from the notebooks unchanged, so the numbers here are
the numbers the figures show. `VERIFY.txt` records the assertions against the
text outputs the notebooks themselves store.

## What is covered

**9 GOWN wells**, daily gap-filled levels in mamsl, 2005-01-01 to 2024-12-31:

| area | wells |
|---|---|
| Marmot Creek | 0931, 0301, 0303, 0386, 0305 |
| Canmore | 0764, 0760, 0759, 0364 |

**10 driver series**, all daily:

| key | series | units | record |
|---|---|---|---|
| `krns` | Kananaskis River near Seebe (05BF001) | m³/s | 2005-01-01 – 2015-12-23 |
| `bl` | Kananaskis River below Barrier Dam (05BF025) | m³/s | 1998 – 2025 |
| `lkl` | Lower Kananaskis Lake (05BF009) | mamsl | 1932 – 2017 |
| `jcnc` | Jumpingpound Creek near Cochrane (05BH013) | mamsl | 1976 – 2024, Apr–Nov only |
| `brac_e` | Bow River at Canmore, water-level elevation | m | 2002 – 2023 |
| `brab` | Bow River at Banff (05BB001), elevation | m | 2012 – 2026 |
| `brac` | Bow River at Canmore, flow | m³/s | 1985 – 2023 |
| `banff` | Precipitation, ECCC Banff CS (3050519) | mm | 2005 – 2024 |
| `exshaw` | Precipitation, ACIS Bow Valley (Exshaw) | mm | 2005 – 2024 |
| `kv` | Precipitation, ACIS Kananaskis Boundary Auto | mm | 2009 – 2024, Apr–Nov only |

All 9 × 10 = 90 pairs are computed. **Read the `pair_class` column first:**

- `primary` — the pairing the notebooks actually plot: Canmore wells against the
  Bow River and Banff/Exshaw precipitation, Marmot Creek wells against the
  Kananaskis series and Kananaskis Boundary precipitation. 45 pairs.
- `cross_basin` — computed for completeness only. A Marmot Creek well correlated
  against the Bow River at Banff will still show a strong annual signal, because
  both follow the same snowmelt calendar, not because they are connected. 45 pairs.

## Where to start

`summaries/` holds one self-contained plain-text digest per entity — nine
`GOWN_*.txt` and ten `driver_*.txt`. Each gathers site metadata, the lithology
log, record statistics, trend, changepoints, seasonal cycle, periodicity, and
every pairwise relationship, in reading order. Start there; the CSVs below are
the same numbers in machine-readable form.

## Files

### `00_inventory/`
| file | contents |
|---|---|
| `series_inventory.csv` | all 19 series: source, span, gaps, distribution statistics |
| `well_metadata.csv` | collar and completion records for the 9 wells |
| `overlap_matrix.csv` | overlapping days and window for every pair |

### `01_cross_correlation/` — from `01_cross_correlation.ipynb`
Pearson r at daily lags from −720 to +720 d. **A positive lag means the driver
leads the well.** Each lag needs 365 overlapping days (`min_periods=365`), and
local extrema are picked with `find_peaks(prominence=0.05)`, as the notebook does.

| file | contents |
|---|---|
| `xcorr_summary.csv` | one row per pair: peak lag and r, trough, r at lag 0, extremum counts |
| `xcorr_peaks_long.csv` | every local peak and trough with its lag, r and prominence |
| `xcorr_driver_vs_driver.csv` | Banff precipitation against the Bow at Banff (notebook cell 27) |
| `curves/` | the full r-vs-lag curve for each pair, 1441 rows apiece |

### `02_hysteresis/` — from `02_hysteresis.ipynb` and `02_01_hysteresis_idx.ipynb`
Seasonal hysteresis index after Zuecco et al. (2016), adapted to closed monthly
climatological loops: the cycle is re-anchored at minimum driver value, both
variables are min-max normalised, and the rising and falling limbs are integrated
over intervals of `du = 0.05`.

**h > 0 is a clockwise loop, h < 0 anticlockwise**; the magnitude is the extent of
the loop. `hyst_class` is Zuecco Table I, I–VIII.

| file | contents |
|---|---|
| `hysteresis_index.csv` | h, class, direction, anchor and peak months, monotonicity, per pair and window |
| `hysteresis_dA/` | the per-interval area differences that sum to h |
| `monthly_climatology.csv` | 12-month mean ± 95 % CI and years per month, per series |
| `daily_climatology.csv` | the same on a fixed 366-day calendar |
| `seasonal_peak_lag.csv` | peak and trough day of the driver and the well, and the offset, unsmoothed and 15-day smoothed |
| `notebook_windows.csv` | which notebook each `window=notebook` date range came from |

Each pair appears with `window=notebook` (the date range the notebooks pass, so
the numbers match the figures) and, where it differs, `window=full_overlap` (the
whole overlapping record, so pairs are comparable with each other).

### `03_trend/` — from `03_trend.ipynb`
Seasonal Mann-Kendall corrected for serial correlation,
`pymannkendall.correlated_seasonal_test(period=12)`.

| file | contents |
|---|---|
| `mann_kendall.csv` | tau, S, var(S), z, p, verdict and Theil-Sen slope per series |

Two row sets. `source=notebook` reproduces exactly what the notebook ran.
`source=uniform` re-runs every series on monthly means over 2005–2024 so slopes
are comparable across series.

### `04_change_point/` — from `04_change_point.ipynb`
BEAST (Bayesian estimator of abrupt change, seasonality and trend) on monthly
means: harmonic season with a 1-year period, at most one trend changepoint and
ten seasonal ones, 24-month segment margins.

| file | contents |
|---|---|
| `beast_changepoints.csv` | each changepoint: date, posterior probability, 90 % CI, step size |
| `beast_model.csv` | expected changepoint counts, R², RMSE, residual variance, marginal likelihood |
| `beast_series.csv` | the fitted trend and season month by month, with CIs and slope sign probabilities |

### `05_shape/` — from `05_shape.ipynb`
| file | contents |
|---|---|
| `interwell_correlation.csv` | Pearson r between z-normalised daily well levels, long form |
| `corr_matrix_marmot.csv`, `corr_matrix_canmore.csv`, `corr_matrix_all.csv` | the same as square matrices, matching the notebook's heatmaps |
| `well_driver_correlation.csv` | lag-0 r between each well and each driver |
| `doy_climatology.csv` | z-normalised day-of-year mean per well |
| `annual_shape_stats.csv` | per well per year: range, timing of the annual maximum and minimum, and hydrograph-shape agreement with the paired driver |

### `06_wavelet/` — from `06_wavelet_transform.ipynb`
pycwt, Morlet(6), `dj = 1/12`, linear detrend and unit variance, with the
notebook's robust lag-1 AR(1) patch. Gaps are filled first by the notebook's
random-forest imputer.

| file | contents |
|---|---|
| `cwt_global_spectrum.csv` | time-averaged power per period, with the AR(1) significance level |
| `cwt_significant_bands.csv` | contiguous period bands whose average power beats red noise |
| `cwt_lag1.csv` | the detrended lag-1 autocorrelation table |
| `xwt_band_summary.csv` | cross-wavelet power, significance and phase per pair per band |
| `wct_band_summary.csv` | wavelet coherence and phase per pair per band |
| `wct_annual_band_by_year.csv` | annual-band coherence and lag, year by year |
| `gap_filling.csv` | how much of each series was imputed, and the imputer's OOB R² |
| `phase_sign_convention.txt` | the lag sign, derived by re-running the notebook's own synthetic test |

Spectra are given for two windows: `full_record`, which is what the notebook
transforms, and `2005-2024`, so stream spectra are comparable with well spectra.
Band statistics use only cells outside the cone of influence.

`well_lag_days > 0` means the well lags the driver — groundwater peaks later.
Read `phase_sign_convention.txt` before using any lag, and ignore lags where
coherence is low.

### `07_lithology/` — from `cross_section.ipynb`
| file | contents |
|---|---|
| `strip_logs.txt` | readable borehole log per well, with the screen marked |
| `well_lithology_intervals.csv` | every logged interval: depths, elevations, material, description |
| `well_completions.csv` | all screen and production records, flagging the one the notebook draws |
| `lithology_summary.csv` | logged depth, material at the screen, overburden thickness, material totals |

## Caveats

1. **`pair_class` is not decoration.** Half the rows are cross-basin pairs that
   exist only because the matrix was filled in. Filter to `primary` unless you
   have a reason not to.

2. **Seasonal Mann-Kendall slope units.** `correlated_seasonal_test(period=12)`
   returns the Theil-Sen slope *per period*, i.e. per 12 samples. With monthly
   input that is per year, which is what the notebook's `slope_m_yr` column
   assumes and it is correct there. But the notebook feeds `brac`, `krns`,
   `brac_e` and `lkl` **daily** data with `period=12`, so those figures are per
   12 *days*: the notebook's "BRatC −0.003051 m³/yr" is really about
   −0.093 m³/s per year. `mann_kendall.csv` states both, in
   `sen_slope_per_period` (with `slope_period` and `sen_slope_units`) and
   `sen_slope_per_year`.

3. **The two hysteresis notebooks disagree on windows.** `02_hysteresis.ipynb`
   plots `brac_e` over 2006–2023 while `02_01_hysteresis_idx.ipynb` plots it over
   2005–2024, and similarly for `bl` and `lkl`. `notebook_windows.csv` records
   which choice was used for each driver, and `window=full_overlap` rows sidestep
   the issue.

4. **No Monte-Carlo significance for wavelet coherence.** The notebook calls
   `wct(..., sig=False)`. Turning it on costs about 13 s per iteration on these
   7305-point series, i.e. roughly an hour per pair at `mc_count=300` and several
   days for the full matrix. Coherence is instead reported with the cone of
   influence masked out and explicit thresholds (`pct_coherence_gt_0p5`,
   `pct_coherence_gt_0p8`). Treat coherence values as descriptive.

5. **BEAST is a sampler.** Changepoint locations and CIs are stable across runs;
   the posterior probabilities move by a few percent. The extraction fixes
   `mcmc_seed=42` — the notebook does not — so repeated runs agree exactly.
   The notebook also passes `start=[2005,1,1]` to every series including
   `brac_e`, whose record begins in 2002; here every series is sliced to
   2005–2024 first and its true start is passed.

6. **`05BH013.csv` (jcnc) contains duplicate records.** Two overlapping blocks
   cover 2012-01-01 to 2024-10-31 twice, disagreeing by up to 22 m on some days.
   Duplicate dates are collapsed to their mean, which is what the notebooks do
   implicitly through `resample('D')` and monthly grouping. The count is in
   `series_inventory.csv`.

7. **Seasonal stations.** `jcnc` and `kv` only report April to November, so their
   hysteresis loops are built from 8 months, and April and November rest on as
   few as 1–4 years. `min_years_per_month` in `hysteresis_index.csv` flags this.

8. **Record lengths differ a lot.** `krns` stops on 2015-12-23, `lkl` on
   2017-12-31, `brac_e` and `brac` on 2023-12-31, and `brab` only starts in 2012.
   Every table carries `n_overlap_days` or `n_days`; a lag estimated on four
   years is not the same evidence as one estimated on nineteen.

9. **The GOWN levels are gap-filled** (`combined_absval_20052024_filled.csv`).
   Every statistic downstream inherits that infilling.

10. **Random-forest gap filling before the wavelet transforms** is the notebook's
    method, and it imposes a smooth seasonal shape on filled stretches. `kv` is
    over half imputed and `jcnc` about a third; see `gap_filling.csv` and treat
    their wavelet results with suspicion.
"""


def run():
    c.write_text(TEXT, "README.md")
    print("  readme: README.md")
