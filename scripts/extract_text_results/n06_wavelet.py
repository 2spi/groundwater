"""Wavelet results from 06_wavelet_transform.ipynb.

Uses the notebook's pycwt path: Morlet(6), dj=1/12, linear detrend and unit
variance, and the notebook's robust lag-1 AR(1) patch. Gaps are filled with
the notebook's random-forest imputer before transforming.

Wavelet coherence is computed with sig=False, exactly as the notebook calls
it. Monte-Carlo significance at mc_count=300 costs ~13 s per iteration on
these 7305-point series (~65 min per pair, ~4 days for the 90-pair matrix),
so it is deliberately not computed; coherence is reported with the cone of
influence masked out and explicit coherence thresholds instead.
"""
import warnings

import numpy as np
import pandas as pd

import common as c  # noqa: F401  -- installs the np.int shim pycwt needs

import pycwt as wavelet
import pycwt.wavelet
import pycwt.helpers

DJ = 1 / 12
MOTHER = wavelet.Morlet(6)
MONTH = c.MONTH_DAYS

# period bands, in months
BANDS = [
    ("sub-monthly", 0.0, 1.0),
    ("1-3 mo", 1.0, 3.0),
    ("3-6 mo", 3.0, 6.0),
    ("semiannual (6-9 mo)", 6.0, 9.0),
    ("annual (10-14 mo)", 10.0, 14.0),
    ("interannual (20-30 mo)", 20.0, 30.0),
    ("30-60 mo", 30.0, 60.0),
    ("decadal (>60 mo)", 60.0, np.inf),
]
ANNUAL = ("annual (10-14 mo)", 10.0, 14.0)


def _lag1(y):
    """Plain, always-defined lag-1 autocorrelation (notebook cell 13)."""
    y = np.asarray(y, float)
    y = y - y.mean()
    return float(np.clip(np.corrcoef(y[:-1], y[1:])[0, 1], -0.999, 0.999))


def _robust_ar1(x):
    return _lag1(x), 1.0, 0.0


pycwt.wavelet.ar1 = _robust_ar1
pycwt.helpers.ar1 = _robust_ar1
pycwt.ar1 = _robust_ar1


def _detrend(y):
    n = len(y)
    return y - np.polyval(np.polyfit(np.arange(n), y, 1), np.arange(n))


def impute_rf_series(s, n_estimators=200, random_state=0):
    """Random-forest gap fill (notebook cell 3). Returns (series, OOB R2)."""
    from sklearn.ensemble import RandomForestRegressor

    s = s.sort_index().astype(float)
    idx = s.index
    time_feat = pd.DataFrame({
        "t": np.arange(len(idx)),
        "year": idx.year,
        "sin_doy": np.sin(2 * np.pi * idx.dayofyear / 365.25),
        "cos_doy": np.cos(2 * np.pi * idx.dayofyear / 365.25),
        "sin_month": np.sin(2 * np.pi * idx.month / 12),
        "cos_month": np.cos(2 * np.pi * idx.month / 12),
    }, index=idx)

    m = s.notna()
    miss = ~m
    out = s.copy()
    if m.all():
        return out, 1.0
    if m.sum() < 50:
        return s.interpolate(method="time").ffill().bfill(), float("nan")

    rf = RandomForestRegressor(n_estimators=n_estimators, max_features="sqrt",
                               min_samples_leaf=2, n_jobs=-1,
                               random_state=random_state, oob_score=True)
    rf.fit(time_feat.loc[m], s.loc[m])
    out.loc[miss] = rf.predict(time_feat.loc[miss])
    return out, float(rf.oob_score_)


_PREP = {}
_PREP_INFO = []

STUDY = ("2005", "2024")   # the window the well records define


def prepared(key, series, window=STUDY):
    """Series on a gap-free daily grid, gaps filled by the notebook's RF model.

    `window` is applied before imputation so the model is never trained on
    decades outside the period being analysed (lkl reaches back to 1932).
    None keeps the full record, which is what the notebook's own cwt() sees.
    """
    tag = "full_record" if window is None else "-".join(window)
    if (key, tag) in _PREP:
        return _PREP[(key, tag)]

    s = series[key]
    if window is not None:
        s = s.loc[window[0]:window[1]]
    v = s.dropna()
    if v.empty:
        _PREP[(key, tag)] = v
        return v
    idx = pd.date_range(v.index.min(), v.index.max(), freq="D")
    s = s.reindex(idx)
    n_missing = int(s.isna().sum())
    if n_missing:
        s, oob = impute_rf_series(s)
    else:
        oob = 1.0
    _PREP[(key, tag)] = s
    _PREP_INFO.append(dict(series=key, name=c.SERIES_META[key]["name"],
                           window=tag, n=len(s), n_imputed=n_missing,
                           pct_imputed=round(100 * n_missing / len(s), 2),
                           oob_r2=oob,
                           start=idx.min().date(), end=idx.max().date()))
    return s


def _circmean(a):
    return float(np.angle(np.mean(np.exp(1j * np.asarray(a)))))


def _band_mask(per_m, lo, hi):
    return (per_m >= lo) & (per_m < hi)


def _coi_mask(per_m, coi_m):
    """True where the period lies inside the cone of influence (usable)."""
    return per_m[:, None] <= coi_m[None, :]


def cwt_tables(series):
    """Global (time-averaged) wavelet spectra.

    Two windows per series: `full_record`, which is what the notebook's own
    cwt() transforms, and `2005-2024`, the study period the well records
    define, so stream spectra are comparable with well spectra.
    """
    spec_rows, band_rows, lag_rows = [], [], []

    for key in series:
        for window, tag in ((None, "full_record"), (STUDY, "2005-2024")):
            _one_cwt(series, key, window, tag, spec_rows, band_rows, lag_rows)

    return (pd.DataFrame(spec_rows), pd.DataFrame(band_rows),
            pd.DataFrame(lag_rows))


def _one_cwt(series, key, window, tag, spec_rows, band_rows, lag_rows):
    """One global spectrum; appends to the three row lists."""
    s = prepared(key, series, window)
    if len(s) < 730:
        return
    y = _detrend(s.to_numpy(float))
    lag_rows.append(dict(series=key, name=c.SERIES_META[key]["name"],
                         window=tag, n=len(y),
                         start=s.index.min().date(), end=s.index.max().date(),
                         detrended_lag1=round(_lag1(y), 4)))
    yn = y / y.std()

    W, sj, freq, coi, _, _ = wavelet.cwt(yn, 1.0, DJ, -1, -1, MOTHER)
    power = np.abs(W) ** 2
    per_m = (1 / freq) / MONTH
    coi_m = coi / MONTH
    usable = _coi_mask(per_m, coi_m)

    signif, _ = wavelet.significance(1.0, 1.0, sj, 0, _lag1(y), wavelet=MOTHER)

    # the longest scales lie entirely inside the cone of influence, so
    # their time average is over an empty slice and stays NaN by design
    with np.errstate(invalid="ignore"), \
            warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        masked = np.where(usable, power, np.nan)
        tavg = np.nanmean(masked, axis=1)
        pct_sig = np.nanmean(np.where(usable, power > signif[:, None], np.nan),
                             axis=1) * 100
    ratio = tavg / signif
    n_usable = usable.sum(axis=1)

    for i, p in enumerate(per_m):
        spec_rows.append(dict(
            series=key, window=tag, period_months=p, period_days=p * MONTH,
            time_avg_power=tavg[i],
            time_avg_power_rectified=tavg[i] / sj[i],
            signif_level=signif[i], signif_ratio=ratio[i],
            pct_time_significant=pct_sig[i], n_days_outside_coi=int(n_usable[i])))

    # contiguous runs where the time-averaged power beats the AR(1) level
    sig = np.nan_to_num(ratio, nan=0.0) > 1
    i = 0
    while i < len(sig):
        if not sig[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(sig) and sig[j + 1]:
            j += 1
        seg = slice(i, j + 1)
        k = i + int(np.nanargmax(np.nan_to_num(ratio[seg], nan=-np.inf)))
        band_rows.append(dict(
            series=key, window=tag,
            band_lo_months=float(per_m[seg].min()),
            band_hi_months=float(per_m[seg].max()),
            peak_period_months=float(per_m[k]),
            peak_period_days=float(per_m[k] * MONTH),
            peak_signif_ratio=float(ratio[k]),
            peak_time_avg_power=float(tavg[k]),
            pct_time_significant=float(pct_sig[k]),
            n_scales=int(j - i + 1)))
        i = j + 1


def _pair_arrays(series, driver, well):
    a, b = prepared(driver, series), prepared(well, series)
    idx = a.index.intersection(b.index)
    return idx, a.loc[idx].to_numpy(float), b.loc[idx].to_numpy(float)


def _band_stats(per_m, coi_m, field, phase, lo, hi, thresholds):
    """Mean field, circular-mean phase and implied lag inside one period band."""
    bm = _band_mask(per_m, lo, hi)
    if not bm.any():
        return None
    usable = _coi_mask(per_m, coi_m)[bm]
    f = field[bm]
    ph = phase[bm]
    if usable.sum() == 0:
        return None

    vals = f[usable]
    out = dict(n_scales=int(bm.sum()),
               coi_coverage=float(usable.mean()),
               mean_value=float(vals.mean()),
               median_value=float(np.median(vals)),
               max_value=float(vals.max()))
    for t in thresholds:
        out[f"pct_gt_{t}"] = float((vals > t).mean() * 100)

    # per-scale circular mean, then convert with that scale's own period
    per_band = per_m[bm]
    lags, phases = [], []
    for i in range(bm.sum()):
        u = usable[i]
        if u.sum() == 0:
            continue
        p = _circmean(ph[i][u])
        phases.append(p)
        lags.append(p / (2 * np.pi) * per_band[i] * MONTH)
    if phases:
        out["mean_phase_rad"] = _circmean(phases)
        out["mean_phase_deg"] = np.degrees(out["mean_phase_rad"])
        out["mean_lag_days"] = float(np.mean(lags))
    return out


def xwt_wct_tables(series):
    xwt_rows, wct_rows, year_rows = [], [], []

    for driver, well, klass in c.pairs():
        idx, y1, y2 = _pair_arrays(series, driver, well)
        if len(idx) < 730:
            continue
        base = dict(driver=driver, well=well, pair_class=klass,
                    n_days=len(idx), start=idx.min().date(), end=idx.max().date())

        # --- XWT (detrended, as plot_xwt) -----------------------------
        W12, coi, freq, signif = wavelet.xwt(
            _detrend(y1), _detrend(y2), 1.0, dj=DJ, s0=-1, J=-1,
            significance_level=0.95, wavelet=MOTHER, normalize=True)
        power, phase = np.abs(W12), np.angle(W12)
        per_m, coi_m = (1 / freq) / MONTH, coi / MONTH
        sig = power / signif[:, None]
        for label, lo, hi in BANDS:
            st = _band_stats(per_m, coi_m, power, phase, lo, hi, thresholds=[])
            if st is None:
                continue
            sg = _band_stats(per_m, coi_m, sig, phase, lo, hi, thresholds=[1])
            xwt_rows.append(dict(**base, band=label, band_lo_months=lo,
                                 band_hi_months=hi,
                                 mean_cross_power=st["mean_value"],
                                 max_cross_power=st["max_value"],
                                 pct_significant=sg["pct_gt_1"],
                                 mean_phase_rad=st.get("mean_phase_rad"),
                                 mean_phase_deg=st.get("mean_phase_deg"),
                                 well_lag_days=st.get("mean_lag_days"),
                                 n_scales=st["n_scales"],
                                 coi_coverage=st["coi_coverage"]))

        # --- WCT (not detrended, sig=False, as plot_wct) --------------
        WCT, aWCT, coi2, freq2, _ = wavelet.wct(
            y1, y2, 1.0, dj=DJ, s0=-1, J=-1, sig=False,
            significance_level=0.95, wavelet=MOTHER, normalize=True,
            mc_count=300)
        per_m2, coi_m2 = (1 / freq2) / MONTH, coi2 / MONTH
        for label, lo, hi in BANDS:
            st = _band_stats(per_m2, coi_m2, WCT, aWCT, lo, hi,
                             thresholds=[0.5, 0.8])
            if st is None:
                continue
            wct_rows.append(dict(**base, band=label, band_lo_months=lo,
                                 band_hi_months=hi,
                                 mean_coherence=st["mean_value"],
                                 median_coherence=st["median_value"],
                                 max_coherence=st["max_value"],
                                 pct_coherence_gt_0p5=st["pct_gt_0.5"],
                                 pct_coherence_gt_0p8=st["pct_gt_0.8"],
                                 mean_phase_rad=st.get("mean_phase_rad"),
                                 mean_phase_deg=st.get("mean_phase_deg"),
                                 well_lag_days=st.get("mean_lag_days"),
                                 n_scales=st["n_scales"],
                                 coi_coverage=st["coi_coverage"]))

        # --- annual band, year by year -------------------------------
        bm = _band_mask(per_m2, ANNUAL[1], ANNUAL[2])
        usable = _coi_mask(per_m2, coi_m2)
        years = pd.DatetimeIndex(idx).year
        for yr in np.unique(years):
            cols = years == yr
            sub = WCT[bm][:, cols]
            subu = usable[bm][:, cols]
            subp = aWCT[bm][:, cols]
            if subu.sum() < 30:
                continue
            per_band = per_m2[bm]
            lags, phases = [], []
            for i in range(bm.sum()):
                u = subu[i]
                if u.sum() == 0:
                    continue
                p = _circmean(subp[i][u])
                phases.append(p)
                lags.append(p / (2 * np.pi) * per_band[i] * MONTH)
            year_rows.append(dict(
                driver=driver, well=well, pair_class=klass, year=int(yr),
                n_days=int(cols.sum()),
                mean_coherence=float(sub[subu].mean()),
                pct_coherence_gt_0p8=float((sub[subu] > 0.8).mean() * 100),
                mean_phase_rad=_circmean(phases) if phases else np.nan,
                mean_phase_deg=np.degrees(_circmean(phases)) if phases else np.nan,
                well_lag_days=float(np.mean(lags)) if lags else np.nan))

    return (pd.DataFrame(xwt_rows), pd.DataFrame(wct_rows),
            pd.DataFrame(year_rows))


def phase_convention():
    """Re-run the notebook's synthetic quarter-period test (cell 31)."""
    idx = pd.date_range("2005-01-01", periods=3650, freq="D")
    k = np.arange(3650)
    T, lag = 365.0, 365.0 / 4
    s_orig = pd.Series(np.sin(2 * np.pi * k / T), idx)
    s_later = pd.Series(np.sin(2 * np.pi * (k - lag) / T), idx)

    WCT, aWCT, coi, freq, _ = wavelet.wct(
        s_later.to_numpy(), s_orig.to_numpy(), 1.0, dj=DJ, s0=-1, J=-1,
        sig=False, significance_level=0.95, wavelet=MOTHER, normalize=True,
        mc_count=300)
    per_m, coi_m = (1 / freq) / MONTH, coi / MONTH
    bm = _band_mask(per_m, 10.0, 14.0)
    usable = _coi_mask(per_m, coi_m)[bm]
    ph = _circmean(aWCT[bm][usable])
    lag_days = ph / (2 * np.pi) * 365.25

    # calibration: arg1 was built to peak +91.3 d after arg2, and the measured
    # lag came back at lag_days. So the delay of arg1 behind arg2 is -lag_days,
    # and the delay of arg2 behind arg1 is +lag_days.
    return f"""Phase and lag sign convention
=============================

Reproduces cell 31 of 06_wavelet_transform.ipynb: two pure sine waves with a
365-day period, the first delayed by a quarter period (91.3 d) relative to the
second, passed as wct(s_later, s_orig).

  annual-band circular-mean phase : {ph:+.4f} rad ({np.degrees(ph):+.1f} deg)
  lag = phase / (2*pi) * period   : {lag_days:+.1f} days
  ground truth                    : argument 1 peaks 91.3 d AFTER argument 2

Calibration: argument 1 delayed behind argument 2 by +91.3 d produced a lag of
{lag_days:+.1f} d. The quantity phase/(2*pi)*period therefore measures how far
ARGUMENT 2 lags ARGUMENT 1.

Every call in 06_wavelet/ is made as (driver, well): argument 1 is the
surface-water or precipitation series, argument 2 is the groundwater series.
The column is named accordingly:

  well_lag_days > 0  the WELL lags the DRIVER -- groundwater peaks later,
                     the expected direction for a river or precipitation
                     signal propagating into an aquifer
  well_lag_days < 0  the WELL leads the DRIVER -- groundwater peaks first

It is computed per scale as phase/(2*pi) * that scale's period in days, then
averaged over the scales in the band, using only cells outside the cone of
influence.

Sanity check on real data: Bow River at Canmore vs GOWN_0764 returns about
+42 d in the annual band, and the monthly climatology has the river peaking in
June and that well in July -- consistent. Lower Kananaskis Lake, a regulated
reservoir that peaks in October, returns about -100 d against wells that peak
in June -- also consistent.

Lags mean nothing where the two series are not coherent. Filter on
mean_coherence and pct_coherence_gt_0p8 in wct_band_summary.csv before reading
any lag.
"""


def run(series):
    spec, bands, lag1 = cwt_tables(series)
    c.write_csv(lag1, "06_wavelet", "cwt_lag1.csv")
    c.write_csv(spec, "06_wavelet", "cwt_global_spectrum.csv")
    c.write_csv(bands, "06_wavelet", "cwt_significant_bands.csv")

    xwt_df, wct_df, year_df = xwt_wct_tables(series)
    c.write_csv(xwt_df, "06_wavelet", "xwt_band_summary.csv")
    c.write_csv(wct_df, "06_wavelet", "wct_band_summary.csv")
    c.write_csv(year_df, "06_wavelet", "wct_annual_band_by_year.csv")
    c.write_csv(pd.DataFrame(_PREP_INFO), "06_wavelet", "gap_filling.csv")
    c.write_text(phase_convention(), "06_wavelet", "phase_sign_convention.txt")

    print(f"  wavelet: {len(spec)} spectrum rows, {len(bands)} significant bands, "
          f"{len(wct_df)} coherence band rows")
    return wct_df
