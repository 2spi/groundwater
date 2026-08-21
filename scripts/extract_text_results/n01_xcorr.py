"""Cross-correlation results from 01_cross_correlation.ipynb.

xcorr_daily() and the peak/trough detection are the notebook's, with the
drawing removed and the numbers kept.
"""
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

import common as c

MAX_LAG = 720
MIN_PERIODS = 365
PROMINENCE = 0.05


def xcorr_daily(x, y, max_lag=MAX_LAG):
    """Pearson r at daily lags. Positive lag = x leads y."""
    x = x.resample("D").mean()
    y = y.resample("D").mean()
    x, y = x.align(y, join="inner")

    lags = np.arange(-max_lag, max_lag + 1)
    r = [x.corr(y.shift(-lag), min_periods=MIN_PERIODS) for lag in lags]
    return pd.Series(r, index=lags, name="r")


def extrema(r):
    """Local peaks and troughs, as plot_xcorr marks them."""
    valid = r.dropna()
    loc, props = find_peaks(valid.values, prominence=PROMINENCE)
    tloc, tprops = find_peaks(-valid.values, prominence=PROMINENCE)
    rows = [dict(type="peak", lag_d=int(valid.index[i]), r=float(valid.values[i]),
                 prominence=float(p))
            for i, p in zip(loc, props["prominences"])]
    rows += [dict(type="trough", lag_d=int(valid.index[i]), r=float(valid.values[i]),
                  prominence=float(p))
             for i, p in zip(tloc, tprops["prominences"])]
    return sorted(rows, key=lambda d: d["lag_d"])


def run(series, write_curves=True):
    summary, peaks = [], []

    for driver, well, klass in c.pairs():
        x, y = series[driver], series[well]
        r = xcorr_daily(x, y)
        valid = r.dropna()
        n_over = len(c.overlap(x, y))

        if valid.empty:
            summary.append(dict(driver=driver, well=well, pair_class=klass,
                                n_overlap_days=n_over, n_valid_lags=0,
                                note="no lag met min_periods=365"))
            continue

        ex = extrema(r)
        for e in ex:
            peaks.append(dict(driver=driver, well=well, pair_class=klass, **e))

        peak_lag = int(valid.idxmax())
        trough_lag = int(valid.idxmin())
        pos_peaks = [e for e in ex if e["type"] == "peak" and e["lag_d"] >= 0]
        near = min(ex, key=lambda e: abs(e["lag_d"])) if ex else None

        summary.append(dict(
            driver=driver, well=well, pair_class=klass,
            driver_units=c.SERIES_META[driver]["units"],
            n_overlap_days=n_over, n_valid_lags=len(valid),
            r_lag0=float(r.loc[0]) if 0 in valid.index else np.nan,
            peak_lag_d=peak_lag, peak_r=float(valid.max()),
            trough_lag_d=trough_lag, trough_r=float(valid.min()),
            n_local_peaks=sum(e["type"] == "peak" for e in ex),
            n_local_troughs=sum(e["type"] == "trough" for e in ex),
            first_pos_peak_lag_d=pos_peaks[0]["lag_d"] if pos_peaks else np.nan,
            first_pos_peak_r=pos_peaks[0]["r"] if pos_peaks else np.nan,
            nearest_extremum_lag_d=near["lag_d"] if near else np.nan,
            nearest_extremum_type=near["type"] if near else None,
            nearest_extremum_r=near["r"] if near else np.nan,
            note="",
        ))

        if write_curves:
            c.write_csv(r.rename_axis("lag_d").reset_index(),
                        "01_cross_correlation", "curves",
                        f"{driver}__GOWN_{well}.csv")

    # the notebook also runs precip at Banff against the Bow at Banff
    r = xcorr_daily(series["banff"], series["brab"])
    v = r.dropna()
    extra = pd.DataFrame([dict(
        driver="banff", target="brab", n_valid_lags=len(v),
        peak_lag_d=int(v.idxmax()), peak_r=float(v.max()),
        trough_lag_d=int(v.idxmin()), trough_r=float(v.min()),
        r_lag0=float(r.loc[0]))])

    sm = pd.DataFrame(summary)
    c.write_csv(sm, "01_cross_correlation", "xcorr_summary.csv")
    c.write_csv(pd.DataFrame(peaks), "01_cross_correlation", "xcorr_peaks_long.csv")
    c.write_csv(extra, "01_cross_correlation", "xcorr_driver_vs_driver.csv")
    print(f"  xcorr: {len(sm)} pairs, {len(peaks)} extrema")
    return sm
