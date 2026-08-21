"""Mann-Kendall trend results from 03_trend.ipynb.

mk.correlated_seasonal_test(period=12) returns the Theil-Sen slope PER PERIOD,
i.e. per 12 samples -- verified against synthetic data with a known slope.
For monthly input that is per year; for the daily input the notebook passes
for brac/krns/brac_e/lkl it is per 12 DAYS. Both readings are recorded.
"""
import numpy as np
import pandas as pd
import pymannkendall as mk

import common as c

PERIOD = 12

# how the notebook fed each series to the test
NOTEBOOK_SAMPLING = {
    **{w: ("monthly_MS", None) for w in c.WELLS},
    "brac":   ("daily_asis", None),
    "krns":   ("daily_asis", None),
    "brac_e": ("daily_asis", None),
    "lkl":    ("daily_asis", "2005"),
    "bl":     ("monthly_MS", "2005"),
    "brab":   ("monthly_MS", "2005"),
}


def _prep(s, sampling, start):
    """NaNs are left in place: the notebook passes the raw .values array and
    pymannkendall's own preprocessing drops them, which is not the same as
    dropping them before the seasonal reshape."""
    x = s.resample("MS").mean() if sampling == "monthly_MS" else s.copy()
    if start:
        x = x.loc[start:]
    return x


def _row(key, x, sampling, source):
    r = mk.correlated_seasonal_test(x.values, period=PERIOD)
    units = c.SERIES_META[key]["units"]
    if sampling == "monthly_MS":
        per_period, per_year = "year", r.slope
        slope_units = f"{units}/yr"
    else:                                    # daily samples, period=12 -> 12 days
        per_period, per_year = "12 days", r.slope * 365.25 / 12
        slope_units = f"{units}/12 days"
    return dict(
        series=key, name=c.SERIES_META[key]["name"],
        kind=c.SERIES_META[key]["kind"], basin=c.SERIES_META[key]["basin"],
        sampling=sampling, source=source, n=len(x),
        n_valid=int(x.notna().sum()),
        start=x.index.min().date(), end=x.index.max().date(),
        tau=r.Tau, S=r.s, var_S=r.var_s, z=r.z, p=r.p,
        significant_p05=bool(r.h), verdict=r.trend,
        sen_slope_per_period=r.slope, slope_period=per_period,
        sen_slope_units=slope_units,
        sen_slope_per_year=per_year,
        sen_slope_per_year_units=f"{units}/yr",
        intercept=r.intercept,
    )


def run(series):
    rows = []

    # (a) exactly what the notebook ran
    for key, (sampling, start) in NOTEBOOK_SAMPLING.items():
        x = _prep(series[key], sampling, start)
        rows.append(_row(key, x, sampling, "notebook"))

    # (b) uniform monthly means over the common 2005-2024 study period, so
    #     slopes are comparable across series
    for key, s in series.items():
        x = _prep(s.loc["2005":"2024"], "monthly_MS", None)
        v = x.dropna()
        if len(v) < 2 * PERIOD:
            continue
        x = x.loc[v.index.min():v.index.max()]   # trim all-NaN ends
        try:
            rows.append(_row(key, x, "monthly_MS", "uniform"))
        except ZeroDivisionError:
            # a season with fewer than two valid values leaves Kendall's
            # denominator at zero; kv is 55% missing and hits this.
            rows.append(dict(series=key, name=c.SERIES_META[key]["name"],
                             sampling="monthly_MS", source="uniform",
                             n=len(x), n_valid=int(x.notna().sum()),
                             start=x.index.min().date(), end=x.index.max().date(),
                             verdict="not computed",
                             note="a month-of-year season has <2 valid values"))

    df = pd.DataFrame(rows)
    c.write_csv(df, "03_trend", "mann_kendall.csv")

    nb = df[(df["source"] == "notebook") & (df["series"].isin(c.WELLS))]
    counts = nb["verdict"].value_counts().to_dict()
    print(f"  trend: {len(df)} rows; notebook wells {counts}")
    return df
