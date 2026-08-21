"""Hydrograph-shape results from 05_shape.ipynb.

The notebook renders three correlation heatmaps and two grids of z-normalised
hydrographs; this turns both into tables.
"""
import numpy as np
import pandas as pd

import common as c

GROUPS = {
    "marmot": c.MARMOT_WELLS,
    "canmore": c.CANMORE_WELLS,
    "all": c.WELLS,
}


def interwell_correlation(series):
    z = pd.DataFrame({w: c.z_norm(series[w]) for w in c.WELLS})
    rows = []
    for gname, members in GROUPS.items():
        corr = z[members].corr()
        for a in members:
            for b in members:
                rows.append(dict(group=gname, well_a=a, well_b=b,
                                 r=corr.loc[a, b],
                                 same_basin=(a in c.MARMOT_WELLS) ==
                                            (b in c.MARMOT_WELLS)))
    return pd.DataFrame(rows)


def well_driver_correlation(series):
    rows = []
    for driver, well, klass in c.pairs():
        idx = c.overlap(series[driver], series[well])
        if len(idx) < 365:
            rows.append(dict(driver=driver, well=well, pair_class=klass,
                             n=len(idx), r_lag0=np.nan,
                             note="fewer than 365 overlapping days"))
            continue
        a, b = series[driver].loc[idx], series[well].loc[idx]
        rows.append(dict(driver=driver, well=well, pair_class=klass,
                         n=len(idx), r_lag0=float(a.corr(b)),
                         start=idx.min().date(), end=idx.max().date(),
                         note=""))
    return pd.DataFrame(rows)


def doy_climatology(series):
    """z-normalised day-of-year mean per well, as the small-multiples plot."""
    rows = []
    for w in c.WELLS:
        s = series[w]
        x = c.z_norm(s.groupby(s.index.strftime("%m-%d")).mean())
        raw = s.groupby(s.index.strftime("%m-%d")).mean()
        for day, zv in x.items():
            rows.append(dict(well=w, month_day=day, z=zv, mean_mamsl=raw[day]))
    return pd.DataFrame(rows)


def annual_shape_stats(series):
    """Per well per year: level statistics and shape agreement with its driver."""
    rows = []
    for w in c.WELLS:
        drv_key = c.primary_driver(w)
        drv = series[drv_key]
        for year in range(2005, 2025):
            lo, hi = f"{year}-01-01", f"{year}-12-31"
            y = series[w].loc[lo:hi].dropna()
            if y.empty:
                continue
            d = drv.loc[lo:hi].dropna()      # empty once the driver record ends
            idx = y.index.intersection(d.index)
            r = float(c.z_norm(y.loc[idx]).corr(c.z_norm(d.loc[idx]))) \
                if len(idx) >= 90 else np.nan
            rows.append(dict(
                well=w, basin=c.SERIES_META[w]["basin"], year=year,
                n_days=len(y), mean=y.mean(), min=y.min(), max=y.max(),
                range_m=y.max() - y.min(), std=y.std(),
                doy_max=int(y.idxmax().dayofyear),
                date_max=y.idxmax().date(),
                doy_min=int(y.idxmin().dayofyear),
                date_min=y.idxmin().date(),
                driver=drv_key, n_days_driver_overlap=len(idx),
                r_with_driver=r,
            ))
    return pd.DataFrame(rows)


def run(series):
    iw = interwell_correlation(series)
    c.write_csv(iw, "05_shape", "interwell_correlation.csv")
    c.write_csv(well_driver_correlation(series), "05_shape",
                "well_driver_correlation.csv")
    c.write_csv(doy_climatology(series), "05_shape", "doy_climatology.csv")
    ann = annual_shape_stats(series)
    c.write_csv(ann, "05_shape", "annual_shape_stats.csv")

    # square matrices, as the heatmaps show them
    z = pd.DataFrame({w: c.z_norm(series[w]) for w in c.WELLS})
    for gname, members in GROUPS.items():
        c.write_csv(z[members].corr().round(4).rename_axis("well").reset_index(),
                    "05_shape", f"corr_matrix_{gname}.csv")

    print(f"  shape: {len(iw)} correlation rows, {len(ann)} well-years")
    return ann
