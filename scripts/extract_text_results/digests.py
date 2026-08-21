"""Per-entity plain-text digests assembled from the CSV tables.

One file per well and one per driver, each self-contained: everything the
extraction found about that entity, in reading order, without needing the CSVs.
"""
import numpy as np
import pandas as pd

import common as c

ANNUAL = "annual (10-14 mo)"
RULE = "=" * 78
THIN = "-" * 78


def _load():
    def rd(*p):
        f = c.OUT.joinpath(*p)
        return pd.read_csv(f, dtype={"well": str, "series": str, "well_a": str,
                                     "well_b": str}) if f.exists() else pd.DataFrame()
    return dict(
        inv=rd("00_inventory", "series_inventory.csv"),
        meta=rd("00_inventory", "well_metadata.csv"),
        xc=rd("01_cross_correlation", "xcorr_summary.csv"),
        pk=rd("01_cross_correlation", "xcorr_peaks_long.csv"),
        hy=rd("02_hysteresis", "hysteresis_index.csv"),
        pl=rd("02_hysteresis", "seasonal_peak_lag.csv"),
        mc=rd("02_hysteresis", "monthly_climatology.csv"),
        mk=rd("03_trend", "mann_kendall.csv"),
        cp=rd("04_change_point", "beast_changepoints.csv"),
        bm=rd("04_change_point", "beast_model.csv"),
        bs=rd("04_change_point", "beast_series.csv"),
        iw=rd("05_shape", "interwell_correlation.csv"),
        ann=rd("05_shape", "annual_shape_stats.csv"),
        cb=rd("06_wavelet", "cwt_significant_bands.csv"),
        wc=rd("06_wavelet", "wct_band_summary.csv"),
        xw=rd("06_wavelet", "xwt_band_summary.csv"),
        wy=rd("06_wavelet", "wct_annual_band_by_year.csv"),
        lith=rd("07_lithology", "lithology_summary.csv"),
        strips=(c.OUT / "07_lithology" / "strip_logs.txt").read_text()
        if (c.OUT / "07_lithology" / "strip_logs.txt").exists() else "",
    )


def _fmt(v, spec=".3f", dash="n/a"):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return dash
    try:
        return format(v, spec)
    except (TypeError, ValueError):
        return str(v)


def _one(df, **eq):
    for k, v in eq.items():
        df = df[df[k] == v]
    return df.iloc[0] if len(df) else None


def _record_block(d, key):
    r = _one(d["inv"], key=key)
    if r is None:
        return []
    return [
        "RECORD",
        THIN,
        f"  source            : {r['source']}",
        f"  span              : {r['first']} to {r['last']}  "
        f"({r['span_years']} yr, {r['n_valid']} valid days, "
        f"{r['n_missing']} missing)",
        f"  values ({r['units']:<5}) : min {_fmt(r['min'])}   "
        f"median {_fmt(r['median'])}   mean {_fmt(r['mean'])}   "
        f"max {_fmt(r['max'])}   sd {_fmt(r['std'])}",
    ] + ([f"  NOTE              : {int(r['n_duplicate_dates_collapsed'])} "
          f"duplicate dates in the source file were averaged"]
         if r.get("n_duplicate_dates_collapsed", 0) else []) + [""]


def _trend_block(d, key):
    out = ["TREND  (Mann-Kendall, seasonal, autocorrelation-corrected)", THIN]
    rows = d["mk"][d["mk"]["series"] == key]
    if rows.empty:
        return out + ["  not computed", ""]
    # for the wells the uniform re-run is the same computation as the
    # notebook's, so printing both twice says nothing
    if len(rows) == 2 and np.isclose(rows.iloc[0]["tau"], rows.iloc[1]["tau"]) \
            and np.isclose(rows.iloc[0]["p"], rows.iloc[1]["p"]):
        rows = rows.head(1)
    for _, r in rows.iterrows():
        if r["verdict"] == "not computed":
            out.append(f"  [{r['source']:8s}] not computed -- {r.get('note', '')}")
            continue
        out.append(
            f"  [{r['source']:8s}] {r['sampling']:11s} {r['start']}..{r['end']}  "
            f"n={int(r['n'])}")
        out.append(
            f"             tau {r['tau']:+.4f}   p {r['p']:.4f}   "
            f"{r['verdict'].upper()}")
        out.append(
            f"             Sen slope {r['sen_slope_per_period']:+.5f} "
            f"{r['sen_slope_units']}"
            f"  =  {r['sen_slope_per_year']:+.5f} {r['sen_slope_per_year_units']}")
    out.append("")

    cps = d["cp"][(d["cp"]["series"] == key) & (d["cp"]["component"] == "trend")]
    bm = _one(d["bm"], series=key)
    span = f"{bm['start']} to {bm['end']}" if bm is not None else "2005-2024"
    out.append(f"CHANGEPOINTS  (BEAST, monthly means, {span})")
    out.append(THIN)
    if bm is not None:
        out.append(f"  fit               : R2 {_fmt(bm['R2'])} "
                   f"(trend + season vs observed)  "
                   f"RMSE {_fmt(bm['RMSE'], '.4f')}  "
                   f"expected trend cps {_fmt(bm.get('trend_ncp'), '.2f')}  "
                   f"seasonal cps {_fmt(bm.get('season_ncp'), '.2f')}")
    if cps.empty:
        out.append("  no trend changepoint detected")
    for _, r in cps.iterrows():
        out.append(
            f"  trend cp          : {r['cp_date']}  probability {r['cp_prob']:.3f}  "
            f"90% CI {r['cp_ci_lo_date']} to {r['cp_ci_hi_date']}  "
            f"step {r['abrupt_change']:+.3f}")
    # mean fitted trend slope either side of the trend changepoint
    ser = d["bs"][d["bs"]["series"] == key] if len(d["bs"]) else None
    if ser is not None and len(ser) and len(cps):
        cut = pd.Timestamp(cps.iloc[0]["cp_date"])
        dt = pd.to_datetime(ser["date"])
        before, after = ser[dt < cut], ser[dt >= cut]
        if len(before) and len(after):
            out.append(
                f"  segment slopes    : {before['slope_per_month'].mean() * 12:+.4f} "
                f"{c.SERIES_META[key]['units']}/yr before, "
                f"{after['slope_per_month'].mean() * 12:+.4f} after")

    scp = d["cp"][(d["cp"]["series"] == key) & (d["cp"]["component"] == "season")]
    for _, r in scp.iterrows():
        out.append(f"  seasonal cp       : {r['cp_date']}  "
                   f"probability {r['cp_prob']:.3f}")
    return out + [""]


def _cwt_block(d, key):
    out = ["PERIODICITY  (continuous wavelet transform, time-averaged spectrum)",
           THIN]
    for window in ("2005-2024", "full_record"):
        rows = d["cb"][(d["cb"]["series"] == key) & (d["cb"]["window"] == window)]
        if rows.empty:
            continue
        out.append(f"  [{window}] bands above the AR(1) red-noise level:")
        for _, r in rows.sort_values("peak_signif_ratio", ascending=False).iterrows():
            out.append(
                f"     {r['band_lo_months']:7.2f}-{r['band_hi_months']:.2f} mo   "
                f"peak {r['peak_period_months']:.2f} mo "
                f"({r['peak_period_days']:.0f} d)   "
                f"power/noise {r['peak_signif_ratio']:.2f}   "
                f"significant {r['pct_time_significant']:.0f}% of the record")
    if len(out) == 2:
        out.append("  no band exceeded the red-noise level")
    return out + [""]


def _climatology_block(d, key):
    rows = d["mc"][d["mc"]["series"] == key]
    if rows.empty:
        return []
    rows = rows[rows["valid"]]
    if rows.empty:
        return []
    out = ["MEAN ANNUAL CYCLE  (monthly climatology, mean +/- 95% CI)", THIN]
    for _, r in rows.iterrows():
        out.append(f"     {r['month_name']}  {r['mean']:12.3f} +/- "
                   f"{r['ci95_halfwidth']:7.3f}   ({int(r['n_years'])} yr)")
    hi = rows.loc[rows["mean"].idxmax()]
    lo = rows.loc[rows["mean"].idxmin()]
    out.append(f"  peak {hi['month_name']}, trough {lo['month_name']}, "
               f"seasonal range {hi['mean'] - lo['mean']:.3f} {rows.iloc[0]['units']}")
    return out + [""]


def _pair_block(d, driver, well, from_well):
    """One driver-well relationship, written from either side."""
    other = driver if from_well else well
    label = (f"{c.SERIES_META[driver]['name']}" if from_well
             else f"GOWN_{well} ({c.SERIES_META[well]['basin']})")
    klass = c.pair_class(driver, well)
    out = [f"  >> {label}   [{klass}]"]

    r = _one(d["xc"], driver=driver, well=well)
    if r is not None and np.isfinite(r.get("peak_r", np.nan)):
        out.append(
            f"       cross-correlation : peak r {r['peak_r']:+.3f} at "
            f"{int(r['peak_lag_d']):+d} d"
            f"   (r at lag 0 {r['r_lag0']:+.3f};"
            f" min r {r['trough_r']:+.3f} at {int(r['trough_lag_d']):+d} d;"
            f" {int(r['n_overlap_days'])} overlapping days)")
        if np.isfinite(r.get("first_pos_peak_lag_d", np.nan)):
            out.append(
                f"                           first local peak at a non-negative "
                f"lag: {int(r['first_pos_peak_lag_d']):+d} d, "
                f"r {r['first_pos_peak_r']:+.3f}")

    h = _one(d["hy"], driver=driver, well=well, window="notebook")
    if h is not None and "h" in h and np.isfinite(h.get("h", np.nan)):
        out.append(
            f"       hysteresis loop   : h {h['h']:+.3f}  class {h['hyst_class']}  "
            f"{h['direction']}")
        out.append(
            f"                           {h['data_start']} to {h['data_end']}, "
            f"{int(h['n_valid_months'])} months; driver peaks "
            f"{h['driver_peak_month_name']}, well peaks {h['gw_peak_month_name']}; "
            f"well range {h['gw_range_m']:.3f} m")
        if int(h["min_years_per_month"]) < 5:
            out.append(
                f"                           CAUTION: one month rests on only "
                f"{int(h['min_years_per_month'])} years")

    p = d["pl"][(d["pl"]["driver"] == driver) & (d["pl"]["well"] == well) &
                (d["pl"]["window"] == "notebook") & (d["pl"]["smooth_days"] == 15)]
    if len(p):
        p = p.iloc[0]
        out.append(
            f"       climatology peaks : driver {p['flow_peak_day']}, "
            f"well {p['gw_peak_day']}  ->  well lags by "
            f"{int(p['peak_lag_days']):+d} d  (15-day smoothing)")

    w = _one(d["wc"], driver=driver, well=well, band=ANNUAL)
    if w is not None:
        out.append(
            f"       annual coherence  : {w['mean_coherence']:.3f} mean, "
            f"{w['pct_coherence_gt_0p8']:.0f}% of the band above 0.8; "
            f"well lags driver by {w['well_lag_days']:+.0f} d "
            f"({w['mean_phase_deg']:+.0f} deg)")
    x = _one(d["xw"], driver=driver, well=well, band=ANNUAL)
    if x is not None:
        out.append(
            f"       annual cross-power: {x['mean_cross_power']:.3f} mean, "
            f"{x['pct_significant']:.0f}% significant at 95%")

    yy = d["wy"][(d["wy"]["driver"] == driver) & (d["wy"]["well"] == well)]
    if len(yy) >= 3:
        best = yy.loc[yy["mean_coherence"].idxmax()]
        worst = yy.loc[yy["mean_coherence"].idxmin()]
        out.append(
            f"       coherence by year : strongest {int(best['year'])} "
            f"({best['mean_coherence']:.3f}), weakest {int(worst['year'])} "
            f"({worst['mean_coherence']:.3f})")
    return out + [""]


def well_digest(d, well):
    meta = _one(d["meta"], well=well)
    lith = _one(d["lith"], well=well)
    basin = c.SERIES_META[well]["basin"]

    L = [RULE, f"GOWN_{well}   --   {basin.title()} area   "
               f"({meta['station_no'] if meta is not None else ''})", RULE,
         "Textual extraction of every analysis in notebooks/bow_valley/.",
         "Units: groundwater level in metres above sea level (mamsl).", ""]

    L += ["SITE", THIN]
    if meta is not None:
        L += [
            f"  location          : {meta['latitude']:.5f}, {meta['longitude']:.5f}"
            f"   (UTM11N {meta['x_utm11n']:.0f}, {meta['y_utm11n']:.0f})",
            f"  measuring point   : {meta['mp_elev_m']} mamsl   "
            f"(DEM ground {meta['dem_ground_m']} m)",
            f"  depth             : {meta['well_depth_m']} m  "
            f"(TD {meta['td_m']} m)",
            f"  aquifer           : {meta['aquifer']} / "
            f"{meta['aquifer_lithology']}  ({meta['aquifer_type']})",
            f"  completion        : {meta['notebook_from_m']}-"
            f"{meta['notebook_to_m']} m  "
            f"({meta['notebook_base_elev_m']}-{meta['notebook_top_elev_m']} mamsl)"
            f"  [{meta['notebook_type']}]",
            f"  use / drilled     : {meta['well_use']}, "
            f"{str(meta['drill_date'])[:10]}",
        ]
    if lith is not None:
        L += [
            f"  material at screen: {lith['material_at_screen']}",
            f"  overburden        : {_fmt(lith['overburden_thickness_m'], '.2f', 'no bedrock logged')} m to first bedrock unit",
            f"  logged materials  : {lith['materials']}",
        ]
    L.append("")

    # the interval table for this well, pulled out of the shared strip log
    # (its header repeats the SITE block above, so only the table is kept)
    blocks = d["strips"].split("=" * 78)
    for i, b in enumerate(blocks):
        if b.strip().startswith(f"GOWN_{well}"):
            rows = blocks[i + 1].split("\n")
            start = next((j for j, ln in enumerate(rows)
                          if ln.strip().startswith("from")), None)
            if start is not None:
                L += ["LITHOLOGY LOG", THIN]
                L += [ln for ln in rows[start:] if ln.strip()]
                L.append("")
            break

    L += _record_block(d, well)
    L += _trend_block(d, well)
    L += _climatology_block(d, well)

    ann = d["ann"][d["ann"]["well"] == well]
    if len(ann):
        L += ["YEAR-TO-YEAR BEHAVIOUR", THIN,
              f"  annual range      : {ann['range_m'].mean():.3f} m mean, "
              f"{ann['range_m'].min():.3f}-{ann['range_m'].max():.3f} m",
              f"  annual max        : typically day {int(ann['doy_max'].median())} "
              f"of the year (range {int(ann['doy_max'].min())}-"
              f"{int(ann['doy_max'].max())})",
              f"  annual min        : typically day {int(ann['doy_min'].median())} "
              f"of the year (range {int(ann['doy_min'].min())}-"
              f"{int(ann['doy_min'].max())})",
              f"  wettest year      : {int(ann.loc[ann['mean'].idxmax(), 'year'])} "
              f"(mean {ann['mean'].max():.2f} m); driest "
              f"{int(ann.loc[ann['mean'].idxmin(), 'year'])} "
              f"(mean {ann['mean'].min():.2f} m)",
              f"  shape agreement   : r with {ann.iloc[0]['driver']} averages "
              f"{ann['r_with_driver'].mean():+.3f} "
              f"({ann['r_with_driver'].min():+.3f} to "
              f"{ann['r_with_driver'].max():+.3f} across years)", ""]

    L += _cwt_block(d, well)

    iw = d["iw"][(d["iw"]["group"] == "all") & (d["iw"]["well_a"] == well) &
                 (d["iw"]["well_b"] != well)]
    if len(iw):
        L += ["CORRELATION WITH THE OTHER WELLS  (z-normalised daily levels)",
              THIN]
        for _, r in iw.sort_values("r", ascending=False).iterrows():
            L.append(f"     GOWN_{r['well_b']}  r {r['r']:+.3f}"
                     f"{'   (same area)' if r['same_basin'] else ''}")
        L.append("")

    L += ["RELATION TO EACH DRIVER SERIES", THIN,
          "  Sign conventions: cross-correlation lag > 0 means the driver leads",
          "  the well. well_lag_days > 0 means the well lags the driver.",
          "  Hysteresis h > 0 is a clockwise loop, h < 0 anticlockwise.", ""]
    ordered = ([dv for dv in c.DRIVERS if c.pair_class(dv, well) == "primary"] +
               [dv for dv in c.DRIVERS if c.pair_class(dv, well) == "cross_basin"])
    for dv in ordered:
        L += _pair_block(d, dv, well, from_well=True)

    return "\n".join(L)


def driver_digest(d, key):
    meta = c.SERIES_META[key]
    L = [RULE, f"{meta['name']}   --   key '{key}'   station {meta['station']}",
         RULE,
         "Textual extraction of every analysis in notebooks/bow_valley/.",
         f"Variable: {meta['var']} in {meta['units']}.  Basin: {meta['basin']}.",
         ""]
    L += _record_block(d, key)
    L += _trend_block(d, key)
    L += _climatology_block(d, key)
    L += _cwt_block(d, key)

    L += ["RELATION TO EACH GOWN WELL", THIN,
          "  Sign conventions: cross-correlation lag > 0 means this driver leads",
          "  the well. well_lag_days > 0 means the well lags this driver.",
          "  Hysteresis h > 0 is a clockwise loop, h < 0 anticlockwise.", ""]
    ordered = ([w for w in c.WELLS if c.pair_class(key, w) == "primary"] +
               [w for w in c.WELLS if c.pair_class(key, w) == "cross_basin"])
    for w in ordered:
        L += _pair_block(d, key, w, from_well=False)
    return "\n".join(L)


def run():
    d = _load()
    n = 0
    for w in c.WELLS:
        c.write_text(well_digest(d, w), "summaries", f"GOWN_{w}.txt")
        n += 1
    for k in c.DRIVERS:
        c.write_text(driver_digest(d, k), "summaries", f"driver_{k}.txt")
        n += 1
    print(f"  digests: {n} files")
