"""Seasonal hysteresis results from 02_hysteresis.ipynb and 02_01_hysteresis_idx.ipynb.

hysteresis_index(), _limb(), _classify() and to_calendar_month() are copied
unchanged from 02_01_hysteresis_idx.ipynb (cells 4-6, 10).
"""
import numpy as np
import pandas as pd

import common as c

_trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
_CLASSES = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII"]

DU = 0.05

# The two notebooks disagree on the window for some drivers. The registry
# window is the one used here; this records where each came from.
WINDOW_SOURCE = {
    "krns":   "02_hysteresis + 02_01 default ('2006'-'2023')",
    "bl":     "02_hysteresis call ('2005'-'2024'); 02_01 uses the default",
    "lkl":    "02_hysteresis call ('2005'-'2017'); 02_01 uses the default",
    "jcnc":   "02_hysteresis call ('2005'-'2024')",
    "brac_e": "02_01 call ('2005'-'2024'); 02_hysteresis uses the default",
    "brab":   "02_hysteresis call ('2005'-'2024')",
    "brac":   "02_hysteresis default ('2006'-'2023')",
    "banff":  "not plotted in 02*; window set to the 2005-2024 record",
    "exshaw": "not plotted in 02*; window set to the 2005-2024 record",
    "kv":     "not plotted in 02*; window set to the 2005-2024 record",
}


def _limb(u, v):
    """Make one limb usable by np.interp: ascending u, duplicates averaged."""
    d = np.diff(u)
    monotonic = bool(np.all(d > 0) or np.all(d < 0))
    idx = np.argsort(u, kind="stable")
    us, vs = u[idx], v[idx]
    uu, inv = np.unique(us, return_inverse=True)
    vv = np.bincount(inv, weights=vs) / np.bincount(inv)
    return uu, vv, monotonic


def _classify(dA_min, dA_max, h, rises_on_rising_limb):
    """Map (dA_min, dA_max, h) onto Zuecco Table I classes I-VIII."""
    if dA_min > 0 and dA_max > 0:
        k = 1                       # pure clockwise
    elif dA_min < 0 and dA_max < 0:
        k = 4                       # pure anticlockwise
    elif h >= 0:
        k = 2                       # figure-eight, dominantly clockwise
    else:
        k = 3                       # figure-eight, dominantly anticlockwise
    return _CLASSES[(0 if rises_on_rising_limb else 4) + k - 1]


def hysteresis_index(x, y, du=DU, u_start=0.0, n_sub=51):
    """Hysteresis index h for one closed loop of monthly means."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError("x and y must be 1-D arrays of equal length.")
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
        raise ValueError("x and y must be finite; drop or fill invalid months first.")

    n = x.size

    # 1. re-anchor the cycle at min(x), then close the loop
    i0 = int(np.argmin(x))
    order = np.roll(np.arange(n), -i0)
    xs = np.append(x[order], x[order][0])
    ys = np.append(y[order], y[order][0])

    # 2. min-max normalise both variables to [0, 1]
    u = (xs - xs.min()) / (xs.max() - xs.min())
    v = (ys - ys.min()) / (ys.max() - ys.min())

    # 3. split at the peak of the independent variable
    ip = int(np.argmax(u))
    ur, vr, mono_r = _limb(u[: ip + 1], v[: ip + 1])
    uf, vf, mono_f = _limb(u[ip:], v[ip:])

    # 4. definite integrals on fixed intervals of u, and their difference
    edges = np.arange(u_start, 1.0 + 1e-9, du)
    if edges.size < 2:
        raise ValueError("du too large for the requested integration range.")
    dA = np.empty(edges.size - 1)
    for k in range(dA.size):
        g = np.linspace(edges[k], edges[k + 1], n_sub)
        dA[k] = _trapz(np.interp(g, ur, vr), g) - _trapz(np.interp(g, uf, vf), g)

    h = float(dA.sum())
    dA_min, dA_max = float(dA.min()), float(dA.max())
    rises = bool(v[ip] >= v[0])

    return {
        "h": h,
        "dA_min": dA_min,
        "dA_max": dA_max,
        "dA": dA,
        "edges": edges,
        "hyst_class": _classify(dA_min, dA_max, h, rises),
        "anchor_month": int(order[0]) + 1,
        "peak_month": int(order[ip % n]) + 1,
        "monotonic_rising": mono_r,
        "monotonic_falling": mono_f,
        "gw_rises_on_rising_limb": rises,
    }


def to_calendar_month(pos, valid_months):
    """Translate a 1-based position from hysteresis_index into a calendar month."""
    return int(valid_months[pos - 1]) + 1


def _direction(res):
    if res["dA_min"] > 0 and res["dA_max"] > 0:
        return "clockwise"
    if res["dA_min"] < 0 and res["dA_max"] < 0:
        return "anticlockwise"
    return "figure-eight (clockwise-dominant)" if res["h"] >= 0 \
        else "figure-eight (anticlockwise-dominant)"


def _windows(driver, well, series):
    """(label, start, end) pairs: the notebook window and the full overlap."""
    nb = c.SERIES_META[driver]["notebook_window"]
    idx = c.overlap(series[driver], series[well])
    full = (str(idx.min().year), str(idx.max().year)) if len(idx) else nb
    out = [("notebook", *nb)]
    if full != nb:
        out.append(("full_overlap", *full))
    return out


def climatologies(series):
    """Monthly and daily climatology tables for every series, notebook window."""
    m_rows, d_rows = [], []
    for key, s in series.items():
        start, end = c.SERIES_META[key]["notebook_window"]
        w = s.loc[start:end].dropna()
        if w.empty:
            continue
        mean, ci, valid, n = c.monthly_stats(w)
        for i in range(12):
            m_rows.append(dict(series=key, window=f"{start}-{end}",
                               month=i + 1, month_name=c.MONTH_ABBR[i],
                               mean=mean[i], ci95_halfwidth=ci[i],
                               n_years=int(n[i]), valid=bool(valid[i]),
                               units=c.SERIES_META[key]["units"]))
        mean, ci, valid, n = c.daily_stats(w)
        for i in range(366):
            d_rows.append(dict(series=key, window=f"{start}-{end}",
                               day_slot=i, day_label=c.slot_label(i),
                               mean=mean[i], ci95_halfwidth=ci[i],
                               n_years=int(n[i]), valid=bool(valid[i])))
    return pd.DataFrame(m_rows), pd.DataFrame(d_rows)


def _peak_lag(flow, gw, valid, smooth):
    """Peak/trough day offsets from the daily climatology (plot_seasonal_cycle)."""
    f = np.where(valid, flow, np.nan)
    g = np.where(valid, gw, np.nan)
    if smooth > 1:
        pad = int(smooth)
        for arr_name in ("f", "g"):
            a = f if arr_name == "f" else g
            ext = np.concatenate([a[-pad:], a, a[:pad]])
            sm = pd.Series(ext).rolling(smooth, center=True, min_periods=1).mean().values
            sm = np.where(np.isnan(a), np.nan, sm[pad:pad + a.size])
            if arr_name == "f":
                f = sm
            else:
                g = sm
    if np.all(np.isnan(f)) or np.all(np.isnan(g)):
        return {}
    fmax, gmax = int(np.nanargmax(f)), int(np.nanargmax(g))
    fmin, gmin = int(np.nanargmin(f)), int(np.nanargmin(g))
    return dict(
        smooth_days=smooth,
        flow_peak_day=c.slot_label(fmax), gw_peak_day=c.slot_label(gmax),
        peak_lag_days=gmax - fmax,
        flow_min_day=c.slot_label(fmin), gw_min_day=c.slot_label(gmin),
        min_lag_days=gmin - fmin,
        flow_amplitude=float(np.nanmax(f) - np.nanmin(f)),
        gw_amplitude=float(np.nanmax(g) - np.nanmin(g)),
    )


def run(series):
    m_clim, d_clim = climatologies(series)
    c.write_csv(m_clim, "02_hysteresis", "monthly_climatology.csv")
    c.write_csv(d_clim, "02_hysteresis", "daily_climatology.csv")

    idx_rows, lag_rows = [], []

    for driver, well, klass in c.pairs():
        for label, start, end in _windows(driver, well, series):
            dr = series[driver].loc[start:end].dropna()
            gw = series[well].loc[start:end].dropna()
            if dr.empty or gw.empty:
                continue

            f_mean, f_ci, f_valid, f_n = c.monthly_stats(dr)
            g_mean, g_ci, g_valid, g_n = c.monthly_stats(gw)
            valid = f_valid & g_valid
            vm = np.where(valid)[0]
            if vm.size < 3:
                idx_rows.append(dict(driver=driver, well=well, pair_class=klass,
                                     window=label, start=start, end=end,
                                     n_valid_months=int(vm.size),
                                     note="too few overlapping months"))
                continue

            res = hysteresis_index(f_mean[vm], g_mean[vm])
            idx_rows.append(dict(
                driver=driver, well=well, pair_class=klass,
                window=label, start=start, end=end,
                # the window is a request; these are the dates actually used
                data_start=max(dr.index.min(), gw.index.min()).date(),
                data_end=min(dr.index.max(), gw.index.max()).date(),
                h=res["h"], hyst_class=res["hyst_class"],
                direction=_direction(res),
                dA_min=res["dA_min"], dA_max=res["dA_max"],
                anchor_month=to_calendar_month(res["anchor_month"], vm),
                anchor_month_name=c.MONTH_ABBR[
                    to_calendar_month(res["anchor_month"], vm) - 1],
                driver_peak_month=to_calendar_month(res["peak_month"], vm),
                driver_peak_month_name=c.MONTH_ABBR[
                    to_calendar_month(res["peak_month"], vm) - 1],
                gw_peak_month=int(vm[int(np.argmax(g_mean[vm]))]) + 1,
                gw_peak_month_name=c.MONTH_ABBR[
                    int(vm[int(np.argmax(g_mean[vm]))])],
                gw_rises_on_rising_limb=res["gw_rises_on_rising_limb"],
                monotonic_rising=res["monotonic_rising"],
                monotonic_falling=res["monotonic_falling"],
                n_valid_months=int(vm.size),
                valid_months="".join(c.MONTH_ABBR[i][0] for i in vm),
                min_years_per_month=int(min(f_n[vm].min(), g_n[vm].min())),
                median_years_per_month=float(np.median(
                    np.minimum(f_n[vm], g_n[vm]))),
                driver_range=float(f_mean[vm].max() - f_mean[vm].min()),
                gw_range_m=float(g_mean[vm].max() - g_mean[vm].min()),
                driver_units=c.SERIES_META[driver]["units"],
                note="",
            ))

            c.write_csv(pd.DataFrame(dict(
                interval=np.arange(res["dA"].size),
                u_lo=res["edges"][:-1], u_hi=res["edges"][1:],
                dA=res["dA"])),
                "02_hysteresis", "hysteresis_dA",
                f"{driver}__GOWN_{well}__{label}.csv")

            # seasonal peak lag from the daily climatology
            fd, fdci, fdv, _ = c.daily_stats(dr)
            gd, gdci, gdv, _ = c.daily_stats(gw)
            dvalid = fdv & gdv
            for smooth in (1, 15):
                pl = _peak_lag(fd, gd, dvalid, smooth)
                if pl:
                    lag_rows.append(dict(driver=driver, well=well,
                                         pair_class=klass, window=label,
                                         start=start, end=end, **pl))

    hi = pd.DataFrame(idx_rows)
    c.write_csv(hi, "02_hysteresis", "hysteresis_index.csv")
    c.write_csv(pd.DataFrame(lag_rows), "02_hysteresis", "seasonal_peak_lag.csv")

    c.write_csv(pd.DataFrame([dict(driver=k, window="-".join(
        c.SERIES_META[k]["notebook_window"]), source=v)
        for k, v in WINDOW_SOURCE.items()]),
        "02_hysteresis", "notebook_windows.csv")

    print(f"  hysteresis: {len(hi)} loop rows, {len(lag_rows)} peak-lag rows")
    return hi
