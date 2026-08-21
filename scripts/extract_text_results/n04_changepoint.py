"""BEAST changepoint results from 04_change_point.ipynb.

Same configuration as plot_beast(): monthly means stamped mid-month, harmonic
season with a 1-year period, at most one trend changepoint and ten seasonal
ones, 24-month segment margins.
"""
import sys
import warnings

import numpy as np
import pandas as pd

import common as c

sys.unraisablehook = lambda u: None
warnings.filterwarnings("ignore")

# mcmc_seed is added on top of the notebook's configuration: BEAST samples the
# posterior, so changepoint probabilities drift by a few percent between runs
# unless the chain is seeded. Locations and CIs are stable either way.
BEAST_KW = dict(deltat="1 month", period="1 year", season="harmonic",
                tcp_minmax=[0, 1], scp_minmax=[0, 10],
                print_progress=False, print_options=False, quiet=True,
                ci=True, tseg_leftmargin=24, tseg_rightmargin=24,
                mcmc_seed=42)


def _dec_year_to_date(t):
    if not np.isfinite(t):
        return None
    year = int(t)
    rem = t - year
    start = pd.Timestamp(year=year, month=1, day=1)
    days = 366 if start.is_leap_year else 365
    return (start + pd.Timedelta(days=rem * days)).date()


def _monthly(s, start="2005", end="2024"):
    """Monthly means on the notebook's mid-month stamp."""
    m = s.loc[start:end].resample("MS").mean()
    m.index = m.index + pd.Timedelta(days=14)
    return m


def _fit(y, first):
    """`first` is the timestamp of the first monthly sample.

    The notebook hard-codes start=[2005,1,1] even for series that begin
    earlier (brac_e starts 2002), which shifts their time axis. Here the
    series are sliced to 2005-2024 first and the true start is passed.
    """
    import Rbeast as rb
    return rb.beast(y, start=[first.year, first.month, 1], **BEAST_KW)


def run(series):
    import Rbeast as rb  # noqa: F401  (import cost paid once)

    cp_rows, model_rows, ser_rows = [], [], []

    for key, s in series.items():
        m = _monthly(s)
        if m.isna().any():
            m = m.interpolate(limit_direction="both")
        if len(m) < 60:
            continue
        y = m.values.astype(float)
        o = _fit(y, m.index[0])

        for comp_name, comp in (("trend", o.trend), ("season", o.season)):
            cps = np.atleast_1d(comp.cp)
            prs = np.atleast_1d(comp.cpPr)
            cis = np.atleast_2d(comp.cpCI) if getattr(comp, "cpCI", None) is not None \
                else np.full((len(cps), 2), np.nan)
            amps = np.atleast_1d(getattr(comp, "cpAbruptChange", np.nan))
            for i, (cp, pr) in enumerate(zip(cps, prs)):
                if not np.isfinite(cp):
                    continue
                lo, hi = (cis[i] if i < len(cis) else (np.nan, np.nan))
                cp_rows.append(dict(
                    series=key, name=c.SERIES_META[key]["name"],
                    component=comp_name, rank=i + 1,
                    cp_decimal_year=float(cp), cp_date=_dec_year_to_date(cp),
                    cp_prob=float(pr),
                    cp_ci_lo_year=float(lo), cp_ci_hi_year=float(hi),
                    cp_ci_lo_date=_dec_year_to_date(lo),
                    cp_ci_hi_date=_dec_year_to_date(hi),
                    abrupt_change=float(amps[i]) if i < len(amps) else np.nan,
                    units=c.SERIES_META[key]["units"],
                ))

        def _ncp(comp):
            return {k: float(np.atleast_1d(getattr(comp, k))[0])
                    for k in ("ncp", "ncp_median", "ncp_mode",
                              "ncp_pct10", "ncp_pct90")
                    if getattr(comp, k, None) is not None}

        model_rows.append(dict(
            series=key, name=c.SERIES_META[key]["name"],
            kind=c.SERIES_META[key]["kind"], units=c.SERIES_META[key]["units"],
            n_months=len(y),
            start=m.index.min().date(), end=m.index.max().date(),
            **{f"trend_{k}": v for k, v in _ncp(o.trend).items()},
            **{f"season_{k}": v for k, v in _ncp(o.season).items()},
            # Rbeast's own R2 field does not behave like a coefficient of
            # determination here (it returns values above 1), so R2 is
            # recomputed from the fitted trend + season. Its RMSE is correct.
            R2=float(1 - np.sum((y - (o.trend.Y + o.season.Y)) ** 2) /
                     np.sum((y - y.mean()) ** 2)),
            beast_R2_field=float(np.atleast_1d(o.R2)[0]),
            RMSE=float(np.atleast_1d(o.RMSE)[0]),
            sig2=float(np.atleast_1d(o.sig2).ravel()[0]),
            marg_lik=float(np.atleast_1d(o.marg_lik)[0]),
            season_type=str(o.season_type),
            config="tcp_minmax=[0,1] scp_minmax=[0,10] tseg_margin=24 "
                   "season=harmonic period=1yr deltat=1month",
        ))

        tr, se = o.trend, o.season
        ser_rows.append(pd.DataFrame(dict(
            series=key, date=m.index.date, observed=y,
            trend=tr.Y, trend_ci_lo=tr.CI[:, 0], trend_ci_hi=tr.CI[:, 1],
            trend_sd=tr.SD,
            slope_per_month=tr.slp, slope_ci_lo=tr.slpCI[:, 0],
            slope_ci_hi=tr.slpCI[:, 1],
            slope_pos_prob=tr.slpSgnPosPr, slope_zero_prob=tr.slpSgnZeroPr,
            season=se.Y, season_ci_lo=se.CI[:, 0], season_ci_hi=se.CI[:, 1],
            trend_cp_occ_prob=tr.cpOccPr, season_cp_occ_prob=se.cpOccPr,
        )))

    cps = pd.DataFrame(cp_rows)
    c.write_csv(cps, "04_change_point", "beast_changepoints.csv")
    c.write_csv(pd.DataFrame(model_rows), "04_change_point", "beast_model.csv")
    c.write_csv(pd.concat(ser_rows, ignore_index=True),
                "04_change_point", "beast_series.csv")
    print(f"  changepoint: {len(model_rows)} series, {len(cps)} changepoints")
    return cps
