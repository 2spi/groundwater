"""Assert the extraction reproduces the values stored in the notebooks.

Anchors are the text outputs saved in the .ipynb files themselves, so a
mismatch means the reproduction drifted, not that the notebooks are wrong.
"""
import numpy as np
import pandas as pd

import common as c

# 03_trend.ipynb cell 6 (well table) and cell 4 (rivers)
MK_WELLS = {
    "0303": (-0.325439, -0.044508, 0.004460, "decreasing"),
    "0764": (-0.336842, -0.038566, 0.008247, "decreasing"),
    "0305": (-0.381579, -0.031324, 0.000129, "decreasing"),
    "0759": (-0.214035, -0.016055, 0.039079, "decreasing"),
    "0364": (-0.335965, -0.014041, 0.024932, "decreasing"),
    "0386": (-0.450000, -0.013536, 0.000070, "decreasing"),
    "0301": (-0.276316, -0.012168, 0.015159, "decreasing"),
    "0760": (-0.022807, -0.001375, 0.829248, "no trend"),
    "0931": (0.276316, 0.035785, 0.013236, "increasing"),
}
MK_RIVERS = {
    "brac": (-0.044795, -0.003051, 0.018879, "decreasing"),
    "krns": (-0.003283, -0.000370, 0.928648, "no trend"),
}
# 06_wavelet_transform.ipynb cell 20
LAG1 = {"0931": 0.9983, "0301": 0.9959, "0303": 0.9990, "0386": 0.9962,
        "0305": 0.9902, "0764": 0.9990, "0760": 0.9987, "0759": 0.9950,
        "0364": 0.9978}
# 03_trend.ipynb cell 12
LKL_MEAN = 1668.0250183235046


def _rd(*p):
    return pd.read_csv(c.OUT.joinpath(*p), dtype={"series": str, "well": str})


def run(series):
    L, ok, fail = [], 0, 0

    def check(name, cond, detail=""):
        nonlocal ok, fail
        if cond:
            ok += 1
            L.append(f"  PASS  {name}")
        else:
            fail += 1
            L.append(f"  FAIL  {name}   {detail}")

    L.append("Verification against the values stored in the notebooks")
    L.append("=" * 70)
    L.append("")
    L.append("1. Mann-Kendall, 03_trend.ipynb cells 3-6")

    mk = _rd("03_trend", "mann_kendall.csv")
    nb = mk[mk["source"] == "notebook"].set_index("series")
    for key, (tau, slope, p, verdict) in {**MK_WELLS, **MK_RIVERS}.items():
        r = nb.loc[key]
        check(f"{key}: tau/slope/p/verdict",
              np.isclose(r["tau"], tau, atol=1e-6) and
              np.isclose(r["sen_slope_per_period"], slope, atol=1e-6) and
              np.isclose(r["p"], p, atol=1e-6) and r["verdict"] == verdict,
              f"got tau={r['tau']:.6f} slope={r['sen_slope_per_period']:.6f} "
              f"p={r['p']:.6f} {r['verdict']}, expected {tau} {slope} {p} {verdict}")

    wells = nb.loc[list(MK_WELLS)]
    counts = wells["verdict"].value_counts().to_dict()
    check("verdict counts are decreasing 7 / increasing 1 / no trend 1",
          counts == {"decreasing": 7, "increasing": 1, "no trend": 1}, str(counts))
    check("8 of 9 wells significant at p<0.05",
          int((wells["p"] < 0.05).sum()) == 8,
          str(int((wells['p'] < 0.05).sum())))
    check("lkl.mean() over 2005+ matches cell 12",
          np.isclose(series["lkl"].loc["2005":].mean(), LKL_MEAN, atol=1e-9))

    L.append("")
    L.append("2. Detrended lag-1 autocorrelation, 06_wavelet_transform.ipynb cell 20")
    lag1 = _rd("06_wavelet", "cwt_lag1.csv")
    lag1 = lag1[lag1["window"] == "full_record"].set_index("series")
    for key, want in LAG1.items():
        r = lag1.loc[key]
        check(f"{key}: N=7305 and lag-1={want}",
              int(r["n"]) == 7305 and np.isclose(r["detrended_lag1"], want, atol=5e-5),
              f"got n={int(r['n'])} lag1={r['detrended_lag1']}")

    L.append("")
    L.append("3. BEAST changepoint on GOWN_0931, 04_change_point.ipynb")
    cp = _rd("04_change_point", "beast_changepoints.csv")
    r = cp[(cp["series"] == "0931") & (cp["component"] == "trend")]
    check("GOWN_0931 trend changepoint at 2012.42 with CI 2012.1-2013.4",
          len(r) == 1 and np.isclose(r.iloc[0]["cp_decimal_year"], 2012.418, atol=0.01)
          and 2012.0 < r.iloc[0]["cp_ci_lo_year"] < 2012.3
          and 2013.3 < r.iloc[0]["cp_ci_hi_year"] < 3000,
          r[["cp_decimal_year", "cp_prob", "cp_ci_lo_year",
             "cp_ci_hi_year"]].to_dict("records"))

    L.append("")
    L.append("4. Hysteresis loops, 02_hysteresis.ipynb / 02_01_hysteresis_idx.ipynb")
    hy = _rd("02_hysteresis", "hysteresis_index.csv")
    nbw = hy[hy["window"] == "notebook"]
    full_year = nbw[~nbw["driver"].isin(["jcnc", "kv"])]
    check("every non-seasonal driver gives a full 12-month loop",
          bool((full_year["n_valid_months"] == 12).all()),
          str(sorted(full_year.loc[full_year["n_valid_months"] != 12,
                                   "driver"].unique())))
    seasonal = nbw[nbw["driver"].isin(["jcnc", "kv"])]
    check("the two seasonal stations give partial loops (Apr-Nov)",
          bool((seasonal["n_valid_months"] == 8).all()),
          str(seasonal["n_valid_months"].unique()))
    check("hysteresis class is always one of I-VIII",
          bool(nbw["hyst_class"].isin(list("I II III IV V VI VII VIII".split())).all()))

    L.append("")
    L.append("5. Coverage")
    xc = _rd("01_cross_correlation", "xcorr_summary.csv")
    inv = _rd("00_inventory", "series_inventory.csv")
    check("xcorr_summary has 9 wells x 10 drivers = 90 rows", len(xc) == 90, str(len(xc)))
    check("every xcorr pair found a peak", int(xc["peak_r"].notna().sum()) == 90,
          str(int(xc["peak_r"].notna().sum())))
    check("series_inventory has 19 series", len(inv) == 19, str(len(inv)))
    check("hysteresis covers all 90 pairs in the notebook window",
          nbw.groupby(["driver", "well"]).ngroups == 90,
          str(nbw.groupby(["driver", "well"]).ngroups))
    wc = _rd("06_wavelet", "wct_band_summary.csv")
    check("wavelet coherence covers all 90 pairs",
          wc.groupby(["driver", "well"]).ngroups == 90,
          str(wc.groupby(["driver", "well"]).ngroups))

    empty = [str(p.relative_to(c.OUT)) for p in sorted(c.OUT.rglob("*"))
             if p.is_file() and p.stat().st_size < 50]
    check("no output file is empty", not empty, str(empty))

    L.append("")
    L.append("=" * 70)
    L.append(f"{ok} passed, {fail} failed")
    text = "\n".join(L)
    c.write_text(text, "VERIFY.txt")
    print(f"  verify: {ok} passed, {fail} failed")
    if fail:
        print("\n".join(l for l in L if l.startswith("  FAIL")))
    return fail == 0
