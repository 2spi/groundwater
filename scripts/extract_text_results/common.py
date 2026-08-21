"""Shared loading, series registry and climatology helpers.

Functions here are lifted from the bow_valley notebooks unchanged so the
extracted numbers reproduce what the figures show.
"""
from pathlib import Path

import numpy as np
import pandas as pd

# numpy alias shim -- notebook 06 cell 1. pycwt still calls np.int internally,
# so this must run before pycwt is imported anywhere.
for _name, _builtin in {"int": int, "float": float, "complex": complex,
                        "bool": bool}.items():
    if not hasattr(np, _name):
        setattr(np, _name, _builtin)

ROOT = Path("/home/py/groundwater")
DATA = ROOT / "data"
BOW_VALLEY = DATA / "bow_valley"
STREAM = BOW_VALLEY / "stream"
XSEC = BOW_VALLEY / "cross_section"
ECCC_DIR = DATA / "eccc" / "processed"
ACIS_DIR = DATA / "acis" / "processed"

OUT = ROOT / "outputs" / "text_results"

MARMOT_WELLS = ["0931", "0301", "0303", "0386", "0305"]
CANMORE_WELLS = ["0764", "0760", "0759", "0364"]
WELLS = MARMOT_WELLS + CANMORE_WELLS

MONTH_ABBR = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
MONTH_DAYS = 30.4375  # 365.25 / 12


def read_series(path, date_col=0, usecols=None, drop_first_col=False, sort=False):
    """Notebook helper, unchanged."""
    df = pd.read_csv(path, parse_dates=[date_col], index_col=date_col, usecols=usecols)
    if drop_first_col:
        df = df.drop(columns=df.columns[0])
    s = df.squeeze()
    return s.sort_index() if sort else s


def _read_eccc(file):
    df = pd.read_csv(ECCC_DIR / file, usecols=[0, 12], parse_dates=[0], index_col=0)
    return df.loc["2005":"2024"].squeeze()


def _read_acis(file):
    df = pd.read_csv(ACIS_DIR / file, usecols=[0, 2], parse_dates=[0], index_col=0)
    return df.loc["2005":"2024"].squeeze()


# --- series registry ----------------------------------------------------
# notebook_window: the (start, end) string pair the 02* notebooks pass to
# .loc[] for this driver. None means the plot_* default of ('2006', '2023').

DRIVER_META = {
    "krns":   dict(name="Kananaskis River near Seebe",
                   station="05BF001", kind="stream", basin="kananaskis",
                   units="m3/s", var="discharge",
                   source="bow_valley/stream/krns_daily.csv",
                   notebook_window=("2006", "2023")),
    "bl":     dict(name="Kananaskis River below Barrier Dam",
                   station="05BF025", kind="stream", basin="kananaskis",
                   units="m3/s", var="discharge",
                   source="bow_valley/stream/05BF025_BarrierLk_TAU_Day_Mean.csv",
                   notebook_window=("2005", "2024")),
    "lkl":    dict(name="Lower Kananaskis Lake",
                   station="05BF009", kind="stream", basin="kananaskis",
                   units="mamsl", var="stage",
                   source="bow_valley/stream/05BF009.csv",
                   notebook_window=("2005", "2017")),
    "jcnc":   dict(name="Jumpingpound Creek near Cochrane",
                   station="05BH013", kind="stream", basin="kananaskis",
                   units="mamsl", var="stage",
                   source="bow_valley/stream/05BH013.csv",
                   notebook_window=("2005", "2024")),
    "brac_e": dict(name="Bow River at Canmore (water-level elevation)",
                   station="CR", kind="stream", basin="bow",
                   units="m", var="stage",
                   source="bow_valley/stream/cr.csv",
                   notebook_window=("2005", "2024")),
    "brab":   dict(name="Bow River at Banff (elevation)",
                   station="05BB001", kind="stream", basin="bow",
                   units="m", var="stage",
                   source="bow_valley/stream/brab_elevation.csv",
                   notebook_window=("2005", "2024")),
    "brac":   dict(name="Bow River at Canmore (flow)",
                   station="CR", kind="stream", basin="bow",
                   units="m3/s", var="discharge",
                   source="bow_valley/stream/cr_flow.csv",
                   notebook_window=("2006", "2023")),
    "banff":  dict(name="Precipitation at Banff CS",
                   station="3050519", kind="climate", basin="bow",
                   units="mm", var="precip",
                   source="eccc/processed/3050519_BANFF_CS.csv",
                   notebook_window=("2005", "2024")),
    "exshaw": dict(name="Precipitation at Bow Valley (Exshaw)",
                   station="ACIS Bow_Valley", kind="climate", basin="bow",
                   units="mm", var="precip",
                   source="acis/processed/Bow_Valley.csv",
                   notebook_window=("2005", "2024")),
    "kv":     dict(name="Precipitation at Kananaskis Boundary Auto",
                   station="ACIS Kananaskis_Boundary_Auto", kind="climate",
                   basin="kananaskis", units="mm", var="precip",
                   source="acis/processed/Kananaskis_Boundary_Auto.csv",
                   notebook_window=("2005", "2024")),
}

DRIVERS = list(DRIVER_META)

WELL_META = {w: dict(name=f"GOWN_{w}", kind="well",
                     basin="marmot" if w in MARMOT_WELLS else "canmore",
                     units="mamsl", var="groundwater level",
                     source="gown/combined_absval_20052024_filled.csv",
                     station=w, notebook_window=("2005", "2024"))
             for w in WELLS}

SERIES_META = {**WELL_META, **DRIVER_META}


def load_gown():
    """Daily gap-filled GOWN levels, columns as zero-padded strings."""
    return pd.read_csv(DATA / "gown" / "combined_absval_20052024_filled.csv",
                       parse_dates=[0], index_col=0)


def load_drivers():
    return {
        "krns":   read_series(STREAM / "krns_daily.csv"),
        "bl":     read_series(STREAM / "05BF025_BarrierLk_TAU_Day_Mean.csv"),
        "lkl":    read_series(STREAM / "05BF009.csv"),
        "jcnc":   read_series(STREAM / "05BH013.csv", sort=True),
        "brac_e": read_series(STREAM / "cr.csv", usecols=[0, 2]),
        "brab":   read_series(STREAM / "brab_elevation.csv", usecols=[0, 2]),
        "brac":   read_series(STREAM / "cr_flow.csv", date_col=1, drop_first_col=True),
        "banff":  _read_eccc("3050519_BANFF_CS.csv"),
        "exshaw": _read_acis("Bow_Valley.csv"),
        "kv":     _read_acis("Kananaskis_Boundary_Auto.csv"),
    }


# dates that appeared more than once in the source file, per series
DUPLICATE_DATES = {}


def load_all():
    """Every series keyed by its registry key: 9 wells + 10 drivers.

    Duplicate dates are collapsed to their mean. 05BH013.csv (jcnc) holds two
    overlapping blocks covering 2012-01-01..2024-10-31 twice, with values that
    differ by up to 22 m on the same day. The notebooks never see this because
    every path they use (resample('D'), groupby(year, month)) averages
    duplicates implicitly; the wavelet transforms need an explicit fix.
    """
    gown = load_gown()
    series = {w: gown[w].dropna() for w in WELLS}
    series.update(load_drivers())
    for k in list(series):
        s = series[k]
        n_dup = int(s.index.duplicated().sum())
        DUPLICATE_DATES[k] = n_dup
        if n_dup:
            s = s.groupby(level=0).mean()
        s = s.sort_index()
        s.name = k
        series[k] = s
    return series


# --- pairing ------------------------------------------------------------
# Pairings the notebooks actually plot: Canmore wells against Bow River
# series, Marmot Creek wells against Kananaskis series. Precipitation is
# paired by which station the notebooks use for which well group.

_PRIMARY = {
    "krns":   MARMOT_WELLS,
    "bl":     MARMOT_WELLS,
    "lkl":    MARMOT_WELLS,
    "jcnc":   MARMOT_WELLS,
    "kv":     MARMOT_WELLS,
    "brac_e": CANMORE_WELLS,
    "brab":   CANMORE_WELLS,
    "brac":   CANMORE_WELLS,
    "banff":  CANMORE_WELLS,
    "exshaw": CANMORE_WELLS,
}


def pair_class(driver, well):
    return "primary" if well in _PRIMARY[driver] else "cross_basin"


def pairs():
    """All 9 x 10 (driver, well, pair_class), primary first."""
    out = [(d, w, "primary") for d in DRIVERS for w in _PRIMARY[d]]
    out += [(d, w, "cross_basin") for d in DRIVERS
            for w in WELLS if w not in _PRIMARY[d]]
    return out


def primary_driver(well, kind="stream"):
    """The driver a well is chiefly compared against, for one-driver summaries."""
    if kind == "stream":
        return "brac_e" if well in CANMORE_WELLS else "krns"
    return "banff" if well in CANMORE_WELLS else "kv"


# --- climatology (02_01_hysteresis_idx.ipynb cell 8, unchanged) ---------

def monthly_stats(series, fill_mean=np.nan):
    """(mean, 95% CI half-width, valid) arrays of length 12, one per month.

    Averages within each (year, month) first, then across years.
    """
    df = series.to_frame("val")
    df["month"] = series.index.month
    df["year"] = series.index.year
    per_year_month = df.groupby(["year", "month"])["val"].mean()
    agg = per_year_month.groupby("month").agg(["mean", "std", "count"])
    agg = agg.reindex(range(1, 13))

    valid = agg["count"].fillna(0).values > 0
    agg["mean"] = agg["mean"].fillna(fill_mean)
    ci = (1.96 * agg["std"] / np.sqrt(agg["count"])).fillna(0)
    return agg["mean"].values, ci.values, valid, agg["count"].fillna(0).values


# start index of each month in a fixed 366-day calendar
_MONTH_START = np.cumsum([0, 31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30])[:12]


def _day_slot(index):
    """Position 0-365 of each timestamp in a fixed 366-day calendar."""
    return _MONTH_START[index.month.values - 1] + index.day.values - 1


def daily_stats(series, fill_mean=np.nan):
    """(mean, 95% CI half-width, valid) arrays of length 366, one per day."""
    df = series.to_frame("val")
    df["day"] = _day_slot(series.index)
    df["year"] = series.index.year
    per_year_day = df.groupby(["year", "day"])["val"].mean()
    agg = per_year_day.groupby("day").agg(["mean", "std", "count"])
    agg = agg.reindex(range(366))

    valid = agg["count"].fillna(0).values > 0
    agg["mean"] = agg["mean"].fillna(fill_mean)
    ci = (1.96 * agg["std"] / np.sqrt(agg["count"])).fillna(0)
    return agg["mean"].values, ci.values, valid, agg["count"].fillna(0).values


def slot_label(slot):
    """'15-Jun' style label for a 0-365 fixed-calendar slot."""
    m = int(np.searchsorted(_MONTH_START, slot, "right")) - 1
    return f"{slot - _MONTH_START[m] + 1:02d}-{MONTH_ABBR[m]}"


def z_norm(s):
    return (s - s.mean()) / s.std()


def overlap(a, b):
    """Dates where both series have a finite value."""
    return a.dropna().index.intersection(b.dropna().index)


def write_csv(df, *parts, index=False):
    path = OUT.joinpath(*parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    # "note" only carries text when a row went wrong; drop it when it never did
    if "note" in df.columns and df["note"].replace("", np.nan).isna().all():
        df = df.drop(columns="note")
    df.to_csv(path, index=index, float_format="%.6g")
    return path


def write_text(text, *parts):
    path = OUT.joinpath(*parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path
