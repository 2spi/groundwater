"""Series inventory, well metadata and pairwise overlap windows."""
import numpy as np
import pandas as pd

import common as c


def series_inventory(series):
    rows = []
    for key, s in series.items():
        meta = c.SERIES_META[key]
        v = s.dropna()
        rows.append(dict(
            key=key, name=meta["name"], station=meta["station"],
            kind=meta["kind"], basin=meta["basin"], units=meta["units"],
            variable=meta["var"], source=meta["source"],
            n_rows=len(s), n_valid=len(v), n_missing=int(s.isna().sum()),
            n_duplicate_dates_collapsed=c.DUPLICATE_DATES.get(key, 0),
            first=s.index.min().date(), last=s.index.max().date(),
            first_valid=v.index.min().date(), last_valid=v.index.max().date(),
            span_years=round((s.index.max() - s.index.min()).days / 365.25, 2),
            inferred_freq=pd.infer_freq(s.index[:30]),
            mean=v.mean(), std=v.std(), min=v.min(), p25=v.quantile(.25),
            median=v.median(), p75=v.quantile(.75), max=v.max(),
        ))
    return pd.DataFrame(rows)


def well_metadata():
    collars = pd.read_csv(c.XSEC / "collars.csv", dtype={"code": str})
    comps = pd.read_csv(c.XSEC / "completions.csv", dtype={"code": str})

    collars = collars[collars["code"].isin(c.WELLS)].set_index("code")
    keep = ["station_no", "gic_well_id", "latitude", "longitude", "mp_elev_m",
            "depth_str", "aquifer", "aquifer_lithology", "aquifer_type",
            "production_str", "drill_date", "well_depth_m", "td_m",
            "awwid_elev_m", "elev_discrepancy_m", "type_of_work", "well_use",
            "drilling_method", "casing_material", "casing_od_in",
            "casing_bottom_m", "dem_ground_m", "x_utm11n", "y_utm11n"]
    collars = collars[[k for k in keep if k in collars.columns]]

    # screen intervals: one row per well, preferring the AWWID screen record
    scr = comps[comps["code"].isin(c.WELLS)].copy()
    scr["is_screen"] = scr["type"].str.contains("screen", case=False, na=False)
    first = scr.drop_duplicates(subset="code", keep="first").set_index("code")
    screen = (scr[scr["is_screen"]].drop_duplicates(subset="code", keep="first")
              .set_index("code"))

    out = collars.copy()
    for src, tag in [(first, "notebook"), (screen, "screen")]:
        for col in ["type", "source", "from_m", "to_m", "top_elev_m",
                    "base_elev_m", "detail"]:
            out[f"{tag}_{col}"] = src[col] if col in src.columns else np.nan

    out["basin"] = ["marmot" if w in c.MARMOT_WELLS else "canmore"
                    for w in out.index]
    out["n_completion_records"] = scr.groupby("code").size()
    return out.reset_index().rename(columns={"code": "well"})


def overlap_matrix(series):
    rows = []
    for driver, well, klass in c.pairs():
        idx = c.overlap(series[driver], series[well])
        rows.append(dict(
            driver=driver, well=well, pair_class=klass,
            n_overlap_days=len(idx),
            start=idx.min().date() if len(idx) else None,
            end=idx.max().date() if len(idx) else None,
            overlap_years=round(len(idx) / 365.25, 2),
        ))
    return pd.DataFrame(rows)


def run(series):
    inv = series_inventory(series)
    c.write_csv(inv, "00_inventory", "series_inventory.csv")
    c.write_csv(well_metadata(), "00_inventory", "well_metadata.csv")
    ov = overlap_matrix(series)
    c.write_csv(ov, "00_inventory", "overlap_matrix.csv")
    print(f"  inventory: {len(inv)} series, {len(ov)} pairs")
    return inv, ov
