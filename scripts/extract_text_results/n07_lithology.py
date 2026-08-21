"""Borehole lithology and completion tables from cross_section.ipynb."""
import numpy as np
import pandas as pd

import common as c

BEDROCK = {"Bedrock", "Shale", "Sandstone & Gravel", "Limestone", "Coal",
           "Rocks", "Shale & Gravel"}


def _load():
    lith = pd.read_csv(c.XSEC / "lithology_intervals.csv", index_col=0,
                       dtype={"code": str})
    lith["code"] = lith["code"].str.zfill(4)
    lith = lith[lith["code"].isin(c.WELLS)].copy()

    comps = pd.read_csv(c.XSEC / "completions.csv", dtype={"code": str})
    comps = comps[comps["code"].isin(c.WELLS)].copy()
    comps["used_by_notebook"] = ~comps.duplicated(subset="code", keep="first")

    collars = pd.read_csv(c.XSEC / "collars.csv", dtype={"code": str})
    collars = collars[collars["code"].isin(c.WELLS)].set_index("code")
    return lith, comps, collars


def _screen(comps, well):
    """The interval the notebook draws: first completion record per well."""
    rows = comps[(comps["code"] == well) & comps["used_by_notebook"]]
    if rows.empty:
        return None
    return rows.iloc[0]


def summary(lith, comps, collars):
    rows = []
    for well in c.WELLS:
        log = lith[lith["code"] == well].sort_values("from_m")
        scr = _screen(comps, well)
        col = collars.loc[well]

        at_screen = ""
        if scr is not None and not log.empty:
            mid = (scr["from_m"] + scr["to_m"]) / 2
            hit = log[(log["from_m"] <= mid) & (log["to_m"] >= mid)]
            at_screen = hit["material"].iloc[0] if not hit.empty else ""

        bedrock_top = np.nan
        for _, r in log.iterrows():
            if r["material"] in BEDROCK:
                bedrock_top = r["from_m"]
                break

        thick = log.groupby("material")["thickness_m"].sum()
        rows.append(dict(
            well=well, basin=c.SERIES_META[well]["basin"],
            station_no=col.get("station_no"),
            n_intervals=len(log),
            logged_depth_m=log["to_m"].max() if not log.empty else np.nan,
            well_depth_m=col.get("well_depth_m"), td_m=col.get("td_m"),
            mp_elev_m=col.get("mp_elev_m"),
            aquifer=col.get("aquifer"),
            aquifer_lithology=col.get("aquifer_lithology"),
            aquifer_type=col.get("aquifer_type"),
            screen_from_m=scr["from_m"] if scr is not None else np.nan,
            screen_to_m=scr["to_m"] if scr is not None else np.nan,
            screen_top_elev_m=scr["top_elev_m"] if scr is not None else np.nan,
            screen_base_elev_m=scr["base_elev_m"] if scr is not None else np.nan,
            screen_source=scr["source"] if scr is not None else "",
            material_at_screen=at_screen,
            first_bedrock_top_m=bedrock_top,
            overburden_thickness_m=bedrock_top,
            dominant_material=thick.idxmax() if len(thick) else "",
            dominant_material_thickness_m=thick.max() if len(thick) else np.nan,
            materials="; ".join(f"{m} {t:.1f}m" for m, t in
                                thick.sort_values(ascending=False).items()),
        ))
    return pd.DataFrame(rows)


def strip_logs(lith, comps, collars):
    out = []
    for well in c.WELLS:
        log = lith[lith["code"] == well].sort_values("from_m")
        scr = _screen(comps, well)
        col = collars.loc[well]

        out.append("=" * 78)
        out.append(f"GOWN_{well}   ({col.get('station_no')})   "
                   f"{c.SERIES_META[well]['basin']} basin")
        out.append("=" * 78)
        out.append(f"  location        : {col.get('latitude'):.5f}, "
                   f"{col.get('longitude'):.5f}")
        out.append(f"  MP elevation    : {col.get('mp_elev_m')} m")
        out.append(f"  total depth     : {col.get('well_depth_m')} m "
                   f"(TD {col.get('td_m')} m)")
        out.append(f"  aquifer         : {col.get('aquifer')} / "
                   f"{col.get('aquifer_lithology')} ({col.get('aquifer_type')})")
        out.append(f"  production zone : {col.get('production_str')}")
        out.append(f"  well use        : {col.get('well_use')}    "
                   f"drilled {str(col.get('drill_date'))[:10]}")
        if scr is not None:
            detail = scr["detail"] if isinstance(scr["detail"], str) else ""
            out.append(f"  screen          : {scr['from_m']}-{scr['to_m']} m "
                       f"({scr['base_elev_m']}-{scr['top_elev_m']} mamsl)  "
                       f"[{scr['type']}, {scr['source']}]  {detail}".rstrip())
        out.append("")
        out.append(f"  {'from':>7} {'to':>7} {'thick':>7}  "
                   f"{'top el':>8} {'base el':>8}  material / description")
        out.append("  " + "-" * 74)
        for _, r in log.iterrows():
            mark = ""
            if scr is not None and r["from_m"] < scr["to_m"] and r["to_m"] > scr["from_m"]:
                mark = "  <== SCREEN"
            desc = " / ".join(str(x) for x in (r["description"], r["colour"])
                              if isinstance(x, str) and x)
            out.append(f"  {r['from_m']:7.2f} {r['to_m']:7.2f} "
                       f"{r['thickness_m']:7.2f}  {r['top_elev_m']:8.2f} "
                       f"{r['base_elev_m']:8.2f}  {r['material']}"
                       f"{' - ' + desc if desc else ''}{mark}")
        out.append("")
    return "\n".join(out)


def run():
    lith, comps, collars = _load()
    c.write_csv(lith.rename(columns={"code": "well"}), "07_lithology",
                "well_lithology_intervals.csv")
    c.write_csv(comps.rename(columns={"code": "well"}), "07_lithology",
                "well_completions.csv")
    s = summary(lith, comps, collars)
    c.write_csv(s, "07_lithology", "lithology_summary.csv")
    c.write_text(strip_logs(lith, comps, collars), "07_lithology", "strip_logs.txt")
    print(f"  lithology: {len(lith)} intervals across {len(s)} wells")
    return s
