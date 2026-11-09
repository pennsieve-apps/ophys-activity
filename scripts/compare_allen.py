"""
Compare ophys-activity's tuning metrics with the Allen Brain Observatory's published per-cell
metrics (de Vries et al. 2020) for the demo session (experiment container 511510779, session
503109347, VISp layer 2/3). Development tool, not part of the image; needs network once to fetch
the Allen metrics (api.brain-map.org, ApiCamCellMetric) unless --metrics is given.

Usage:
  python scripts/compare_allen.py <ophys-activity rois.csv> [--metrics allen_cell_metrics.csv] [--out compare.csv]

ROIs are matched by id: the Allen companion NWB's PlaneSegmentation ids are the Allen
cell_specimen_ids. (For our own Suite2p ROIs, match them to Allen's by mask overlap first.)
"""
from __future__ import annotations

import argparse
import json
import urllib.request

import numpy as np
import pandas as pd

API = ("https://api.brain-map.org/api/v2/data/query.json?criteria=model::ApiCamCellMetric,"
       "rma::criteria,[experiment_container_id$eq{}]&num_rows=5000")
PAIRS = [("osi", "osi_dg"), ("dsi", "dsi_dg"), ("gosi", "g_osi_dg"), ("gdsi", "g_dsi_dg"),
         ("reliability_trials", "reliability_dg")]


def fetch(container: int) -> pd.DataFrame:
    with urllib.request.urlopen(API.format(container), timeout=120) as r:
        rows = json.load(r)["msg"]
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rois")
    ap.add_argument("--metrics")
    ap.add_argument("--container", type=int, default=511510779)
    ap.add_argument("--out")
    a = ap.parse_args()
    ours = pd.read_csv(a.rois)
    allen = pd.read_csv(a.metrics) if a.metrics else fetch(a.container)
    m = ours.merge(allen, left_on="roi_id", right_on="cell_specimen_id", how="inner")
    m = m[m["dsi_dg"].notna()]
    rows = []
    for name, sel in (("responsive (ours)", m["responsive"].astype(bool)), ("all matched", np.ones(len(m), bool))):
        s = m[sel]
        r = {"set": name, "n": len(s)}
        for x, y in PAIRS:
            ok = s[x].notna() & s[y].notna()
            r[f"{x}_pearson"] = round(float(np.corrcoef(s.loc[ok, x], s.loc[ok, y])[0, 1]), 3) if ok.sum() > 2 else None
            r[f"{x}_median_ours"] = round(float(s.loc[ok, x].median()), 3) if ok.any() else None
            r[f"{x}_median_allen"] = round(float(s.loc[ok, y].median()), 3) if ok.any() else None
        pcol = next((c for c in s.columns if c.startswith("pref_orientation") or c.startswith("pref_direction")), None)
        if pcol:
            ok = s[pcol].notna() & s["pref_dir_dg"].notna()
            r["pref_dir_exact_match"] = round(float((s.loc[ok, pcol] == s.loc[ok, "pref_dir_dg"]).mean()), 3)
        if "pref_dir_vs" in s:
            d = np.abs((s["pref_dir_vs"] - s["pref_dir_dg"] + 180) % 360 - 180)
            r["pref_dir_vs_median_abs_diff_deg"] = round(float(d.median()), 1)
        tcol = next((c for c in s.columns if c.startswith("pref_temporal_frequency")), None)
        if tcol:
            r["pref_tf_match"] = round(float((s[tcol] == s["pref_tf_dg"]).mean()), 3)
        rows.append(r)
    res = pd.DataFrame(rows)
    print(res.T.to_string(header=False))
    if a.out:
        res.to_csv(a.out, index=False)


if __name__ == "__main__":
    main()
