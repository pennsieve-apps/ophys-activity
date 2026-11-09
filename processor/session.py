"""One session: read, analyse, write tables and figures."""
from __future__ import annotations

import logging
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from . import analysis as an
from . import figures as fg
from .assemblies import detect_assemblies
from .inputs import Session
from .nwbread import SKIP_TABLES, list_time_intervals, read_masks, read_time_intervals, read_traces, session_meta
from .raster import rastermap_sort

log = logging.getLogger("ophys-activity")

SIGNAL_LABEL = {"deconvolved": "deconvolved activity", "dff": "dF/F", "fluorescence": "fluorescence",
                "events": "event amplitudes"}
SHEET_SKIP = {"path", "session_id"}


# ------------------------------------------------------------------ stimulus table selection

def _csv_table(path: Path, sid: str) -> pd.DataFrame:
    sep = "\t" if path.suffix.lower() == ".tsv" else ","
    df = pd.read_csv(path, sep=sep)
    df.columns = [c.strip() for c in df.columns]
    if "session_id" in df.columns:
        df = df[df["session_id"].astype(str) == sid].drop(columns=["session_id"])
    if "stop_time" not in df.columns:
        df["stop_time"] = np.nan
    return df.reset_index(drop=True)


def select_stimulus(sess: Session, params: dict) -> tuple[pd.DataFrame | None, dict, list[str]]:
    """The stimulus / trials table to align to: (table, info, warnings)."""
    want = params["STIM_TABLE"]
    notes: list[str] = []
    if want.lower() == "none":
        return None, {"stim_table": None, "reason": "STIM_TABLE=none"}, notes
    named = want.lower() != "auto"
    cands = []   # (priority, name, source, loader)
    for prio, files in ((0, sess.stim_nwb), (2, [sess.traces])):
        for p in files:
            try:
                with h5py.File(p, "r") as f:
                    tabs = list_time_intervals(f, anywhere=named)
            except OSError as exc:
                notes.append(f"cannot read {p.name}: {exc}")
                continue
            for name, (path, n) in tabs.items():
                cands.append((prio, name, f"{p.name}:{path}", n, ("nwb", p, path)))
    for p in sess.stim_csv:
        try:
            n = len(_csv_table(p, sess.session_id))
        except (OSError, ValueError, pd.errors.ParserError) as exc:
            notes.append(f"cannot read {p.name}: {exc}")
            continue
        name = p.stem
        for suf in (sess.session_id + "_", sess.session_id):
            name = name.replace(suf, "")
        cands.append((1, name or "stimulus", p.name, n, ("csv", p, None)))
    if named:
        hits = [c for c in cands if c[1] == want or c[2].endswith("/" + want) or c[1] == f"{want}"]
        if not hits:
            avail = ", ".join(sorted({c[1] for c in cands})) or "none"
            notes.append(f"STIM_TABLE={want} not found (tables: {avail}); spontaneous analysis only")
            return None, {"stim_table": None, "available": sorted({c[1] for c in cands})}, notes
        chosen = sorted(hits, key=lambda c: c[0])[0]
    else:
        usable = [c for c in cands if c[1] not in SKIP_TABLES and "spontaneous" not in c[1].lower()
                  and c[3] >= params["MIN_TRIALS"]]
        if not usable:
            return None, {"stim_table": None, "available": sorted({c[1] for c in cands})}, notes

        def score(c):
            df = load_table(c[4], sess.session_id)
            has_cond = an.detect_conditions(df, "auto", params["MAX_LEVELS"]).primary is not None
            return (c[0], c[1] != "trials", not has_cond, -c[3])

        chosen = sorted(usable, key=score)[0]
        others = [c[1] for c in usable if c is not chosen]
        if others:
            notes.append(f"stimulus table '{chosen[1]}' chosen; also present: {', '.join(others)} (set STIM_TABLE to pick)")
    df = load_table(chosen[4], sess.session_id)
    df = df[np.isfinite(df["start_time"].astype(float))].sort_values("start_time").reset_index(drop=True)
    return df, {"stim_table": chosen[1], "stim_source": chosen[2], "available": sorted({c[1] for c in cands})}, notes


def load_table(src, sid: str) -> pd.DataFrame:
    kind, p, path = src
    if kind == "csv":
        return _csv_table(p, sid)
    with h5py.File(p, "r") as f:
        return read_time_intervals(f, path)


# ------------------------------------------------------------------ the session

def run_session(sess: Session, params: dict, out_root: Path, in_dir: Path) -> dict:
    sid = sess.session_id
    out = out_root / sid
    out.mkdir(parents=True, exist_ok=True)
    rel = lambda p: f"{sid}/{p.name}"  # noqa: E731
    warnings: list[str] = []
    figures: list[dict] = []
    tables: list[dict] = []
    rng = np.random.default_rng(params["SEED"])

    tr = read_traces(sess.traces, params["SIGNAL"], params["SERIES"], params["T_START"], params["T_STOP"],
                     params["ACCEPTED_ONLY"])
    X, t = tr.data, tr.t
    R = X.shape[1]
    signal = SIGNAL_LABEL[tr.series.kind]
    log.info("%s: %s, %d frames x %d ROIs (%s; %d of %d dropped), %.2f Hz, %.1f-%.1f s", sid, tr.series.path,
             X.shape[0], R, tr.selection, tr.n_rois_dropped, tr.n_rois_total, tr.frame_rate, t[0], t[-1])
    meta = session_meta(sess.traces)
    sheet = {k: v for k, v in sess.sheet.items() if k not in SHEET_SKIP}

    rois = pd.DataFrame({"session_id": sid, "roi_id": tr.roi_ids, "roi_row": tr.roi_rows})
    for k in ("subject_id", "condition", "group"):
        rois[k] = sheet.get(k, "")
    # ---------------- activity over the whole window
    act = an.activity_summary(X, t, params["EVENT_Z"])
    log.info("%s: event rates done (median %.3f events/s)", sid, float(act["event_rate_hz"].median()))
    rois = pd.concat([rois, act], axis=1)
    Xb, tb = an.bin_activity(X, t, params["BIN_S"])
    Zb = an.zscore(Xb)
    log.info("%s: Rastermap on %d bins x %d ROIs", sid, Zb.shape[0], R)
    isort, rm_settings = rastermap_sort(Zb, params["SEED"])
    rank = np.empty(R, dtype=np.int64)
    rank[isort] = np.arange(R)
    rois["rastermap_rank"] = rank
    C_act = an.corr_matrix(Zb)
    asm = detect_assemblies(Zb, params["SEED"], params["ASSEMBLY_NULL"])
    log.info("%s: activity done (Rastermap, correlations, %d assemblies)", sid, asm["n"])

    summary = {
        "session_id": sid, **{k: v for k, v in sheet.items()},
        "traces": sess.traces.name, "series": tr.series.path, "signal": tr.series.kind, "unit": tr.unit,
        "roi_selection": tr.selection, "n_rois_total": tr.n_rois_total, "n_rois": R,
        "frame_rate_hz": round(tr.frame_rate, 4), "time_source": tr.time_source,
        "t_start": float(t[0]), "t_stop": float(t[-1]), "duration_s": round(float(t[-1] - t[0]), 2),
        "bin_s_for_population": round(float(np.median(np.diff(tb))), 4) if len(tb) > 1 else None,
        "mean_event_rate_hz": float(act["event_rate_hz"].mean()),
        "median_event_rate_hz": float(act["event_rate_hz"].median()),
        "mean_activity_correlation": an.mean_offdiag(C_act),
        "n_assemblies": asm["n"],
        "assembly_sizes": [int(len(m)) for m in asm["members"]],
        "rastermap": rm_settings,
        "nwb": meta,
    }

    # ---------------- stimulus
    stim, stim_info, notes = select_stimulus(sess, params)
    warnings += notes
    summary.update(stim_info)
    tuned_mode = None
    pref_angle = np.full(R, np.nan)
    tuned_mask = np.zeros(R, bool)
    corr_mats = []
    if stim is not None:
        res = analyse_stimulus(sid, stim, X, t, tr, params, rng, out, warnings)
        if res is not None:
            rois = pd.concat([rois, res["rois"]], axis=1)
            summary.update(res["summary"])
            figures += res["figures"]
            tables += res["tables"]
            tuned_mode = res["mode"]
            pref_angle = res["pref_angle"]
            tuned_mask = res["tuned_mask"]
            corr_mats = res["corr"]
    else:
        summary["analysis"] = "spontaneous (no stimulus table)"

    # ---------------- figures common to both paths
    trials_bar = None
    if stim is not None and summary.get("n_trials"):
        trials_bar = summary.pop("_trials_bar", None)
    summary.pop("_trials_bar", None)
    p = out / f"{sid}_rastermap.png"
    cap = fg.raster(p, sid, Zb[:, isort].T, tb, trials_bar, signal)
    figures.insert(0, {"path": rel(p), "caption": cap})

    masks, shape, img = read_masks(sess.traces, tr.plane, tr.roi_rows)
    if masks:
        cx = [float(np.average(m[1], weights=np.abs(m[2]) + 1e-12)) if len(m[0]) else np.nan for m in masks]
        cy = [float(np.average(m[0], weights=np.abs(m[2]) + 1e-12)) if len(m[0]) else np.nan for m in masks]
        rois["x"], rois["y"] = cx, cy
        image, image_name = img
        p = out / f"{sid}_roi_map.png"
        if tuned_mode in ("direction", "orientation"):
            cap = fg.roi_map(p, sid, masks, shape, image, pref_angle, tuned_mask, tuned_mode, image_name)
        else:
            cap = fg.roi_map(p, sid, masks, shape, image, act["event_rate_hz"].to_numpy(),
                             np.ones(R, bool), "event rate (events/s)", image_name)
        figures.append({"path": rel(p), "caption": cap})
    else:
        warnings.append("no ROI masks in the NWB: no ROI map")

    corr_mats = corr_mats + [("activity correlation", C_act)]
    if tuned_mode in ("direction", "orientation") and np.isfinite(pref_angle).any():
        order = np.lexsort((rank, np.where(tuned_mask, 0, 1), np.nan_to_num(pref_angle, nan=999)))
        order_label = "preferred direction" if tuned_mode == "direction" else "preferred orientation"
    else:
        order, order_label = isort, "Rastermap order"
    p = out / f"{sid}_correlations.png"
    figures.append({"path": rel(p), "caption": fg.correlations(p, sid, corr_mats, order, order_label)})
    np.savez_compressed(out / f"{sid}_correlations.npz", roi_ids=tr.roi_ids,
                        **{name.split()[0]: C for name, C in corr_mats})
    tables.append({"path": f"{sid}/{sid}_correlations.npz",
                   "description": "ROI x ROI correlation matrices (" + ", ".join(n for n, _ in corr_mats) + "), roi_ids."})

    # assemblies
    if asm["n"]:
        p = out / f"{sid}_assemblies.png"
        figures.append({"path": rel(p), "caption": fg.assemblies(p, sid, asm, tb, isort)})
        rows = []
        for k in range(asm["n"]):
            mem = set(asm["members"][k].tolist())
            for j in range(R):
                rows.append({"session_id": sid, "assembly": k + 1, "roi_id": tr.roi_ids[j],
                             "weight": float(asm["patterns"][k, j]), "member": j in mem})
        pd.DataFrame(rows).to_csv(out / f"{sid}_assemblies.csv", index=False, float_format="%.6g")
        np.savez_compressed(out / f"{sid}_assemblies.npz", patterns=asm["patterns"], activation=asm["activation"],
                            t=tb, roi_ids=tr.roi_ids, eigenvalues=asm["eigenvalues"], threshold=asm["threshold"],
                            mp_threshold=asm["mp_threshold"])
        tables.append({"path": f"{sid}/{sid}_assemblies.csv",
                       "description": "Assembly weights per ROI (session_id, assembly, roi_id, weight, member)."})
        rois["assemblies"] = [";".join(str(k + 1) for k in range(asm["n"]) if j in set(asm["members"][k].tolist()))
                              for j in range(R)]
    else:
        rois["assemblies"] = ""
    summary["assembly_null"] = asm["null"]
    summary["n_single_roi_components_dropped"] = asm.get("n_single", 0)
    summary["assembly_eigenvalue_threshold"] = asm["threshold"]
    summary["marchenko_pastur_threshold"] = asm["mp_threshold"]

    rois.to_csv(out / f"{sid}_rois.csv", index=False, float_format="%.6g")
    tables.insert(0, {"path": f"{sid}/{sid}_rois.csv", "description": "One row per analysed ROI."})
    summary["warnings"] = warnings
    return {"summary": summary, "figures": figures, "tables": tables, "rois": rois, "warnings": warnings}


def analyse_stimulus(sid, stim: pd.DataFrame, X, t, tr, params, rng, out: Path, warnings: list[str]) -> dict | None:
    R = X.shape[1]
    rel = lambda p: f"{sid}/{p.name}"  # noqa: E731
    conds = an.detect_conditions(stim, params["CONDITION"], params["MAX_LEVELS"])
    warnings += conds.notes
    onsets = stim["start_time"].to_numpy(dtype=float)
    dur = (stim["stop_time"].astype(float) - stim["start_time"].astype(float)).to_numpy()
    dur_ok = dur[np.isfinite(dur) & (dur > 0)]
    response_s = params["RESPONSE_S"] or (float(np.median(dur_ok)) if len(dur_ok) else 1.0)
    response_s = max(response_s, 2.0 / tr.frame_rate)
    al = an.align(X, t, onsets, params["BASELINE_S"], response_s, params["POST_S"], tr.frame_rate)
    log.info("%s: %d of %d trials aligned (baseline %.2f s, response %.2f s); conditions: %s (%s)", sid,
             len(al.keep), len(stim), params["BASELINE_S"], response_s,
             ", ".join([conds.primary, *conds.secondary]) if conds.primary else "none", conds.kind)
    n_kept = len(al.keep)
    if n_kept < params["MIN_TRIALS"]:
        warnings.append(f"only {n_kept} trials of '{len(stim)}' fall inside the imaging window "
                        f"({t[0]:.1f}-{t[-1]:.1f} s); spontaneous analysis only")
        return None
    if n_kept < len(stim):
        warnings.append(f"{len(stim) - n_kept} of {len(stim)} trials are (with their baseline and response windows) "
                        f"outside the analysed time window {t[0]:.1f}-{t[-1]:.1f} s and were not analysed")
    cond = conds.cond[al.keep]
    blank = conds.blank[al.keep]
    C = len(conds.labels)
    r = al.r
    M, S, ncond = an.condition_stats(r, cond, C)
    # Conditions with too few trials would win "preferred" by noise alone: leave them out.
    present = ncond[ncond > 0]
    min_n = max(2, int(np.ceil(0.5 * np.median(present)))) if len(present) else 2
    few = np.flatnonzero((ncond > 0) & (ncond < min_n))
    if len(few):
        warnings.append(f"{len(few)} of {int((ncond > 0).sum())} conditions have fewer than {min_n} trials "
                        f"({int(ncond[few].sum())} trials) and were left out of the analysis")
        cond = np.where(np.isin(cond, few), -1, cond)
        M, S, ncond = an.condition_stats(r, cond, C)
    valid = cond >= 0
    if valid.sum() < params["MIN_TRIALS"]:
        warnings.append("fewer than MIN_TRIALS non-blank trials; spontaneous analysis only")
        return None
    onset_frames = np.clip(np.searchsorted(t, onsets[al.keep]), 0, len(t) - 1)
    groups = [np.flatnonzero(cond == c) for c in range(C)]
    for f in [conds.primary, *conds.secondary] if conds.primary and C > 1 else []:
        fv = stim[f].to_numpy()[al.keep]
        for lv in pd.unique(fv[valid]):
            g = np.flatnonzero(valid & (fv == lv))
            if len(g) > (ncond[ncond > 0].max() if (ncond > 0).any() else 0):
                groups.append(g)   # a factor level pooled over the other factors
    stat, p = an.random_time_test(X, tr.frame_rate, params["BASELINE_S"], response_s, onset_frames, groups,
                                  params["N_SHUFFLES"], rng)
    q = an.bh_fdr(p)
    log.info("%s: responsiveness test done (%d shuffles): %d of %d ROIs with q < %g", sid, params["N_SHUFFLES"],
             int(((q < params["FDR"]) & (stat > 0)).sum()), R, params["FDR"])
    responsive = (q < params["FDR"]) & (stat > 0)
    Mm = np.where(np.isfinite(M), M, -np.inf)
    pref = np.argmax(Mm, axis=0)
    peak = M[pref, np.arange(R)]
    kw_p = an.kruskal_selectivity(r, cond, C)
    kw_q = an.bh_fdr(kw_p)
    tuned = responsive & (kw_q < params["FDR"])
    df = pd.DataFrame({"responsive": responsive, "response_stat": stat, "p_responsive": p, "q_responsive": q,
                       "pref_condition": [conds.labels[k] for k in pref], "peak_response": peak,
                       "kw_p": kw_p, "kw_q": kw_q, "tuned": tuned})
    for f in [conds.primary, *conds.secondary] if conds.primary else []:
        df[f"pref_{f}"] = conds.values[f].to_numpy()[pref]
    # Blank-sweep criterion (as in de Vries et al. 2020): a trial at the preferred condition is
    # significant when its response exceeds the 95th percentile of the blank-trial responses.
    n_blank = int(blank.sum())
    if n_blank >= 10:
        thr = np.percentile(r[blank], 95, axis=0)
        frac = np.array([np.mean(r[cond == pref[j], j] > thr[j]) for j in range(R)])
        df["frac_pref_trials_above_blank"] = frac
        df["responsive_blank"] = frac >= 0.25
        df["blank_mean"] = r[blank].mean(axis=0)
    df["reliability_split_half"] = an.split_half_reliability(r, cond, C)
    df["reliability_trials"] = an.trial_reliability(al.traces, cond, pref)
    df["lifetime_sparseness"] = an.sparseness(M[ncond > 0], axis=0)
    pop_sparse = an.sparseness(M[ncond > 0], axis=1)

    # ---------------- tuning
    mode = conds.kind
    pref_angle = np.full(R, np.nan)
    tuning_rows = []
    fac_levels = {}
    for f in [conds.primary, *conds.secondary] if conds.primary else []:
        vals = stim[f].to_numpy()[al.keep]
        if mode in ("direction", "orientation") and f == conds.primary:
            vals = vals.astype(float) % 360.0
        levels = [lv for lv in conds.values[f].drop_duplicates().tolist()]
        try:
            levels = sorted(levels)
        except TypeError:
            levels = sorted(levels, key=str)
        fac_levels[f] = levels
        for lv in levels:
            idx = valid & (vals == lv)
            n = int(idx.sum())
            if not n:
                continue
            mu = r[idx].mean(axis=0)
            se = r[idx].std(axis=0, ddof=1) / np.sqrt(n) if n > 1 else np.full(R, np.nan)
            for j in range(R):
                tuning_rows.append((sid, tr.roi_ids[j], f, an._fmt_level(lv), "marginal", mu[j], se[j], n))
    curve_at_pref = None
    if mode in ("direction", "orientation"):
        angles = np.array(fac_levels[conds.primary], dtype=float)
        vals = conds.values
        lut = {}
        for c in range(C):
            key = (float(vals[conds.primary].iloc[c]),) + tuple(vals[s].iloc[c] for s in conds.secondary)
            lut[key] = c
        curve = np.full((len(angles), R), np.nan)
        curve_se = np.full((len(angles), R), np.nan)
        curve_n = np.zeros((len(angles), R), dtype=np.int64)
        pref_ang_idx = np.full(R, -1)
        for j in range(R):
            sec = tuple(vals[s].iloc[pref[j]] for s in conds.secondary)
            for a, ang in enumerate(angles):
                c = lut.get((ang,) + sec)
                if c is not None and ncond[c]:
                    curve[a, j], curve_se[a, j], curve_n[a, j] = M[c, j], S[c, j], ncond[c]
            pref_ang_idx[j] = int(np.argmin(np.abs(angles - float(vals[conds.primary].iloc[pref[j]]))))
        # Where a condition at the preferred value of the other factors is missing (too few trials,
        # e.g. in a short recording), use the curve pooled over the other factors instead.
        trial_ang = stim[conds.primary].to_numpy(dtype=float)[al.keep] % 360.0
        marg = np.full((len(angles), R), np.nan)
        marg_se = np.full((len(angles), R), np.nan)
        marg_n = np.zeros(len(angles), dtype=np.int64)
        for a, ang in enumerate(angles):
            idx = valid & (trial_ang == ang)
            marg_n[a] = idx.sum()
            if marg_n[a]:
                marg[a] = r[idx].mean(axis=0)
                if marg_n[a] > 1:
                    marg_se[a] = r[idx].std(axis=0, ddof=1) / np.sqrt(marg_n[a])
        incomplete = ~np.isfinite(curve).all(axis=0)
        curve[:, incomplete] = marg[:, incomplete]
        curve_se[:, incomplete] = marg_se[:, incomplete]
        curve_n[:, incomplete] = marg_n[:, None]
        pref_ang_idx[incomplete] = np.nanargmax(np.where(np.isfinite(marg[:, incomplete]), marg[:, incomplete],
                                                         -np.inf), axis=0)
        curve_kind = np.where(incomplete, "marginal", "at_preferred" if conds.secondary else "marginal")
        df["tuning_curve"] = curve_kind
        if incomplete.any() and conds.secondary:
            warnings.append(f"{int(incomplete.sum())} ROIs: tuning over {conds.primary} pooled over "
                            f"{', '.join(conds.secondary)} (a condition at their preferred value had too few trials)")
        tun = an.angular_tuning(curve, angles, pref_ang_idx, mode)
        for k in ("osi", "dsi", "gosi", "gdsi", "pref_dir_vs", "pref_ori_vs"):
            df[k] = tun[k]
        pref_angle = tun["pref_dir_vs"] if mode == "direction" else tun["pref_ori_vs"]
        curve_at_pref = (angles, curve, curve_se)
        if conds.secondary:
            for j in np.flatnonzero(~incomplete):
                for a, ang in enumerate(angles):
                    if np.isfinite(curve[a, j]):
                        tuning_rows.append((sid, tr.roi_ids[j], conds.primary, an._fmt_level(ang), "at_preferred",
                                            curve[a, j], curve_se[a, j], int(curve_n[a, j])))
    elif mode == "categorical":
        f = conds.primary
        mrows = [(row[1], row[3], row[5]) for row in tuning_rows if row[2] == f]
        if mrows:
            piv = pd.DataFrame(mrows, columns=["roi", "level", "mean"]).pivot_table(index="level", columns="roi",
                                                                                   values="mean", sort=False)
            df["selectivity_index"] = an.categorical_selectivity(piv[list(tr.roi_ids)].to_numpy())
        else:
            df["selectivity_index"] = np.nan
    tuning = pd.DataFrame(tuning_rows, columns=["session_id", "roi_id", "factor", "level", "kind", "mean_response",
                                                "sem", "n_trials"])
    files_tables = []
    if len(tuning):
        tuning.to_csv(out / f"{sid}_tuning.csv", index=False, float_format="%.6g")
        files_tables.append({"path": f"{sid}/{sid}_tuning.csv", "description": "Tuning curves, long format: per ROI, "
                             "factor and level the mean baseline-subtracted response, SEM and trial count; kind "
                             "'marginal' (over all other factors) or 'at_preferred' (at the ROI's preferred value of "
                             "the other factors)."})

    # ---------------- correlations
    ok_c = ncond >= 1
    corr = []
    if ok_c.sum() >= 3:
        corr.append(("signal correlation", an.corr_matrix(M[ok_c])))
    E = an.noise_residuals(r, cond, C)
    if len(E) >= 10:
        corr.append(("noise correlation", an.corr_matrix(E)))
    mask = responsive if responsive.sum() >= 2 else None

    # ---------------- tables
    trials = stim.iloc[al.keep].copy()
    trials.insert(0, "trial_index", al.keep)
    trials["condition_label"] = [conds.labels[c] if c >= 0 else "blank" for c in cond]
    trials["blank"] = blank
    trials.to_csv(out / f"{sid}_trials.csv", index=False, float_format="%.6g")
    extra = {}
    if al.traces is not None:
        extra["aligned"] = al.traces
    else:
        warnings.append("aligned traces too large to store; responses.npz has the response matrix only")
    np.savez_compressed(out / f"{sid}_responses.npz", responses=r.astype(np.float32), baseline=al.baseline,
                        t=al.t.astype(np.float32), roi_ids=tr.roi_ids, trial_index=al.keep,
                        start_time=onsets[al.keep], condition=np.array([conds.labels[c] if c >= 0 else "blank" for c in cond]),
                        condition_index=cond, blank=blank, response_window_s=np.array([0.0, al.response_s]),
                        baseline_window_s=np.array([-al.baseline_s, 0.0]), signal=np.array(tr.series.kind), **extra)
    files_tables += [
        {"path": f"{sid}/{sid}_trials.csv", "description": "The analysed trials: the stimulus table's columns plus "
         "trial_index (row in the table), condition_label and blank."},
        {"path": f"{sid}/{sid}_responses.npz", "description": "responses (trials x ROIs, mean over the response window "
         "minus the pre-stimulus baseline: the matrix for decoding), baseline, aligned (trials x ROIs x time, "
         "baseline-subtracted), t (s from onset), roi_ids, trial_index, start_time, condition, condition_index, blank."},
    ]

    # ---------------- figures
    figs = []
    order_pref = np.argsort(np.nan_to_num(pref_angle, nan=1e9)) if mode in ("direction", "orientation") else np.argsort(pref)
    resp_order = [j for j in order_pref if responsive[j]]
    if resp_order and al.traces is not None:
        heat = np.stack([al.traces[cond == pref[j], j, :].mean(axis=0) for j in resp_order])
        p = out / f"{sid}_mean_responses.png"
        lab = {"direction": "preferred direction", "orientation": "preferred orientation"}.get(mode, "preferred condition")
        figs.append({"path": rel(p), "caption": fg.mean_responses(p, sid, heat, al.t, lab, al.response_s)})
    rel_score = np.nan_to_num(df["reliability_split_half"].to_numpy(), nan=-1)
    top = [j for j in np.argsort(-rel_score) if responsive[j]][: params["TOP_N"]]
    ranking = "the responsive ROIs with the highest split-half reliability"
    if not top:
        top = list(np.argsort(-np.nan_to_num(stat, nan=-np.inf))[: params["TOP_N"]])
        ranking = "no ROI passed the responsiveness test: the largest responses"
    if mode in ("direction", "orientation") and top:
        angles, curve, curve_se = curve_at_pref
        cs = []
        for j in top:
            title = f"ROI {tr.roi_ids[j]}" + (f"  gDSI {df['gdsi'][j]:.2f}" if mode == "direction" else "") + f"  gOSI {df['gosi'][j]:.2f}"
            sec = ", ".join(f"{s}={an._fmt_level(conds.values[s].iloc[pref[j]])}" for s in conds.secondary) \
                if curve_kind[j] == "at_preferred" else ("pooled over " + ", ".join(conds.secondary) if conds.secondary else "")
            cs.append({"levels": [an._fmt_level(a) for a in angles], "angles": angles, "mean": curve[:, j],
                       "sem": curve_se[:, j], "title": title + (f"\n{sec}" if sec else ""),
                       "blank": df["blank_mean"][j] if "blank_mean" in df else None,
                       "at_pref": bool(conds.secondary) and curve_kind[j] == "at_preferred",
                       "pref_vs": pref_angle[j] if mode == "direction" else np.nan,
                       "vs_len": df["gdsi"][j] if mode == "direction" else 1.0})
        p = out / f"{sid}_tuning_curves.png"
        figs.append({"path": rel(p), "caption": fg.tuning_curves(p, sid, cs, conds.primary, True, ranking)})
        if mode == "direction":
            p = out / f"{sid}_polar.png"
            figs.append({"path": rel(p), "caption": fg.polar(p, sid, cs)})
        p = out / f"{sid}_osi_dsi.png"
        figs.append({"path": rel(p), "caption": fg.osi_dsi(p, sid, df["gosi"], df["gdsi"], df["osi"], df["dsi"],
                                                         responsive, mode)})
    elif mode == "categorical" and top:
        f = conds.primary
        cs = []
        for j in top:
            sub = tuning[(tuning.roi_id == tr.roi_ids[j]) & (tuning.factor == f) & (tuning.kind == "marginal")]
            cs.append({"levels": sub.level.tolist(), "mean": sub["mean_response"].to_numpy(), "sem": sub["sem"].to_numpy(),
                       "title": f"ROI {tr.roi_ids[j]}  SI {df['selectivity_index'][j]:.2f}",
                       "blank": df["blank_mean"][j] if "blank_mean" in df else None})
        p = out / f"{sid}_tuning_curves.png"
        figs.append({"path": rel(p), "caption": fg.tuning_curves(p, sid, cs, f, False, ranking)})

    # trial bar for the raster
    cmap = fg.plt.get_cmap(fg.CYCLIC if mode in ("direction", "orientation") else "tab10")
    period = 360.0 if mode == "direction" else 180.0
    colours = []
    for c, b in zip(cond, blank):
        if b or c < 0:
            colours.append("#c8c7c2")
        elif mode in ("direction", "orientation"):
            colours.append(cmap((float(conds.values[conds.primary].iloc[c]) % period) / period))
        elif conds.primary:
            lv = conds.values[conds.primary].iloc[c]
            colours.append(cmap(fac_levels[conds.primary].index(lv) % 10))
        else:
            colours.append(fg.BLUE)
    stop = stim["stop_time"].to_numpy(dtype=float)[al.keep]
    stop = np.where(np.isfinite(stop) & (stop > onsets[al.keep]), stop, onsets[al.keep] + al.response_s)
    bar = {"start": onsets[al.keep], "stop": stop, "colors": colours,
           "what": conds.primary if conds.primary else "trial"}

    def med(x, m):
        x = np.asarray(x, float)[m]
        x = x[np.isfinite(x)]
        return float(np.median(x)) if len(x) else None

    n_resp = int(responsive.sum())
    s = {
        "analysis": "stimulus-aligned",
        "n_trials": n_kept, "n_blank_trials": n_blank, "n_conditions": int((ncond > 0).sum()),
        "min_trials_per_condition": min_n,
        "condition_columns": [conds.primary, *conds.secondary] if conds.primary else [],
        "condition_kind": mode, "baseline_window_s": [-al.baseline_s, 0.0], "response_window_s": [0.0, al.response_s],
        "n_responsive": n_resp, "fraction_responsive": n_resp / R,
        "n_tuned": int(tuned.sum()),
        "n_responsive_blank_criterion": int(df["responsive_blank"].sum()) if "responsive_blank" in df else None,
        "median_reliability_split_half": med(df["reliability_split_half"], responsive),
        "median_reliability_trials": med(df["reliability_trials"], responsive),
        "median_lifetime_sparseness": med(df["lifetime_sparseness"], responsive),
        "median_population_sparseness": float(np.nanmedian(pop_sparse)) if np.isfinite(pop_sparse).any() else None,
        "_trials_bar": bar,
    }
    if mode in ("direction", "orientation"):
        s.update({"median_gosi": med(df["gosi"], responsive), "median_osi": med(df["osi"], responsive),
                  "median_gdsi": med(df["gdsi"], responsive), "median_dsi": med(df["dsi"], responsive)})
        if mode == "direction" and n_resp:
            pa = (np.round(pref_angle[responsive] / 45) * 45) % 360
            s["preferred_direction_counts"] = {str(int(a)): int((pa == a).sum()) for a in np.arange(0, 360, 45)}
    if mode == "categorical":
        s["median_selectivity_index"] = med(df["selectivity_index"], responsive)
    for name, Cm in corr:
        key = name.split()[0]
        s[f"mean_{key}_correlation"] = an.mean_offdiag(Cm, mask)
    if len(corr) == 2 and mask is not None and mask.sum() >= 3:
        iu = np.triu_indices(int(mask.sum()), 1)
        a = corr[0][1][np.ix_(mask, mask)][iu]
        b = corr[1][1][np.ix_(mask, mask)][iu]
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() > 3:
            s["signal_noise_correlation_r"] = float(np.corrcoef(a[ok], b[ok])[0, 1])
    s["correlations_over"] = "responsive ROIs" if mask is not None else "all ROIs"
    return {"rois": df, "summary": s, "figures": figs, "tables": files_tables, "mode": mode,
            "pref_angle": pref_angle, "tuned_mask": tuned & np.isfinite(pref_angle), "corr": corr}
