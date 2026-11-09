"""
ophys-activity: what does a population of imaged neurons do, and what does it respond to?

Reads ROI activity from NWB (`processing/ophys`: Deconvolved > DfOverF > Fluorescence), and when a
stimulus / trials table is available (a TimeIntervals table in a session NWB or in the traces NWB,
or a stimulus.csv) aligns the activity to it. Writes, per session, under
$OUTPUT_DIR/ophys-activity/<session_id>/:

  <sid>_rois.csv          one row per analysed ROI: activity, responsiveness, tuning, reliability
  <sid>_tuning.csv        long table of tuning curves (factor, level, mean, sem, n_trials)
  <sid>_trials.csv        the stimulus table as analysed (condition label, blank, kept)
  <sid>_responses.npz     trials x ROIs response matrix (for decoding) and aligned traces
  <sid>_correlations.npz  signal, noise and activity correlation matrices
  <sid>_assemblies.csv / .npz   assembly weights, members, activation strength
  figures (PNG)

and at the top: rois.csv (all sessions), summary.json, provenance.json.
"""
from __future__ import annotations

import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from . import common

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", stream=sys.stdout)
log = logging.getLogger("ophys-activity")
for noisy in ("matplotlib", "PIL", "fontTools", "numba", "h5py"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

APP = "ophys-activity"
SIGNALS = ("auto", "deconvolved", "dff", "fluorescence", "events")
DEFAULTS = {
    "SIGNAL": "auto",
    "SERIES": "",
    "ACCEPTED_ONLY": "true",
    "T_START": "",
    "T_STOP": "",
    "STIM_TABLE": "auto",
    "CONDITION": "auto",
    "MAX_LEVELS": "24",
    "MIN_TRIALS": "5",
    "BASELINE_S": "1.0",
    "RESPONSE_S": "0",
    "POST_S": "1.0",
    "N_SHUFFLES": "10000",
    "FDR": "0.05",
    "BIN_S": "0.2",
    "EVENT_Z": "3.0",
    "ASSEMBLY_NULL": "shift",
    "TOP_N": "12",
    "SEED": "42",
    "THREADS": "0",
}
CITATIONS = [
    "10.1038/s41593-024-01783-4",        # Rastermap
    "10.1016/j.jneumeth.2013.04.010",    # assemblies (Lopes-dos-Santos 2013)
    "10.3389/fncir.2014.00092",          # OSI/DSI vector sum (Mazurek 2014)
    "10.1038/nn.2842",                   # signal and noise correlations (Cohen & Kohn 2011)
    "10.1126/science.287.5456.1273",     # sparseness (Vinje & Gallant 2000)
    "10.1111/j.2517-6161.1995.tb02031.x",  # Benjamini-Hochberg FDR
    "10.1038/s41593-019-0550-9",         # de Vries 2020 (Allen Brain Observatory metrics)
]


class ParameterError(ValueError):
    pass


def _float_or_none(v: str, name: str):
    v = (v or "").strip()
    if v == "" or v.lower() in ("none", "null"):
        return None
    try:
        return float(v)
    except ValueError as exc:
        raise ParameterError(f"{name} must be a number or empty, got {v!r}") from exc


def read_parameters(env=os.environ) -> dict:
    raw = {k: (env.get(k) if (env.get(k) or "").strip() != "" else v) for k, v in DEFAULTS.items()}
    p: dict = {}
    p["SIGNAL"] = raw["SIGNAL"].strip().lower()
    if p["SIGNAL"] not in SIGNALS:
        raise ParameterError(f"SIGNAL must be one of {', '.join(SIGNALS)}, got {raw['SIGNAL']!r}")
    p["SERIES"] = raw["SERIES"].strip()
    acc = raw["ACCEPTED_ONLY"].strip().lower()
    if acc not in ("true", "false", "1", "0", "yes", "no"):
        raise ParameterError(f"ACCEPTED_ONLY must be true or false, got {raw['ACCEPTED_ONLY']!r}")
    p["ACCEPTED_ONLY"] = acc in ("true", "1", "yes")
    p["T_START"] = _float_or_none(raw["T_START"], "T_START")
    p["T_STOP"] = _float_or_none(raw["T_STOP"], "T_STOP")
    if p["T_START"] is not None and p["T_STOP"] is not None and p["T_STOP"] <= p["T_START"]:
        raise ParameterError("T_STOP must be after T_START")
    p["STIM_TABLE"] = raw["STIM_TABLE"].strip() or "auto"
    p["CONDITION"] = raw["CONDITION"].strip() or "auto"
    p["ASSEMBLY_NULL"] = raw["ASSEMBLY_NULL"].strip().lower()
    if p["ASSEMBLY_NULL"] not in ("shift", "mp"):
        raise ParameterError(f"ASSEMBLY_NULL must be 'shift' or 'mp', got {raw['ASSEMBLY_NULL']!r}")
    try:
        for k in ("MAX_LEVELS", "MIN_TRIALS", "N_SHUFFLES", "TOP_N", "SEED", "THREADS"):
            p[k] = int(raw[k])
        for k in ("BASELINE_S", "RESPONSE_S", "POST_S", "FDR", "BIN_S", "EVENT_Z"):
            p[k] = float(raw[k])
    except ValueError as exc:
        raise ParameterError(f"bad numeric parameter: {exc}") from exc
    if p["MAX_LEVELS"] < 2 or p["MIN_TRIALS"] < 2:
        raise ParameterError("MAX_LEVELS and MIN_TRIALS must be >= 2")
    if p["N_SHUFFLES"] < 99:
        raise ParameterError("N_SHUFFLES must be >= 99")
    if p["TOP_N"] < 1 or p["THREADS"] < 0:
        raise ParameterError("TOP_N must be >= 1 and THREADS >= 0")
    if p["BASELINE_S"] <= 0 or p["RESPONSE_S"] < 0 or p["POST_S"] < 0 or p["BIN_S"] <= 0 or p["EVENT_Z"] <= 0:
        raise ParameterError("BASELINE_S, BIN_S, EVENT_Z must be > 0; RESPONSE_S, POST_S >= 0")
    if not 0 < p["FDR"] < 1:
        raise ParameterError("FDR must be in (0, 1)")
    return p


def main() -> int:
    started = datetime.now(timezone.utc)
    t_wall = time.time()
    try:
        params = read_parameters()
    except ParameterError as exc:
        log.error("bad parameters: %s", exc)
        return 2
    threads = common.set_threads(params["THREADS"])
    from .session import run_session, select_stimulus  # noqa: F401  (imports numpy after THREADS is set)
    from .inputs import build_sessions, discover

    in_dir = Path(os.environ.get("INPUT_DIR", "/data/input"))
    out_root = Path(os.environ.get("OUTPUT_DIR", "/data/output"))
    out_dir = out_root / APP
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("%s %s: INPUT_DIR=%s OUTPUT_DIR=%s threads=%d params=%s", APP, common.app_version(), in_dir, out_root,
             threads, params)
    if not in_dir.exists():
        log.error("INPUT_DIR %s does not exist", in_dir)
        return 1
    found = discover(in_dir)
    sessions, warnings = build_sessions(found, in_dir)
    for w in warnings:
        log.warning(w)
    log.info("found %d traces NWB, %d session NWB, %d stimulus CSV, %d sessions.csv; %d sessions",
             len(found.traces), len(found.stim_nwb), len(found.stim_csv), len(found.sheets), len(sessions))
    if not sessions:
        log.error("no NWB file with processing/ophys RoiResponseSeries found under %s", in_dir)
        write_outputs(out_dir, params, {}, [], [], warnings + ["no usable input"], started, t_wall, in_dir, [])
        return 1

    import pandas as pd

    results: dict = {}
    figures: list = []
    tables: list = []
    all_rois = []
    inputs_seen = []
    for sess in sessions:
        t0 = time.time()
        inputs_seen += [common.rel(p, in_dir) for p in [sess.traces, *sess.stim_nwb, *sess.stim_csv]]
        try:
            res = run_session(sess, params, out_dir, in_dir)
        except Exception as exc:  # a failed session is logged and skipped
            log.exception("session %s failed: %s", sess.session_id, exc)
            results[sess.session_id] = {"status": "failed", "error": str(exc), "seconds": round(time.time() - t0, 1)}
            continue
        res["summary"]["status"] = "ok"
        res["summary"]["seconds"] = round(time.time() - t0, 1)
        results[sess.session_id] = res["summary"]
        figures += res["figures"]
        tables += res["tables"]
        all_rois.append(res["rois"])
        warnings += [f"{sess.session_id}: {w}" for w in res["warnings"]]
        s = res["summary"]
        log.info("session %s: %d ROIs, signal %s, %s trials, %s responsive, %s assemblies, %.1f s",
                 sess.session_id, s["n_rois"], s["signal"], s.get("n_trials", 0), s.get("n_responsive", "-"),
                 s.get("n_assemblies", 0), time.time() - t0)
    if all_rois:
        pd.concat(all_rois, ignore_index=True).to_csv(out_dir / "rois.csv", index=False, float_format="%.6g")
        tables.insert(0, {"path": "rois.csv", "description": "All sessions' ROI tables (one row per ROI)."})
    n_ok = sum(1 for r in results.values() if r.get("status") == "ok")
    write_outputs(out_dir, params, results, figures, tables, warnings, started, t_wall, in_dir, inputs_seen)
    log.info("done: %d of %d sessions ok, peak memory %s MB, %.1f s", n_ok, len(results), common.peak_rss_mb(),
             time.time() - t_wall)
    return 0 if n_ok else 1


def write_outputs(out_dir: Path, params, results, figures, tables, warnings, started, t_wall, in_dir, inputs_seen):
    version = common.app_version()
    ok = {k: v for k, v in results.items() if v.get("status") == "ok"}
    summary = {
        "app": APP,
        "version": version,
        "n_sessions": len(results),
        "n_failed": len(results) - len(ok),
        "sessions": results,
        "figures": figures,
        "tables": tables,
        "warnings": warnings,
    }
    common.write_json(out_dir / "summary.json", summary)
    finished = datetime.now(timezone.utc)
    prov = {
        "app": APP,
        "version": version,
        "variant": "cpu",
        "tools": common.tool_versions(["numpy", "scipy", "pandas", "scikit-learn", "rastermap", "numba", "h5py",
                                       "matplotlib"]),
        "parameters": params,
        "inputs": inputs_seen,
        "started": started.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "finished": finished.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "wall_seconds": round(time.time() - t_wall, 1),
        "peak_rss_mb": common.peak_rss_mb(),
        "gpu": None,
        "sessions": {k: {"status": v.get("status"), "seconds": v.get("seconds"),
                         **({"error": v["error"]} if "error" in v else {})} for k, v in results.items()},
        "citations": CITATIONS,
    }
    common.write_json(out_dir / "provenance.json", prov)


if __name__ == "__main__":
    sys.exit(main())
