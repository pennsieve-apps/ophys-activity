"""ophys-activity on the synthetic fixture (tests/make_fixture.py): no Docker, no network."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from tests.make_fixture import write_all  # noqa: E402

PROVENANCE_KEYS = {"app", "version", "variant", "tools", "parameters", "inputs", "started", "finished",
                   "wall_seconds", "peak_rss_mb", "gpu", "sessions", "citations"}


def run_app(inp: Path, out: Path, **env) -> subprocess.CompletedProcess:
    e = {**os.environ, "INPUT_DIR": str(inp), "OUTPUT_DIR": str(out), "THREADS": "1", "N_SHUFFLES": "2000", **env}
    return subprocess.run([sys.executable, "-m", "processor.main"], cwd=REPO, env=e, capture_output=True, text=True,
                          timeout=600)


@pytest.fixture(scope="session")
def fixture_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("fixture")
    truth = write_all(d, "both")
    return d, truth


@pytest.fixture(scope="session")
def linked_run(fixture_dir, tmp_path_factory):
    d, truth = fixture_dir
    out = tmp_path_factory.mktemp("out")
    res = run_app(d / "linked", out)
    assert res.returncode == 0, res.stdout + res.stderr
    return out / "ophys-activity", truth, res


def angdiff(a, b, period=360.0):
    return np.abs((np.asarray(a, float) - np.asarray(b, float) + period / 2) % period - period / 2)


def test_contract_files(linked_run):
    top, _, _ = linked_run
    summary = json.loads((top / "summary.json").read_text())
    prov = json.loads((top / "provenance.json").read_text())
    assert set(prov) == PROVENANCE_KEYS
    assert prov["app"] == summary["app"] == "ophys-activity"
    assert set(summary["sessions"]) == {"sub-syn01_ses-grating", "sub-syn02_ses-spont", "sub-syn03_ses-odor"}
    assert all(s["status"] == "ok" for s in summary["sessions"].values())
    assert summary["figures"], "no figures listed"
    for f in summary["figures"]:
        assert (top / f["path"]).is_file(), f["path"]
        assert len(f["caption"]) > 80
    for t in summary["tables"]:
        assert (top / t["path"]).is_file(), t["path"]
    assert (top / "rois.csv").is_file()
    # sessions.csv columns are carried into the tables and the summary
    s = summary["sessions"]["sub-syn03_ses-odor"]
    assert s["subject_id"] == "syn03" and s["group"] == "B"
    rois = pd.read_csv(top / "sub-syn03_ses-odor" / "sub-syn03_ses-odor_rois.csv")
    assert set(rois["subject_id"].astype(str)) == {"syn03"}


def test_grating_recovers_planted_tuning(linked_run):
    top, truth, _ = linked_run
    sid = "sub-syn01_ses-grating"
    s = json.loads((top / "summary.json").read_text())["sessions"][sid]
    assert s["signal"] == "deconvolved"            # Deconvolved > DfOverF
    assert s["stim_table"] == "drifting_gratings"
    assert s["condition_columns"] == ["direction", "contrast"]
    assert s["condition_kind"] == "direction"
    assert s["n_blank_trials"] == 12
    r = pd.read_csv(top / sid / f"{sid}_rois.csv")
    t = truth[sid]
    dir_t, ori_t, un = t["direction_tuned"], t["orientation_tuned"], t["untuned"]
    # responsiveness: all driven ROIs, no undriven one
    assert r.loc[dir_t + ori_t, "responsive"].all()
    assert not r.loc[un, "responsive"].any()
    # preferred direction (vector sum) within 22.5 deg of the planted one
    pref = np.array([t["pref_direction"][j] for j in dir_t])
    assert (angdiff(r.loc[dir_t, "pref_dir_vs"], pref) < 22.5).all()
    assert (r.loc[dir_t, "pref_direction"].to_numpy() == pref).mean() >= 0.95   # peak condition
    # orientation-tuned ROIs: preferred orientation right, direction selectivity low
    pref_o = np.array([t["pref_direction"][j] for j in ori_t]) % 180
    assert (angdiff(r.loc[ori_t, "pref_ori_vs"], pref_o, 180) < 22.5).all()
    assert r.loc[dir_t, "gdsi"].median() > 2 * r.loc[ori_t, "gdsi"].median()
    assert r.loc[dir_t + ori_t, "gosi"].median() > 0.3
    assert r.loc[dir_t, "dsi"].median() > 0.5
    # reliability: driven ROIs reliable
    assert r.loc[dir_t, "reliability_split_half"].median() > 0.7
    # the response matrix for decoding: trials x ROIs
    z = np.load(top / sid / f"{sid}_responses.npz")
    assert z["responses"].shape == (108, 40)
    assert z["aligned"].shape[:2] == (108, 40)
    # shared fluctuation planted: noise correlations positive on average
    assert s["mean_noise_correlation"] > 0
    tun = pd.read_csv(top / sid / f"{sid}_tuning.csv")
    assert {"marginal", "at_preferred"} <= set(tun["kind"])


def test_spontaneous_assemblies(linked_run):
    top, truth, _ = linked_run
    sid = "sub-syn02_ses-spont"
    s = json.loads((top / "summary.json").read_text())["sessions"][sid]
    assert s["analysis"].startswith("spontaneous")
    assert s["n_rois"] == 28 and s["n_rois_total"] == 30      # accepted == false left out
    assert s["n_assemblies"] == 2
    a = pd.read_csv(top / sid / f"{sid}_assemblies.csv")
    found = [set(g.loc[g.member, "roi_id"]) for _, g in a.groupby("assembly")]
    for planted in truth[sid]["assemblies"]:
        best = max(len(set(planted) & f) / len(set(planted) | f) for f in found)
        assert best >= 0.8, (planted, found)
    r = pd.read_csv(top / sid / f"{sid}_rois.csv")
    assert (r["event_rate_hz"] > 0).all()
    assert {"rastermap_rank", "x", "y"} <= set(r.columns)
    assert sorted(r["rastermap_rank"]) == list(range(28))


def test_categorical_csv_stimulus(linked_run):
    top, truth, _ = linked_run
    sid = "sub-syn03_ses-odor"
    s = json.loads((top / "summary.json").read_text())["sessions"][sid]
    assert s["condition_kind"] == "categorical" and s["condition_columns"] == ["condition"]
    r = pd.read_csv(top / sid / f"{sid}_rois.csv")
    t = truth[sid]
    tuned = [j for j, o in enumerate(t["pref_odor"]) if o]
    assert (r.loc[tuned, "pref_condition"].str.replace("condition=", "") == [t["pref_odor"][j] for j in tuned]).all()
    assert r.loc[tuned, "responsive"].all() and not r.loc[t["untuned"], "responsive"].any()
    assert r.loc[tuned, "selectivity_index"].median() > 0.8


def test_flat_layout_same_sessions(fixture_dir, linked_run, tmp_path):
    d, _ = fixture_dir
    res = run_app(d / "flat", tmp_path)
    assert res.returncode == 0, res.stdout + res.stderr
    a = json.loads((tmp_path / "ophys-activity" / "summary.json").read_text())
    b = json.loads((linked_run[0] / "summary.json").read_text())
    assert set(a["sessions"]) == set(b["sessions"])
    for k in a["sessions"]:
        assert a["sessions"][k]["n_responsive" if "n_responsive" in b["sessions"][k] else "n_rois"] == \
            b["sessions"][k]["n_responsive" if "n_responsive" in b["sessions"][k] else "n_rois"]


def test_reproducible(fixture_dir, linked_run, tmp_path):
    d, _ = fixture_dir
    res = run_app(d / "linked", tmp_path)
    assert res.returncode == 0
    sid = "sub-syn01_ses-grating"
    a = pd.read_csv(tmp_path / "ophys-activity" / sid / f"{sid}_rois.csv")
    b = pd.read_csv(linked_run[0] / sid / f"{sid}_rois.csv")
    pd.testing.assert_frame_equal(a, b)


def test_stim_table_none_and_parameters(fixture_dir, tmp_path):
    d, _ = fixture_dir
    res = run_app(d / "linked", tmp_path, STIM_TABLE="none", SIGNAL="dff", T_STOP="100")
    assert res.returncode == 0, res.stdout + res.stderr
    s = json.loads((tmp_path / "ophys-activity" / "summary.json").read_text())["sessions"]
    g = s["sub-syn01_ses-grating"]
    assert g["analysis"].startswith("spontaneous") and g["signal"] == "dff" and g["t_stop"] <= 100


def test_bad_parameters_exit_2(tmp_path):
    for bad in ({"SIGNAL": "voltage"}, {"FDR": "2"}, {"N_SHUFFLES": "10"}, {"T_START": "10", "T_STOP": "5"},
                {"ASSEMBLY_NULL": "x"}):
        res = run_app(tmp_path, tmp_path / "o", **bad)
        assert res.returncode == 2, (bad, res.stdout)


def test_no_input_exit_1(tmp_path):
    (tmp_path / "in").mkdir()
    res = run_app(tmp_path / "in", tmp_path / "o")
    assert res.returncode == 1


def test_pynwb_written_file(tmp_path):
    """A file written by pynwb itself (not our h5py writer) is read the same way."""
    pynwb = pytest.importorskip("pynwb")
    from datetime import datetime, timezone

    from pynwb.ophys import ImageSegmentation, OpticalChannel, RoiResponseSeries, DfOverF, Fluorescence

    nwb = pynwb.NWBFile("test", "id1", datetime(2026, 1, 1, tzinfo=timezone.utc))
    dev = nwb.create_device("scope")
    ip = nwb.create_imaging_plane(name="plane", optical_channel=OpticalChannel("g", "green", 520.0),
                                  description="test", device=dev, excitation_lambda=920.0, imaging_rate=15.0,
                                  indicator="GCaMP6f", location="VISp")
    mod = nwb.create_processing_module("ophys", "ophys")
    seg = ImageSegmentation()
    mod.add(seg)
    ps = seg.create_plane_segmentation("rois", ip, "PlaneSegmentation")
    for k in range(6):
        ps.add_roi(pixel_mask=[(k * 5 + 2, 3, 1.0), (k * 5 + 3, 3, 1.0)])
    rng = np.random.default_rng(0)
    T = 1500
    region = ps.create_roi_table_region("all", region=list(range(6)))
    ts = np.arange(T) / 15.0
    mod.add(DfOverF(RoiResponseSeries(name="RoiResponseSeries", data=rng.normal(0, 1, (T, 6)), rois=region, unit="n.a.",
                                     timestamps=ts)))
    region2 = ps.create_roi_table_region("all", region=list(range(6)))
    mod.add(Fluorescence(RoiResponseSeries(name="RoiResponseSeries", data=rng.normal(5, 1, (T, 6)), rois=region2,
                                           unit="n.a.", timestamps=ts), name="Deconvolved"))
    nwb.add_trial(start_time=10.0, stop_time=12.0)
    for k in range(1, 10):
        nwb.add_trial(start_time=10.0 + 5 * k, stop_time=12.0 + 5 * k)
    (tmp_path / "in").mkdir()
    with pynwb.NWBHDF5IO(str(tmp_path / "in" / "sub-a_ses-b_ophys.nwb"), "w") as io:
        io.write(nwb)
    res = run_app(tmp_path / "in", tmp_path / "o")
    assert res.returncode == 0, res.stdout + res.stderr
    s = json.loads((tmp_path / "o" / "ophys-activity" / "summary.json").read_text())["sessions"]["sub-a_ses-b"]
    assert s["signal"] == "deconvolved" and s["n_rois"] == 6
    assert s["stim_table"] == "trials" and s["n_trials"] == 10 and s["condition_columns"] == []


def test_session_port_nwb_with_traces(fixture_dir, tmp_path):
    """A session-port NWB that also has ROI traces (the Allen companion file next to ophys-traces'
    output) is used for its stimulus table, not analysed as a second session."""
    import shutil

    import h5py

    d, _ = fixture_dir
    inp = tmp_path / "in"
    (inp / "traces").mkdir(parents=True)
    (inp / "session").mkdir()
    shutil.copy(d / "linked/traces/sub-syn01_ses-grating_ophys.nwb", inp / "traces")
    comp = inp / "session/sub-syn01_ses-grating_behavior+ophys.nwb"
    shutil.copy(d / "linked/traces/sub-syn01_ses-grating_ophys.nwb", comp)
    with h5py.File(d / "linked/session/sub-syn01_ses-grating_behavior.nwb", "r") as src, h5py.File(comp, "a") as dst:
        src.copy(src["stimulus/presentation/drifting_gratings"], dst["stimulus/presentation"], "drifting_gratings")
    res = run_app(inp, tmp_path / "o", N_SHUFFLES="500")
    assert res.returncode == 0, res.stdout + res.stderr
    s = json.loads((tmp_path / "o" / "ophys-activity" / "summary.json").read_text())["sessions"]
    assert list(s) == ["sub-syn01_ses-grating"]
    assert s["sub-syn01_ses-grating"]["stim_table"] == "drifting_gratings"
