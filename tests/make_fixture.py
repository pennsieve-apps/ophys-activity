"""
Synthetic NWB fixture for ophys-activity, written with h5py only (the review generator image has
numpy and h5py but not pynwb), in the NWB layout the app reads.

Three sessions, frame rate 15 Hz, 64 x 64 px field of view, disc-shaped pixel masks:

  sub-syn01_ses-grating  40 ROIs. Drifting gratings: 8 directions x 2 contrasts x 6 repeats +
                         12 blank trials (2 s on, 1 s grey). ROIs 0-31 are direction tuned with a
                         planted preferred direction (ROI k prefers 45 * (k % 8) deg; von Mises
                         tuning, a smaller response to the opposite direction); ROIs 32-35 are
                         orientation tuned (equal responses to pref and pref + 180); ROIs 36-39
                         are not driven. A shared slow fluctuation gives positive noise
                         correlations. The stimulus table is in a separate session NWB
                         (stimulus/presentation/drifting_gratings) in the `session` port.
                         Traces NWB has Deconvolved (picked), DfOverF and Fluorescence/Neuropil.
  sub-syn02_ses-spont    30 ROIs, 300 s, no stimulus. Two planted assemblies (ROIs 0-5 and 10-15)
                         that fire together at random times; the rest fire independently.
                         ROIs 28-29 are `accepted = false` and must be left out.
  sub-syn03_ses-odor     20 ROIs, a stimulus.csv (start_time, stop_time, condition) with three
                         odours x 10 trials; ROI k responds to odour k % 3 (ROIs 15-19 do not).

Layouts: `linked/` (traces/, session/, metadata/ port folders) and `flat/` (all side by side).
Truth for the tests: `truth.json` next to the layouts.

Usage: python -m tests.make_fixture OUT_DIR [--layout linked|flat|both]
"""
from __future__ import annotations

import argparse
import json
import shutil
import uuid
from pathlib import Path

import h5py
import numpy as np

FS = 15.0
SHAPE = (64, 64)
START = "2026-01-01T10:00:00+00:00"
VLEN = h5py.string_dtype()


def _attrs(obj, ntype: str, ns: str = "core", **kw):
    obj.attrs["neurodata_type"] = ntype
    obj.attrs["namespace"] = ns
    obj.attrs["object_id"] = str(uuid.uuid4())
    for k, v in kw.items():
        obj.attrs[k] = v


def _root(f: h5py.File, identifier: str, desc: str, subject_id: str):
    f.attrs["nwb_version"] = "2.8.0"
    _attrs(f, "NWBFile")
    f.create_dataset("identifier", data=identifier, dtype=VLEN)
    f.create_dataset("session_description", data=desc, dtype=VLEN)
    f.create_dataset("session_start_time", data=START, dtype=VLEN)
    f.create_dataset("timestamps_reference_time", data=START, dtype=VLEN)
    f.create_dataset("file_create_date", data=[START], dtype=VLEN)
    for g in ("acquisition", "analysis", "processing", "stimulus/presentation", "stimulus/templates", "intervals",
              "general"):
        f.require_group(g)
    sub = f.create_group("general/subject")
    _attrs(sub, "Subject")
    sub.create_dataset("subject_id", data=subject_id, dtype=VLEN)
    sub.create_dataset("species", data="Mus musculus", dtype=VLEN)


def _disc_masks(n: int, rng: np.random.Generator, radius: int = 3):
    """n disc masks on a jittered 8-px grid (no overlaps), in random grid slots."""
    slots = [(x, y) for x in range(6, SHAPE[0] - 5, 8) for y in range(6, SHAPE[1] - 5, 8)]
    pick = rng.permutation(len(slots))[:n]
    xs, ys = np.meshgrid(np.arange(-radius, radius + 1), np.arange(-radius, radius + 1), indexing="ij")
    disc = xs ** 2 + ys ** 2 <= radius ** 2
    masks, centres = [], []
    for i in pick:
        cx, cy = np.array(slots[i]) + rng.integers(-1, 2, size=2)
        centres.append((cx, cy))
        masks.append((cx + xs[disc], cy + ys[disc], np.ones(disc.sum(), np.float32)))
    return masks, np.array(centres)


def write_traces(path: Path, sid: str, subject: str, t: np.ndarray, dff: np.ndarray, rng, accepted=None,
                 deconvolved: np.ndarray | None = None):
    T, R = dff.shape
    masks, _ = _disc_masks(R, rng)
    with h5py.File(path, "w") as f:
        _root(f, str(uuid.uuid4()), f"synthetic ophys-activity fixture {sid}", subject)
        dev = f.create_group("general/devices/Microscope")
        _attrs(dev, "Device")
        ip = f.create_group("general/optophysiology/ImagingPlane")
        _attrs(ip, "ImagingPlane")
        ip.create_dataset("indicator", data="GCaMP6f (synthetic)", dtype=VLEN)
        ip.create_dataset("location", data="VISp (synthetic)", dtype=VLEN)
        ip.create_dataset("imaging_rate", data=FS)
        mod = f.create_group("processing/ophys")
        _attrs(mod, "ProcessingModule", description="optical physiology")
        seg = mod.create_group("ImageSegmentation")
        _attrs(seg, "ImageSegmentation")
        ps = seg.create_group("PlaneSegmentation")
        cols = ["pixel_mask", "accepted", "iscell"]
        _attrs(ps, "PlaneSegmentation", description="synthetic ROIs", colnames=np.array(cols, dtype=object))
        ps["imaging_plane"] = h5py.SoftLink("/general/optophysiology/ImagingPlane")
        ids = ps.create_dataset("id", data=np.arange(R, dtype=np.int64))
        _attrs(ids, "ElementIdentifiers", ns="hdmf-common")
        pm_dtype = np.dtype([("x", "<u4"), ("y", "<u4"), ("weight", "<f4")])
        allpix = np.concatenate([np.array(list(zip(*m)), dtype=pm_dtype) for m in masks])
        pm = ps.create_dataset("pixel_mask", data=allpix)
        _attrs(pm, "VectorData", ns="hdmf-common", description="pixel masks")
        idx = ps.create_dataset("pixel_mask_index", data=np.cumsum([len(m[0]) for m in masks]).astype(np.uint32))
        _attrs(idx, "VectorIndex", ns="hdmf-common", target=pm.ref)
        acc = np.ones(R, bool) if accepted is None else accepted
        a = ps.create_dataset("accepted", data=acc)
        _attrs(a, "VectorData", ns="hdmf-common", description="accepted")
        ic = ps.create_dataset("iscell", data=np.where(acc, 0.9, 0.2))
        _attrs(ic, "VectorData", ns="hdmf-common", description="cell probability")
        # summary image: the masks, blurred a little, plus noise
        img = rng.normal(100, 5, SHAPE).astype(np.float32)
        for x, y, _ in masks:
            img[x, y] += 60
        imgs = mod.create_group("SummaryImages")
        _attrs(imgs, "Images", description="summary images")
        im = imgs.create_dataset("mean", data=img)
        _attrs(im, "GrayscaleImage")

        def series(container: str, ctype: str, name: str, data: np.ndarray, unit: str):
            g = mod.require_group(container)
            if "neurodata_type" not in g.attrs:
                _attrs(g, ctype)
            s = g.create_group(name)
            _attrs(s, "RoiResponseSeries", description=f"{container}/{name}")
            d = s.create_dataset("data", data=data.astype(np.float32), chunks=(min(T, 1024), R))
            d.attrs["unit"] = unit
            d.attrs["conversion"] = 1.0
            d.attrs["resolution"] = -1.0
            ts = s.create_dataset("timestamps", data=t)
            ts.attrs["interval"] = 1
            ts.attrs["unit"] = "seconds"
            r = s.create_dataset("rois", data=np.arange(R, dtype=np.int64))
            _attrs(r, "DynamicTableRegion", ns="hdmf-common", table=ps.ref, description="ROIs")

        series("DfOverF", "DfOverF", "RoiResponseSeries", dff, "a.u.")
        series("Fluorescence", "Fluorescence", "Neuropil", dff * 0.3 + 1.0, "a.u.")
        series("Fluorescence", "Fluorescence", "RoiResponseSeries", dff + 1.0, "a.u.")
        if deconvolved is not None:
            series("Deconvolved", "Fluorescence", "RoiResponseSeries", deconvolved, "spikes/s")
            series("Deconvolved", "Fluorescence", "OASIS", (deconvolved > 2).astype(np.float32), "a.u.")


def write_session_nwb(path: Path, subject: str, table: dict, name: str):
    n = len(table["start_time"])
    with h5py.File(path, "w") as f:
        _root(f, str(uuid.uuid4()), "synthetic stimulus table", subject)
        g = f.create_group(f"stimulus/presentation/{name}")
        cols = list(table)
        _attrs(g, "TimeIntervals", description="synthetic gratings", colnames=np.array(cols, dtype=object))
        ids = g.create_dataset("id", data=np.arange(n))
        _attrs(ids, "ElementIdentifiers", ns="hdmf-common")
        for k, v in table.items():
            d = g.create_dataset(k, data=np.asarray(v))
            _attrs(d, "VectorData", ns="hdmf-common", description=k)
        ep = f.create_group("intervals/epochs")
        _attrs(ep, "TimeIntervals", description="epochs", colnames=np.array(["start_time", "stop_time"], dtype=object))
        ep.create_dataset("id", data=[0])
        ep.create_dataset("start_time", data=[0.0])
        ep.create_dataset("stop_time", data=[float(np.max(table["stop_time"]))])


def calcium(rate: np.ndarray, tau: float = 0.7) -> np.ndarray:
    """Convolve a (T, R) rate with an exponential kernel (GCaMP6f-like decay)."""
    k = np.exp(-np.arange(int(5 * tau * FS)) / (tau * FS))
    out = np.empty_like(rate)
    for j in range(rate.shape[1]):
        out[:, j] = np.convolve(rate[:, j], k)[: rate.shape[0]]
    return out / k.sum() * 3


def grating_session(rng):
    R = 40
    dirs = np.arange(0, 360, 45.0)
    trials = [(d, c) for d in dirs for c in (0.4, 0.8)] * 6 + [(np.nan, np.nan)] * 12
    order = rng.permutation(len(trials))
    trials = [trials[i] for i in order]
    t0 = 10.0
    starts = t0 + 3.0 * np.arange(len(trials))
    stops = starts + 2.0
    T = int((stops[-1] + 10) * FS)
    t = np.arange(T) / FS + 0.5
    pref = np.full(R, np.nan)
    pref[:32] = 45.0 * (np.arange(32) % 8)
    pref[32:36] = [0, 45, 90, 135]
    rate = np.zeros((T, R))
    for (d, c), s, e in zip(trials, starts, stops):
        if np.isnan(d):
            continue
        sl = (t >= s) & (t < e)
        for j in range(36):
            dd = np.deg2rad(d - pref[j])
            if j < 32:
                g = np.exp(2.5 * (np.cos(dd) - 1)) + 0.25 * np.exp(2.5 * (np.cos(dd - np.pi) - 1))
            else:
                g = np.exp(2.5 * (np.cos(2 * dd) - 1))
            rate[sl, j] += 4.0 * g * (0.6 + c)
    shared = np.convolve(rng.normal(0, 1, T), np.ones(int(2 * FS)) / (2 * FS), mode="same")
    rate += np.clip(0.4 + 0.5 * shared[:, None], 0, None) * rng.uniform(0.5, 1.5, R)[None, :]
    spikes = rng.poisson(rate / FS)
    dff = calcium(spikes.astype(float)) + rng.normal(0, 0.08, (T, R))
    deconv = spikes * FS * 0.5 + np.abs(rng.normal(0, 0.3, (T, R)))
    table = {"start_time": starts, "stop_time": stops,
             "direction": np.array([d for d, _ in trials]), "contrast": np.array([c for _, c in trials]),
             "is_blank_sweep": np.array([np.isnan(d) for d, _ in trials])}
    truth = {"pref_direction": [None if np.isnan(p) else float(p) for p in pref], "direction_tuned": list(range(32)),
             "orientation_tuned": list(range(32, 36)), "untuned": list(range(36, 40))}
    return t, dff, deconv, table, truth


def spont_session(rng):
    R, dur = 30, 300.0
    T = int(dur * FS)
    t = np.arange(T) / FS
    spikes = rng.poisson(0.05 / FS * 15, (T, R)).astype(float)
    groups = [list(range(0, 6)), list(range(10, 16))]
    for g in groups:
        times = rng.choice(T - 5, size=40, replace=False)
        for tt in times:
            for j in g:
                if rng.random() < 0.85:
                    spikes[tt + rng.integers(0, 2), j] += rng.integers(2, 5)
    dff = calcium(spikes) + rng.normal(0, 0.08, (T, R))
    acc = np.ones(R, bool)
    acc[28:] = False
    return t, dff, acc, {"assemblies": groups, "rejected": [28, 29]}


def odor_session(rng):
    R = 20
    odors = ["acetone", "ethyl_butyrate", "limonene"]
    trials = list(np.repeat(odors, 10))
    trials = [trials[i] for i in rng.permutation(len(trials))]
    starts = 5.0 + 6.0 * np.arange(len(trials))
    stops = starts + 1.5
    T = int((stops[-1] + 8) * FS)
    t = np.arange(T) / FS
    rate = np.full((T, R), 0.3)
    for o, s, e in zip(trials, starts, stops):
        sl = (t >= s) & (t < e + 0.5)
        for j in range(15):
            if odors[j % 3] == o:
                rate[sl, j] += 5.0
    spikes = rng.poisson(rate / FS).astype(float)
    dff = calcium(spikes) + rng.normal(0, 0.08, (T, R))
    stim = {"start_time": starts, "stop_time": stops, "condition": trials}
    truth = {"pref_odor": [odors[j % 3] if j < 15 else None for j in range(R)], "untuned": list(range(15, 20))}
    return t, dff, stim, truth


def write_all(out: Path, layout: str) -> dict:
    rng = np.random.default_rng(1234)
    stage = out / "_stage"
    if stage.exists():
        shutil.rmtree(stage)
    (stage / "traces").mkdir(parents=True)
    (stage / "session").mkdir()
    (stage / "metadata").mkdir()
    truth = {}
    t, dff, deconv, table, tr = grating_session(rng)
    write_traces(stage / "traces/sub-syn01_ses-grating_ophys.nwb", "sub-syn01_ses-grating", "syn01", t, dff, rng,
                 deconvolved=deconv)
    write_session_nwb(stage / "session/sub-syn01_ses-grating_behavior.nwb", "syn01", table, "drifting_gratings")
    truth["sub-syn01_ses-grating"] = tr
    t, dff, acc, tr = spont_session(rng)
    write_traces(stage / "traces/sub-syn02_ses-spont_ophys.nwb", "sub-syn02_ses-spont", "syn02", t, dff, rng,
                 accepted=acc)
    truth["sub-syn02_ses-spont"] = tr
    t, dff, stim, tr = odor_session(rng)
    write_traces(stage / "traces/sub-syn03_ses-odor_ophys.nwb", "sub-syn03_ses-odor", "syn03", t, dff, rng)
    with open(stage / "session/sub-syn03_ses-odor_stimulus.csv", "w") as fh:
        fh.write("start_time,stop_time,condition\n")
        for s, e, c in zip(stim["start_time"], stim["stop_time"], stim["condition"]):
            fh.write(f"{s:.3f},{e:.3f},{c}\n")
    truth["sub-syn03_ses-odor"] = tr
    with open(stage / "metadata/sessions.csv", "w") as fh:
        fh.write("session_id,path,subject_id,condition,group\n")
        fh.write("sub-syn01_ses-grating,sub-syn01_ses-grating_ophys.nwb,syn01,gratings,A\n")
        fh.write("sub-syn02_ses-spont,sub-syn02_ses-spont_ophys.nwb,syn02,spontaneous,A\n")
        fh.write("sub-syn03_ses-odor,sub-syn03_ses-odor_ophys.nwb,syn03,odor,B\n")
    if layout in ("linked", "both"):
        dst = out / "linked"
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(stage, dst)
    if layout in ("flat", "both"):
        dst = out / "flat"
        if dst.exists():
            shutil.rmtree(dst)
        dst.mkdir(parents=True)
        for p in stage.rglob("*"):
            if p.is_file():
                shutil.copy2(p, dst / p.name)
    shutil.rmtree(stage)
    (out / "truth.json").write_text(json.dumps(truth, indent=1))
    return truth


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("out", type=Path)
    ap.add_argument("--layout", choices=["linked", "flat", "both"], default="both")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    write_all(a.out, a.layout)
    print(f"fixture written to {a.out}")


if __name__ == "__main__":
    main()
