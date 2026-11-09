"""
Reading optical physiology NWB files with h5py.

Plain h5py (not pynwb) so that any NWB file opens, whatever extensions it uses (ndx-pose,
lab-specific types), and so that only the needed slices are read: the time window of one
RoiResponseSeries, the masks of the analysed ROIs one at a time.

Which series is analysed (SIGNAL=auto): Deconvolved (CASCADE spike rate; OASIS / event series
only when nothing else is in it) > DfOverF > Fluorescence (not Neuropil / residual / background)
> event series (Allen's L0 `DfOverFEvents`, CNMF `events`).
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

log = logging.getLogger("ophys-activity")

KIND_RANK = {"deconvolved": 0, "dff": 1, "fluorescence": 2, "events": 3}
EXCLUDE_RE = re.compile(r"neuropil|residual|background|baseline|noise|contamination", re.I)
EVENT_RE = re.compile(r"event|oasis|^l0", re.I)
DECONV_RE = re.compile(r"deconv|spike|cascade|rate", re.I)
FLUO_PREF = ("corrected", "c", "demixed", "roiresponseseries", "f")
SKIP_TABLES = {"epochs", "invalid_times"}
TABLE_SKIP_COLUMNS = {"id", "timeseries", "tags", "timeseries_index", "tags_index"}


def _ntype(obj) -> str:
    v = obj.attrs.get("neurodata_type", "")
    return v.decode() if isinstance(v, bytes) else str(v)


def _str(v) -> str:
    if isinstance(v, bytes):
        return v.decode("utf-8", "replace")
    if isinstance(v, np.ndarray) and v.shape == ():
        return _str(v[()])
    return str(v)


@dataclass
class SeriesInfo:
    path: str
    container: str
    name: str
    kind: str
    n_t: int
    n_rois: int

    @property
    def rank(self) -> tuple:
        n = self.name.lower()
        within = 0
        if self.kind == "fluorescence":
            within = FLUO_PREF.index(n) if n in FLUO_PREF else len(FLUO_PREF)
        return (KIND_RANK[self.kind], within, self.path)


def _classify(container: str, ctype: str, name: str) -> str | None:
    if EXCLUDE_RE.search(name):
        return None
    if "deconv" in container.lower():
        return "events" if EVENT_RE.search(name) and not DECONV_RE.search(name) else "deconvolved"
    if EVENT_RE.search(name):
        return "events"
    if DECONV_RE.search(name) and "deconv" in name.lower():
        return "deconvolved"
    if ctype == "DfOverF" or "dff" in container.lower() or "dfoverf" in container.lower():
        return "dff"
    return "fluorescence"


def list_series(f: h5py.File) -> list[SeriesInfo]:
    out: list[SeriesInfo] = []
    proc = f.get("processing")
    if proc is None:
        return out
    for mod in proc.values():
        if not isinstance(mod, h5py.Group):
            continue
        for cname, cont in mod.items():
            if not isinstance(cont, h5py.Group):
                continue
            ctype = _ntype(cont)
            members = [(cname, cont)] if ctype == "RoiResponseSeries" else list(cont.items())
            for sname, s in members:
                if not isinstance(s, h5py.Group) or _ntype(s) != "RoiResponseSeries" or "data" not in s:
                    continue
                kind = _classify(cname if s is not cont else mod.name, ctype, sname)
                if kind is None:
                    continue
                d = s["data"]
                n_t = d.shape[0]
                n_r = d.shape[1] if d.ndim > 1 else 1
                out.append(SeriesInfo(s.name, cname, sname, kind, n_t, n_r))
    return sorted(out, key=lambda x: x.rank)


def probe_nwb(path: Path) -> str:
    """'traces' (has an ophys RoiResponseSeries), 'session' (TimeIntervals only), 'other', 'error'."""
    try:
        with h5py.File(path, "r") as f:
            if list_series(f):
                return "traces"
            if list_time_intervals(f):
                return "session"
            return "other"
    except (OSError, KeyError, ValueError) as exc:
        log.warning("cannot open %s as HDF5/NWB: %s", path.name, exc)
        return "error"


def choose_series(series: list[SeriesInfo], signal: str, name: str) -> SeriesInfo | None:
    cands = series
    if name:
        cands = [s for s in series if name.lower() in s.path.lower()]
        if not cands:
            return None
    if signal != "auto":
        cands = [s for s in cands if s.kind == signal]
    return cands[0] if cands else None


def _timestamps(s: h5py.Group, n: int) -> tuple[np.ndarray, str]:
    if "timestamps" in s:
        ts = s["timestamps"]
        if isinstance(ts, h5py.Dataset):
            return np.asarray(ts[:], dtype=np.float64), "timestamps"
    st = s.get("starting_time")
    if st is not None:
        rate = float(st.attrs.get("rate"))
        return float(st[()]) + np.arange(n, dtype=np.float64) / rate, "starting_time + rate"
    raise ValueError(f"{s.name} has neither timestamps nor starting_time")


@dataclass
class Plane:
    path: str
    n_rows: int
    ids: np.ndarray
    columns: dict = field(default_factory=dict)   # 1-D per-ROI columns (accepted, iscell, ...)
    mask_kind: str = ""                            # pixel_mask / image_mask / voxel_mask / ""


@dataclass
class Traces:
    data: np.ndarray            # (time, rois) float32, session-clock samples
    t: np.ndarray               # (time,) seconds, session clock
    roi_ids: np.ndarray         # ids from the PlaneSegmentation (or the series' row indices)
    roi_rows: np.ndarray        # row index in the PlaneSegmentation
    series: SeriesInfo
    plane: Plane | None
    unit: str
    time_source: str
    frame_rate: float
    n_rois_total: int
    n_rois_dropped: int
    selection: str


def _plane_of(f: h5py.File, s: h5py.Group) -> tuple[Plane | None, np.ndarray]:
    rois = s.get("rois")
    if rois is None:
        return None, np.arange(s["data"].shape[1] if s["data"].ndim > 1 else 1)
    rows = np.asarray(rois[:], dtype=np.int64)
    ref = rois.attrs.get("table")
    if ref is None:
        return None, rows
    try:
        ps = f[ref]
    except (ValueError, KeyError, TypeError):
        return None, rows
    ids = np.asarray(ps["id"][:]) if "id" in ps else np.arange(len(rows))
    cols = {}
    for k, v in ps.items():
        if isinstance(v, h5py.Dataset) and v.ndim >= 1 and v.shape[0] == len(ids) and k not in ("id",) \
                and not k.endswith("_index") and k not in ("image_mask", "pixel_mask", "voxel_mask"):
            if v.ndim == 1 and v.dtype.kind in "biuf":
                cols[k] = v[:]
            elif v.ndim == 2 and v.dtype.kind in "biuf" and k == "iscell":
                cols[k] = v[:, 1] if v.shape[1] > 1 else v[:, 0]   # Suite2p (is_cell, probability)
    kind = next((k for k in ("pixel_mask", "image_mask", "voxel_mask") if k in ps), "")
    return Plane(ps.name, len(ids), ids, cols, kind), rows


def read_traces(path: Path, signal: str, series_name: str, t_start: float | None, t_stop: float | None,
                accepted_only: bool) -> Traces:
    with h5py.File(path, "r") as f:
        series = list_series(f)
        if not series:
            raise ValueError("no RoiResponseSeries under processing/")
        info = choose_series(series, signal, series_name)
        if info is None:
            avail = ", ".join(f"{s.path} ({s.kind})" for s in series)
            raise ValueError(f"no series matches SIGNAL={signal} SERIES={series_name or '(any)'}; available: {avail}")
        s = f[info.path]
        d = s["data"]
        t, tsrc = _timestamps(s, d.shape[0])
        i0 = 0 if t_start is None else int(np.searchsorted(t, t_start, side="left"))
        i1 = len(t) if t_stop is None else int(np.searchsorted(t, t_stop, side="right"))
        if i1 - i0 < 10:
            raise ValueError(f"only {max(0, i1 - i0)} samples in the time window")
        plane, rows = _plane_of(f, s)
        n_total = len(rows)
        keep = np.ones(n_total, dtype=bool)
        selection = "all ROIs (no accepted column)"
        if plane is not None and accepted_only:
            if "accepted" in plane.columns:
                keep = np.asarray(plane.columns["accepted"][rows], dtype=bool)
                selection = "accepted == true"
            elif "iscell" in plane.columns and plane.columns["iscell"].dtype.kind == "f":
                keep = plane.columns["iscell"][rows] >= 0.5
                selection = "iscell probability >= 0.5 (no accepted column)"
        elif not accepted_only:
            selection = "all ROIs (ACCEPTED_ONLY=false)"
        if not keep.any():
            raise ValueError(f"no ROIs left after selection ({selection})")
        cols = np.flatnonzero(keep)
        raw = d[i0:i1] if d.ndim > 1 else d[i0:i1][:, None]
        data = np.asarray(raw if len(cols) == raw.shape[1] else raw[:, cols], dtype=np.float32)
        del raw
        conv = float(d.attrs.get("conversion", 1.0) or 1.0)
        off = float(d.attrs.get("offset", 0.0) or 0.0)
        if conv != 1.0 or off != 0.0:
            data = data * np.float32(conv) + np.float32(off)
        unit = _str(d.attrs.get("unit", ""))
        tt = t[i0:i1]
        dt = np.diff(tt)
        fr = float(1.0 / np.median(dt)) if len(dt) else float("nan")
        roi_rows = rows[cols]
        roi_ids = plane.ids[roi_rows] if plane is not None and len(plane.ids) > roi_rows.max() else roi_rows
        bad = ~np.isfinite(data)
        if bad.any():
            # NaN frames (dropped frames in some pipelines): fill per ROI by the ROI's median.
            med = np.nanmedian(np.where(bad, np.nan, data), axis=0)
            data[bad] = np.take(np.nan_to_num(med), np.nonzero(bad)[1])
        return Traces(data, tt, np.asarray(roi_ids), roi_rows, info, plane, unit, tsrc, fr, n_total,
                      int(n_total - keep.sum()), selection)


def read_masks(path: Path, plane: Plane | None, rows: np.ndarray, max_pixels: int = 4_000_000):
    """Per ROI (rows, cols, weights) in image coordinates, the FOV shape and a background image."""
    if plane is None or not plane.mask_kind:
        return None, None, None
    with h5py.File(path, "r") as f:
        ps = f[plane.path]
        masks = []
        shape = None
        if plane.mask_kind == "pixel_mask":
            pm = ps["pixel_mask"]
            idx = np.asarray(ps["pixel_mask_index"][:], dtype=np.int64)
            starts = np.concatenate([[0], idx[:-1]])
            names = pm.dtype.names or ()
            for r in rows:
                seg = pm[starts[r]:idx[r]]
                if names:
                    a, b = np.asarray(seg[names[0]]), np.asarray(seg[names[1]])
                    w = np.asarray(seg[names[2]], dtype=np.float32) if len(names) > 2 else np.ones(len(a), np.float32)
                else:
                    a, b, w = seg[:, 0], seg[:, 1], seg[:, 2] if seg.shape[1] > 2 else np.ones(len(seg))
                # NWB names the fields x, y; they index the image's first and second axis as the
                # summary images are stored ([x, y]), which we draw as rows and columns.
                masks.append((a.astype(np.int64), b.astype(np.int64), np.asarray(w, dtype=np.float32)))
        elif plane.mask_kind == "image_mask":
            im = ps["image_mask"]
            shape = tuple(im.shape[1:3])
            for r in rows:
                m = np.asarray(im[r], dtype=np.float32)
                if m.ndim == 3:
                    m = m.max(axis=-1)
                rr, cc = np.nonzero(m > 0)
                masks.append((rr, cc, m[rr, cc]))
        elif plane.mask_kind == "voxel_mask":
            vm = ps["voxel_mask"]
            idx = np.asarray(ps["voxel_mask_index"][:], dtype=np.int64)
            starts = np.concatenate([[0], idx[:-1]])
            names = vm.dtype.names
            for r in rows:
                seg = vm[starts[r]:idx[r]]
                masks.append((np.asarray(seg[names[0]], np.int64), np.asarray(seg[names[1]], np.int64),
                              np.asarray(seg[names[-1]], np.float32)))
        image, image_name = summary_image(f)
        if shape is None:
            if image is not None:
                shape = image.shape
            else:
                mx = max((int(m[0].max()) for m in masks if len(m[0])), default=0)
                my = max((int(m[1].max()) for m in masks if len(m[1])), default=0)
                shape = (mx + 1, my + 1)
        if image is not None and image.shape != shape:
            image = None
        return masks, shape, (image, image_name)


IMAGE_PREF = ("mean", "meanimg", "average", "avg", "max", "maximum_intensity_projection", "maximum", "correlation", "vcorr")


def summary_image(f: h5py.File):
    """A 2-D summary image (mean > max > correlation) from any `Images` group under processing/."""
    found = []

    def visit(name, obj):
        if isinstance(obj, h5py.Group) and _ntype(obj) == "Images":
            for k, v in obj.items():
                if isinstance(v, h5py.Dataset) and v.ndim == 2:
                    found.append((k, v.name))

    proc = f.get("processing")
    if proc is not None:
        proc.visititems(visit)
    if not found:
        return None, None

    def score(item):
        n = item[0].lower()
        for i, p in enumerate(IMAGE_PREF):
            if n == p or n.startswith(p):
                return i
        return len(IMAGE_PREF)

    k, p = sorted(found, key=score)[0]
    return np.asarray(f[p][:], dtype=np.float32), k


# ---------------------------------------------------------------- time intervals (stimulus tables)

def list_time_intervals(f: h5py.File, anywhere: bool = False) -> dict[str, tuple[str, int]]:
    """{name: (path, n_rows)} of TimeIntervals tables: intervals/*, stimulus/presentation/*
    (and with anywhere=True also processing/*/*, e.g. behavioural states)."""
    out: dict[str, tuple[str, int]] = {}
    roots = ["intervals", "stimulus/presentation"] + (["processing"] if anywhere else [])

    def add(g):
        for k, v in g.items():
            if isinstance(v, h5py.Group) and _ntype(v) == "TimeIntervals" and "start_time" in v:
                out.setdefault(k, (v.name, int(v["start_time"].shape[0])))

    for r in roots:
        g = f.get(r)
        if g is None:
            continue
        if r == "processing":
            for mod in g.values():
                if isinstance(mod, h5py.Group):
                    add(mod)
        else:
            add(g)
    return out


def read_time_intervals(f: h5py.File, path: str) -> pd.DataFrame:
    g = f[path]
    n = int(g["start_time"].shape[0])
    cols = {}
    for k, v in g.items():
        if k in TABLE_SKIP_COLUMNS or k.endswith("_index") or not isinstance(v, h5py.Dataset):
            continue
        if v.ndim != 1 or v.shape[0] != n:
            continue
        if v.dtype.kind in "biuf":
            cols[k] = v[:]
        elif v.dtype.kind in "OSU":
            cols[k] = [_str(x) for x in v[:]]
    df = pd.DataFrame(cols)
    if "stop_time" not in df:
        df["stop_time"] = np.nan
    return df


def session_meta(path: Path) -> dict:
    out = {}
    try:
        with h5py.File(path, "r") as f:
            for k in ("identifier", "session_start_time", "session_description", "timestamps_reference_time"):
                if k in f:
                    out[k] = _str(f[k][()])
            sub = f.get("general/subject")
            if sub is not None:
                out["subject"] = {k: _str(v[()]) for k, v in sub.items() if isinstance(v, h5py.Dataset) and v.shape == ()}
            ip = None

            def visit(name, obj):
                nonlocal ip
                if ip is None and isinstance(obj, h5py.Group) and _ntype(obj) == "ImagingPlane":
                    ip = obj

            if "general" in f:
                f["general"].visititems(visit)
            if ip is not None:
                for k in ("indicator", "location", "imaging_rate", "excitation_lambda"):
                    if k in ip:
                        v = ip[k][()]
                        out[f"imaging_plane_{k}"] = _str(v) if not isinstance(v, (float, np.floating)) else float(v)
    except (OSError, KeyError) as exc:
        log.warning("could not read NWB metadata of %s: %s", path.name, exc)
    return out
