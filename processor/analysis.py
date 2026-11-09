"""
The analyses: activity summary, stimulus-aligned responses, responsiveness, tuning, reliability,
signal and noise correlations, sparseness.

Conventions used throughout:
  X        (time, rois) float32 activity on the session clock, t its times (s)
  r        (trials, rois) trial response = mean over the response window minus mean over the
           pre-stimulus baseline window of the same trial (baseline-subtracted)
  cond     (trials,) int index of the trial's condition (-1 = blank or excluded)
"""
from __future__ import annotations

import logging
import math
import re
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

log = logging.getLogger("ophys-activity")

ANGLE_RE = re.compile(r"direction|orientation|^ori(_|$)|^dir(_|$)|angle|heading|grating_ori", re.I)
BLANK_RE = re.compile(r"blank|null_stim", re.I)
SECONDARY_RE = re.compile(r"temporal_freq|spatial_freq|contrast|^tf(_|$)|^sf(_|$)|^size|^speed|phase", re.I)
GENERIC_RE = re.compile(r"^(condition|stimulus|stim|stim_type|trial_type|odou?r|image|cue|tone|type|label|name)$", re.I)
NOT_CONDITION = {"start_time", "stop_time", "id", "tags", "timeseries", "session_id"}


# ------------------------------------------------------------------ continuous activity

def bin_activity(X: np.ndarray, t: np.ndarray, bin_s: float) -> tuple[np.ndarray, np.ndarray]:
    """Mean of the frames in consecutive bins of bin_s seconds (frames as they are if already coarser)."""
    dt = float(np.median(np.diff(t))) if len(t) > 1 else bin_s
    if bin_s <= dt * 1.01:
        return X, t
    edges_idx = np.floor((t - t[0]) / bin_s).astype(np.int64)
    starts = np.flatnonzero(np.r_[True, np.diff(edges_idx) != 0])
    sums = np.add.reduceat(X.astype(np.float64), starts, axis=0)
    counts = np.diff(np.r_[starts, len(t)])
    Xb = (sums / counts[:, None]).astype(np.float32)
    tb = t[0] + (edges_idx[starts] + 0.5) * bin_s
    return Xb, tb


def zscore(X: np.ndarray) -> np.ndarray:
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True)
    sd[sd == 0] = 1.0
    return ((X - mu) / sd).astype(np.float32)


def robust_noise(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per ROI baseline (median) and noise (1.4826 * MAD; the SD when the MAD is 0)."""
    med = np.median(X, axis=0)
    mad = 1.4826 * np.median(np.abs(X - med), axis=0)
    sd = X.std(axis=0)
    noise = np.where(mad > 0, mad, sd)
    return med, noise


def event_counts(X: np.ndarray, z: float) -> tuple[np.ndarray, np.ndarray]:
    """Transients per ROI: upward crossings of median + z * noise, re-armed only after the trace has
    fallen back below median + z/2 * noise (hysteresis, so one transient is one event).
    Returns (counts, noise). Vectorised: the hi (+1) / lo (-1) state is carried forward over the
    samples in between, and an event is a step into +1."""
    med, noise = robust_noise(X)
    T, R = X.shape
    counts = np.zeros(R, dtype=np.int64)
    rows = np.arange(T, dtype=np.int32)[:, None]
    for a in range(0, R, 64):
        sl = slice(a, min(R, a + 64))
        x = X[:, sl]
        nz = noise[sl]
        st = np.zeros(x.shape, dtype=np.int8)
        st[x < med[sl] + 0.5 * z * nz] = -1
        st[x > med[sl] + z * nz] = 1
        st[:, nz <= 0] = 0
        last = np.where(st != 0, rows, 0)
        np.maximum.accumulate(last, axis=0, out=last)
        filled = np.take_along_axis(st, last, axis=0)
        filled[(last == 0) & (st[:1] == 0)] = -1          # before the first hi/lo sample: armed
        up = (filled[1:] == 1) & (filled[:-1] != 1)
        counts[sl] = up.sum(axis=0) + (filled[0] == 1)
    return counts, noise


def activity_summary(X: np.ndarray, t: np.ndarray, z: float) -> pd.DataFrame:
    duration = float(t[-1] - t[0]) + float(np.median(np.diff(t)))
    counts, noise = event_counts(X, z)
    return pd.DataFrame({
        "mean_signal": X.mean(axis=0),
        "sd_signal": X.std(axis=0),
        "noise_level": noise,
        "n_events": counts,
        "event_rate_hz": counts / duration,
    })


def corr_matrix(X: np.ndarray) -> np.ndarray:
    """Pearson correlation between columns; constant columns give NaN rows."""
    Z = X - X.mean(axis=0, keepdims=True)
    sd = np.sqrt((Z ** 2).sum(axis=0))
    with np.errstate(invalid="ignore", divide="ignore"):
        C = (Z.T @ Z) / np.outer(sd, sd)
    C[~np.isfinite(C)] = np.nan
    return C.astype(np.float32)


def mean_offdiag(C: np.ndarray, mask: np.ndarray | None = None) -> float:
    if mask is not None:
        C = C[np.ix_(mask, mask)]
    n = C.shape[0]
    if n < 2:
        return float("nan")
    iu = np.triu_indices(n, 1)
    v = C[iu]
    v = v[np.isfinite(v)]
    return float(v.mean()) if len(v) else float("nan")


# ------------------------------------------------------------------ stimulus table

@dataclass
class Conditions:
    primary: str | None
    kind: str                      # direction / orientation / categorical / none
    secondary: list[str]
    labels: list[str]              # one per condition index
    values: pd.DataFrame           # per condition: one column per factor
    cond: np.ndarray               # per trial, -1 = blank
    blank: np.ndarray              # per trial bool
    notes: list[str] = field(default_factory=list)


def _levels(col: pd.Series) -> int:
    return col.dropna().nunique()


def _is_angle_col(name: str, col: pd.Series) -> bool:
    return bool(ANGLE_RE.search(name)) and pd.api.types.is_numeric_dtype(col)


def detect_conditions(df: pd.DataFrame, condition: str, max_levels: int) -> Conditions:
    notes: list[str] = []
    blank = np.zeros(len(df), dtype=bool)
    blank_cols = [c for c in df.columns if BLANK_RE.search(c)]
    for c in blank_cols:
        v = df[c]
        if v.dtype == bool or pd.api.types.is_numeric_dtype(v):
            blank |= v.fillna(0).astype(float).to_numpy() > 0
        else:
            blank |= v.astype(str).str.lower().isin(["true", "1", "yes"]).to_numpy()
    cands = []
    for c in df.columns:
        if c in NOT_CONDITION or c in blank_cols or c.endswith("_time") or c.endswith("_index"):
            continue
        n = _levels(df[c])
        if 2 <= n <= max_levels:
            cands.append(c)
    primary = None
    secondary: list[str] = []
    if condition and condition.lower() not in ("auto", "none"):
        want = [c.strip() for c in condition.split(",") if c.strip()]
        missing = [c for c in want if c not in df.columns]
        if missing:
            notes.append(f"CONDITION columns not in the table: {', '.join(missing)}; detecting automatically")
        else:
            primary, secondary = want[0], want[1:]
    elif condition.lower() == "none":
        cands = []
    if primary is None and cands:
        angle = [c for c in cands if _is_angle_col(c, df[c])]
        generic = [c for c in cands if GENERIC_RE.match(c)]
        primary = (angle or generic or cands)[0]
        secondary = [c for c in cands if c != primary and SECONDARY_RE.search(c)][:2]
    kind = "none"
    if primary is not None:
        col = df[primary]
        if _is_angle_col(primary, col):
            # Angles spanning more than a half circle are directions of motion, else orientations.
            vals = col.dropna().to_numpy(dtype=float) % 360.0
            kind = "direction" if vals.max() >= 180.0 else "orientation"
        else:
            kind = "categorical"
        prim = df[primary]
        if pd.api.types.is_numeric_dtype(prim):
            blank |= prim.isna().to_numpy()
        else:
            blank |= prim.isna().to_numpy() | (prim.astype(str).str.strip().isin(["", "nan", "None"])).to_numpy()
    factors = [primary] + secondary if primary else []
    cond = np.full(len(df), -1, dtype=np.int64)
    labels: list[str] = []
    rows = []
    if factors:
        keys = df[factors].copy()
        for c in factors:
            if kind in ("direction", "orientation") and c == primary:
                keys[c] = keys[c].astype(float) % 360.0
        ok = ~blank & keys.notna().all(axis=1).to_numpy()
        uniq = keys[ok].drop_duplicates()
        sort_cols = list(uniq.columns)
        try:
            uniq = uniq.sort_values(sort_cols)
        except TypeError:
            uniq = uniq.astype(str).sort_values(sort_cols)
        lut = {}
        for i, (_, row) in enumerate(uniq.iterrows()):
            key = tuple(row[c] for c in factors)
            lut[key] = i
            labels.append("|".join(f"{c}={_fmt_level(row[c])}" for c in factors))
            rows.append({c: row[c] for c in factors})
        for i in np.flatnonzero(ok):
            cond[i] = lut[tuple(keys.iloc[i][c] for c in factors)]
    else:
        # One condition: every non-blank trial (an event-aligned analysis).
        cond[~blank] = 0
        labels = ["all"]
        rows = [{}]
    return Conditions(primary, kind, secondary, labels, pd.DataFrame(rows), cond, blank, notes)


def _fmt_level(v) -> str:
    if isinstance(v, (float, np.floating)) and float(v).is_integer():
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return f"{v:g}"
    return str(v)


# ------------------------------------------------------------------ alignment

@dataclass
class Aligned:
    r: np.ndarray            # (trials, rois) baseline-subtracted mean response
    baseline: np.ndarray     # (trials, rois) mean of the baseline window
    traces: np.ndarray | None  # (trials, rois, time) baseline-subtracted, or None when too big
    t: np.ndarray            # (time,) seconds from onset
    keep: np.ndarray         # indices of the table rows kept (window inside the data)
    response_s: float
    baseline_s: float


def align(X: np.ndarray, t: np.ndarray, onsets: np.ndarray, baseline_s: float, response_s: float,
          post_s: float, frame_rate: float, max_elements: int = 150_000_000) -> Aligned:
    dt = 1.0 / frame_rate
    grid = np.arange(-baseline_s, response_s + post_s + dt / 2, dt)
    if len(grid) < 3:
        raise ValueError("aligned window shorter than 3 frames")
    keep = np.flatnonzero((onsets - baseline_s >= t[0]) & (onsets + response_s + post_s <= t[-1]))
    on = onsets[keep]
    R = X.shape[1]
    bmask = (grid < 0) & (grid >= -baseline_s - 1e-9)
    rmask = (grid >= 0) & (grid <= response_s + 1e-9)
    if bmask.sum() == 0 or rmask.sum() == 0:
        raise ValueError("baseline or response window contains no frames")
    frame_idx = np.arange(len(t), dtype=np.float64)
    store = len(on) * R * len(grid) <= max_elements
    traces = np.empty((len(on), R, len(grid)), dtype=np.float32) if store else None
    rr = np.empty((len(on), R), dtype=np.float32)
    bb = np.empty((len(on), R), dtype=np.float32)
    chunk = max(1, int(20_000_000 // max(1, R * len(grid))))
    for a in range(0, len(on), chunk):
        times = (on[a:a + chunk, None] + grid[None, :]).ravel()
        fi = np.interp(times, t, frame_idx)
        i0 = np.clip(np.floor(fi).astype(np.int64), 0, len(t) - 2)
        w = (fi - i0).astype(np.float32)[:, None]
        v = X[i0] * (1 - w) + X[i0 + 1] * w                      # (n*grid, R)
        v = v.reshape(-1, len(grid), R).transpose(0, 2, 1)       # (n, R, grid)
        base = v[:, :, bmask].mean(axis=2)
        bb[a:a + chunk] = base
        rr[a:a + chunk] = v[:, :, rmask].mean(axis=2) - base
        if store:
            traces[a:a + chunk] = v - base[:, :, None]
    return Aligned(rr, bb, traces, grid, keep, response_s, baseline_s)


# ------------------------------------------------------------------ statistics

def bh_fdr(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (NaN kept)."""
    p = np.asarray(p, dtype=float)
    out = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    m = ok.sum()
    if m == 0:
        return out
    pv = p[ok]
    order = np.argsort(pv)
    ranked = pv[order] * m / np.arange(1, m + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    res = np.empty(m)
    res[order] = np.minimum(q, 1.0)
    out[ok] = res
    return out


def condition_stats(r: np.ndarray, cond: np.ndarray, n_cond: int):
    """Per condition mean, SEM and trial count: (C, R), (C, R), (C,)."""
    R = r.shape[1]
    M = np.full((n_cond, R), np.nan, dtype=np.float64)
    S = np.full((n_cond, R), np.nan, dtype=np.float64)
    n = np.zeros(n_cond, dtype=np.int64)
    for c in range(n_cond):
        idx = cond == c
        n[c] = idx.sum()
        if n[c]:
            M[c] = r[idx].mean(axis=0)
            if n[c] > 1:
                S[c] = r[idx].std(axis=0, ddof=1) / math.sqrt(n[c])
    return M, S, n


def window_responses(X: np.ndarray, frame_rate: float, baseline_s: float, response_s: float,
                     frames: np.ndarray) -> np.ndarray:
    """Trial response with the onset at each of `frames`: mean of the response window (onset frame
    and the following response_s) minus the mean of the baseline_s before it. (len(frames), R)
    float32, NaN where the windows do not fit. Cumulative sums per block of ROIs keep the float64
    temporaries small."""
    T, R = X.shape
    wb = max(1, int(round(baseline_s * frame_rate)))
    wr = max(1, int(round(response_s * frame_rate)))
    out = np.full((len(frames), R), np.nan, dtype=np.float32)
    ok = (frames - wb >= 0) & (frames + wr <= T)
    i = frames[ok]
    for a in range(0, R, 64):
        cs = np.zeros((T + 1, min(64, R - a)), dtype=np.float64)
        np.cumsum(X[:, a:a + 64], axis=0, out=cs[1:])
        out[ok, a:a + 64] = (cs[i + wr] - cs[i]) / wr - (cs[i] - cs[i - wb]) / wb
    return out


def random_time_test(X: np.ndarray, frame_rate: float, baseline_s: float, response_s: float,
                     onset_frames: np.ndarray, groups: list[np.ndarray], n_shuffles: int,
                     rng: np.random.Generator, max_candidates: int = 20000):
    """Does the ROI respond to any stimulus condition more than it does at random times?

    groups: trial index arrays, one per stimulus condition (and per level of each stimulus factor,
    pooled over the others, which gives power when a condition has only a few trials).
    Statistic: the largest group mean of the frame-based trial response (window_responses at each
    trial's onset frame). Null: the same statistic with every trial's onset moved to a random frame
    of the recording (groups kept), n_shuffles times; random onsets are drawn from up to
    max_candidates evenly spaced frames. The null keeps each ROI's own autocorrelation and slow
    fluctuations, and taking the maximum over groups in both the statistic and the null corrects
    for testing every condition. Random frames include stimulus periods, which makes the test
    conservative for strongly driven ROIs.
    Returns (statistic (R,), p (R,))."""
    groups = [g for g in groups if len(g)]
    used = np.unique(np.concatenate(groups))
    pos = {k: i for i, k in enumerate(used)}
    G = np.zeros((len(groups), len(used)), dtype=np.float32)
    for k, g in enumerate(groups):
        G[k, [pos[i] for i in g]] = 1.0 / len(g)
    obs_r = window_responses(X, frame_rate, baseline_s, response_s, onset_frames[used])
    obs = (G @ np.nan_to_num(obs_r)).max(axis=0)
    step = max(1, int(np.ceil(X.shape[0] / max_candidates)))
    cand = window_responses(X, frame_rate, baseline_s, response_s, np.arange(0, X.shape[0], step))
    cand = cand[np.isfinite(cand[:, 0])]
    exceed = np.zeros(X.shape[1], dtype=np.int64)
    t_last = time.time()
    for k in range(n_shuffles):
        idx = rng.integers(0, len(cand), size=len(used))
        exceed += (G @ cand[idx]).max(axis=0) >= obs
        if time.time() - t_last > 60:
            log.info("responsiveness test: %d of %d shuffles", k + 1, n_shuffles)
            t_last = time.time()
    return obs, (1 + exceed) / (n_shuffles + 1)


def kruskal_selectivity(r: np.ndarray, cond: np.ndarray, n_cond: int) -> np.ndarray:
    """Kruskal-Wallis p across conditions per ROI (conditions with >= 2 trials)."""
    groups = [r[cond == c] for c in range(n_cond) if (cond == c).sum() >= 2]
    if len(groups) < 2:
        return np.full(r.shape[1], np.nan)
    with np.errstate(all="ignore"):
        try:
            p = stats.kruskal(*groups, axis=0, nan_policy="omit").pvalue
        except ValueError:
            p = np.full(r.shape[1], np.nan)
    return np.asarray(p, dtype=float)


def sparseness(M: np.ndarray, axis: int) -> np.ndarray:
    """Vinje & Gallant (2000) sparseness of rectified responses along an axis (0 = dense, 1 = sparse)."""
    x = np.clip(M, 0, None)
    n = x.shape[axis]
    if n < 2:
        return np.full(x.shape[1 - axis], np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        a = (x.sum(axis=axis) / n) ** 2 / ((x ** 2).sum(axis=axis) / n)
        s = (1 - a) / (1 - 1.0 / n)
    return s


def split_half_reliability(r: np.ndarray, cond: np.ndarray, n_cond: int) -> np.ndarray:
    """Correlation over conditions between the condition means of odd and even trials,
    Spearman-Brown corrected to the full number of trials."""
    odd = np.full((n_cond, r.shape[1]), np.nan)
    even = np.full((n_cond, r.shape[1]), np.nan)
    for c in range(n_cond):
        idx = np.flatnonzero(cond == c)
        if len(idx) >= 2:
            odd[c] = r[idx[0::2]].mean(axis=0)
            even[c] = r[idx[1::2]].mean(axis=0)
    ok = np.isfinite(odd[:, 0])
    if ok.sum() < 3:
        return np.full(r.shape[1], np.nan)
    rho = np.array([np.corrcoef(odd[ok, j], even[ok, j])[0, 1] if odd[ok, j].std() > 0 and even[ok, j].std() > 0
                    else np.nan for j in range(r.shape[1])])
    with np.errstate(invalid="ignore", divide="ignore"):
        return 2 * rho / (1 + rho)


def trial_reliability(traces: np.ndarray | None, cond: np.ndarray, pref: np.ndarray) -> np.ndarray:
    """Mean pairwise Pearson correlation between single-trial time courses at the preferred condition."""
    R = len(pref)
    out = np.full(R, np.nan)
    if traces is None:
        return out
    for j in range(R):
        idx = np.flatnonzero(cond == pref[j])
        if len(idx) < 2:
            continue
        tr = traces[idx, j, :].astype(np.float64)
        tr = tr - tr.mean(axis=1, keepdims=True)
        sd = np.sqrt((tr ** 2).sum(axis=1))
        if (sd == 0).any():
            continue
        c = (tr @ tr.T) / np.outer(sd, sd)
        out[j] = c[np.triu_indices(len(idx), 1)].mean()
    return out


def noise_residuals(r: np.ndarray, cond: np.ndarray, n_cond: int) -> np.ndarray:
    """Trial responses minus their condition mean, z-scored within each condition (>= 3 trials)."""
    E = []
    for c in range(n_cond):
        idx = cond == c
        if idx.sum() >= 3:
            x = r[idx] - r[idx].mean(axis=0)
            sd = x.std(axis=0, ddof=1)
            sd[sd == 0] = 1.0
            E.append(x / sd)
    return np.concatenate(E, axis=0) if E else np.empty((0, r.shape[1]))


# ------------------------------------------------------------------ tuning

def _nearest_level(levels: np.ndarray, target: float, period: float) -> int | None:
    d = np.abs((levels - target + period / 2) % period - period / 2)
    i = int(np.argmin(d))
    spacing = np.min(np.diff(np.sort(levels))) if len(levels) > 1 else period
    return i if d[i] <= spacing / 2 + 1e-6 else None


def angular_tuning(curve: np.ndarray, angles_deg: np.ndarray, pref_idx: np.ndarray, kind: str) -> dict:
    """Selectivity of tuning curves over angles (levels, rois).

    Peak-based (as the Allen Brain Observatory reports them): OSI = (R_pref - R_orth)/(R_pref + R_orth),
    R_orth the mean at pref +/- 90 deg; DSI = (R_pref - R_null)/(R_pref + R_null), null = pref + 180.
    Vector sum (global, Mazurek et al. 2014) on the rectified curve: gOSI = |sum R e^{2i theta}| / sum R,
    gDSI = |sum R e^{i theta}| / sum R; the preferred direction / orientation is the vector's angle."""
    th = np.deg2rad(angles_deg)[:, None]
    Rr = np.clip(curve, 0, None)
    tot = Rr.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        v1 = (Rr * np.exp(1j * th)).sum(axis=0)
        v2 = (Rr * np.exp(2j * th)).sum(axis=0)
        gdsi = np.abs(v1) / tot
        gosi = np.abs(v2) / tot
    pref_dir_vs = np.rad2deg(np.angle(v1)) % 360
    pref_ori_vs = (np.rad2deg(np.angle(v2)) / 2) % 180
    R = curve.shape[1]
    osi = np.full(R, np.nan)
    dsi = np.full(R, np.nan)
    period = 360.0 if kind == "direction" else 180.0
    for j in range(R):
        k = pref_idx[j]
        if k < 0 or not np.isfinite(curve[k, j]):
            continue
        rp = curve[k, j]
        orth = [_nearest_level(angles_deg, angles_deg[k] + d, period) for d in (90.0, -90.0)]
        orth = [o for o in orth if o is not None and np.isfinite(curve[o, j])]
        if orth:
            ro = float(np.mean([curve[o, j] for o in orth]))
            if rp + ro > 0:
                osi[j] = (rp - ro) / (rp + ro)
        if kind == "direction":
            nl = _nearest_level(angles_deg, angles_deg[k] + 180.0, 360.0)
            if nl is not None and np.isfinite(curve[nl, j]) and rp + curve[nl, j] > 0:
                dsi[j] = (rp - curve[nl, j]) / (rp + curve[nl, j])
    out = {"osi": osi, "gosi": gosi, "pref_ori_vs": pref_ori_vs}
    if kind == "direction":
        out.update({"dsi": dsi, "gdsi": gdsi, "pref_dir_vs": pref_dir_vs})
    else:
        # Orientation-only stimuli: with angles in [0, 180) the 2-theta vector is the orientation one.
        out.update({"dsi": np.full(R, np.nan), "gdsi": np.full(R, np.nan), "pref_dir_vs": np.full(R, np.nan)})
    return out


def categorical_selectivity(curve: np.ndarray) -> np.ndarray:
    """(R_pref - mean R_other) / (R_pref + mean R_other) on the rectified marginal curve."""
    x = np.clip(curve, 0, None)
    if x.shape[0] < 2:
        return np.full(x.shape[1], np.nan)
    k = np.nanargmax(np.where(np.isfinite(x), x, -np.inf), axis=0)
    rp = x[k, np.arange(x.shape[1])]
    others = (np.nansum(x, axis=0) - rp) / (np.isfinite(x).sum(axis=0) - 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        si = (rp - others) / (rp + others)
    return si
