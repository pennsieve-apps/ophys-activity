"""Figures (PNG, 150 dpi, session id in the title). Each function returns the caption."""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402

log = logging.getLogger("ophys-activity")

DPI = 150
BLUE = "#2a78d6"
ORANGE = "#eb6834"
GREY = "#9a9893"
INK = "#1f1f1e"
CYCLIC = "hsv"   # cyclic (0 and 360 deg get the same colour) and saturated everywhere, so no direction
                 # looks like the grey of untuned ROIs on a grey image (twilight passes through white)
plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "axes.edgecolor": "#6b6a66",
    "axes.spines.top": False, "axes.spines.right": False, "xtick.color": "#52514e", "ytick.color": "#52514e",
    "axes.labelcolor": INK, "text.color": INK, "figure.facecolor": "white", "savefig.facecolor": "white",
})


def _save(fig, path: Path):
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def raster(path: Path, sid: str, Zs: np.ndarray, tb: np.ndarray, trials: dict | None, signal: str,
           zoom_s: float = 120.0) -> str:
    """Zs: (rois sorted, bins) z-scored activity in Rastermap order."""
    N, T = Zs.shape
    # Superneurons for display only (as in the Rastermap figures): average groups of adjacent
    # sorted ROIs, about 100 rows in all; the sorting itself used every ROI.
    group = max(1, N // 100)
    n = (N // group) * group
    Zd = Zs[:n].reshape(-1, group, T).mean(axis=1) if group > 1 else Zs
    has_zoom = trials is not None and len(trials["start"]) > 0 and (tb[-1] - tb[0]) > zoom_s * 1.5
    ncols = 2 if has_zoom else 1
    fig = plt.figure(figsize=(11 if has_zoom else 8, 5.2))
    gs = fig.add_gridspec(2, ncols, height_ratios=[1, 0.06] if trials else [1, 0.001], hspace=0.05,
                          width_ratios=[1.6, 1] if has_zoom else [1], wspace=0.08)

    def panel(col, t0, t1, title):
        ax = fig.add_subplot(gs[0, col])
        sel = (tb >= t0) & (tb <= t1)
        img, ts = Zd[:, sel], tb[sel]
        # Average adjacent time bins so that the image has at most ~1200 columns (no aliasing).
        k = max(1, int(np.ceil(img.shape[1] / 1200)))
        if k > 1:
            m = (img.shape[1] // k) * k
            img = img[:, :m].reshape(img.shape[0], -1, k).mean(axis=2)
            ts = ts[:m].reshape(-1, k).mean(axis=1)
        vmax = 1.5 if group == 1 else 1.0
        ax.imshow(img, aspect="auto", cmap="gray_r", vmin=0, vmax=vmax, interpolation="nearest",
                  extent=(ts[0], ts[-1], n, 0))
        ax.set_title(title, loc="left")
        ax.set_ylabel(f"ROIs, Rastermap order{' (rows average ' + str(group) + ')' if group > 1 else ''}" if col == 0 else "")
        if col:
            ax.set_yticks([])
        ax.tick_params(labelbottom=False)
        if trials:
            axs = fig.add_subplot(gs[1, col], sharex=ax)
            for s, e, c in zip(trials["start"], trials["stop"], trials["colors"]):
                if e >= t0 and s <= t1:
                    axs.axvspan(s, e, color=c, lw=0)
            axs.set_yticks([])
            axs.set_xlim(ts[0], ts[-1])
            axs.set_xlabel("time (s, session clock)")
            for sp in ("left", "top", "right"):
                axs.spines[sp].set_visible(False)
        else:
            ax.tick_params(labelbottom=True)
            ax.set_xlabel("time (s, session clock)")
        return ax

    panel(0, tb[0], tb[-1], f"{sid}: population raster ({N} ROIs)")
    if has_zoom:
        t0 = float(trials["start"][0]) - 5
        panel(1, t0, t0 + zoom_s, f"first {zoom_s:.0f} s of stimulation")
    _save(fig, path)
    cap = (f"Activity of all {N} analysed ROIs ({signal}, z-scored, darker = more active) over time, with the ROIs "
           "sorted by Rastermap so that ROIs with similar activity sit next to each other. Bands of co-active ROIs "
           "and diagonal stripes (sequences) are structure in the population; uniform grey speckle is noise.")
    if trials:
        cap += (" The coloured bar under the raster marks the stimulus trials, coloured by "
                f"{trials['what']} (grey = blank); vertical stripes aligned with the bar are stimulus-locked responses.")
    return cap


def mean_responses(path: Path, sid: str, heat: np.ndarray, t: np.ndarray, label: str, response_s: float) -> str:
    """heat: (rois, time) trial-averaged response at each ROI's preferred condition, sorted."""
    fig, ax = plt.subplots(figsize=(5.5, 5))
    # Each row scaled to its own largest absolute value, so weak and strong ROIs are equally visible.
    scale = np.nanmax(np.abs(heat), axis=1, keepdims=True)
    scale[~np.isfinite(scale) | (scale == 0)] = 1.0
    heat = heat / scale
    im = ax.imshow(heat, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest",
                   extent=(t[0], t[-1], heat.shape[0], 0))
    ax.axvline(0, color=INK, lw=0.8)
    ax.axvline(response_s, color=INK, lw=0.8, ls="--")
    ax.set_xlabel("time from stimulus onset (s)")
    ax.set_ylabel(f"responsive ROIs, {label}")
    ax.set_title(f"{sid}: mean response at the preferred stimulus", loc="left")
    fig.colorbar(im, ax=ax, shrink=0.7, label="response / its maximum")
    _save(fig, path)
    return ("Trial-averaged response of each responsive ROI to its preferred stimulus, after subtracting the "
            "pre-stimulus baseline, each row scaled to its own maximum (red = above baseline). Solid line: stimulus onset; dashed: end of the response "
            f"window. ROIs are sorted by {label}. A clean band of red right after onset in most rows means the "
            "population is visually driven; slow ramps or red before onset suggest drift or anticipation.")


def tuning_curves(path: Path, sid: str, curves: list[dict], factor: str, angular: bool, ranking: str) -> str:
    n = len(curves)
    cols = min(4, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.6 * cols, 2.0 * rows), squeeze=False)
    for ax, c in zip(axes.flat, curves):
        x = np.arange(len(c["levels"]))
        ax.axhline(0, color=GREY, lw=0.6)
        if c.get("blank") is not None and np.isfinite(c["blank"]):
            ax.axhline(c["blank"], color=ORANGE, lw=0.8, ls="--")
        ax.errorbar(x, c["mean"], yerr=c["sem"], color=BLUE, lw=1.5, marker="o", ms=3, capsize=1.5, elinewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(c["levels"], rotation=90 if len(x) > 8 else 0, fontsize=6)
        ax.set_title(c["title"], fontsize=7, loc="left")
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    for ax in axes[-1]:
        ax.set_xlabel(factor + (" (deg)" if angular else ""))
    for ax in axes[:, 0]:
        ax.set_ylabel("response")
    fig.suptitle(f"{sid}: tuning curves of {n} ROIs ({ranking})", x=0.01, ha="left", fontsize=9)
    fig.tight_layout()
    _save(fig, path)
    cap = (f"Mean baseline-subtracted response (± SEM over trials) of {n} ROIs ({ranking}), per value "
           f"of {factor}")
    if any(c.get("at_pref") for c in curves):
        cap += (", at each ROI's preferred value of the other stimulus parameters (named under the ROI), or pooled "
                "over them where a condition there had too few trials")
    cap += (". Dashed orange: response to blank trials. A single sharp peak is a selective ROI; two peaks 180 deg "
            "apart are orientation- but not direction-selective." if angular else
            ". Dashed orange: response to blank trials. One condition far above the others is a selective ROI.")
    return cap


def polar(path: Path, sid: str, curves: list[dict]) -> str:
    n = len(curves)
    cols = min(4, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.3 * cols, 2.4 * rows), subplot_kw={"projection": "polar"},
                             squeeze=False)
    for ax, c in zip(axes.flat, curves):
        th = np.deg2rad(np.asarray(c["angles"], float))
        r = np.clip(np.asarray(c["mean"], float), 0, None)
        ax.plot(np.r_[th, th[:1]], np.r_[r, r[:1]], color=BLUE, lw=1.5, marker="o", ms=2.5)
        ax.fill(np.r_[th, th[:1]], np.r_[r, r[:1]], color=BLUE, alpha=0.15, lw=0)
        if np.isfinite(c.get("pref_vs", np.nan)):
            ax.annotate("", xy=(np.deg2rad(c["pref_vs"]), np.nanmax(r) * c.get("vs_len", 1.0)), xytext=(0, 0),
                        arrowprops={"arrowstyle": "->", "color": ORANGE, "lw": 1.5})
        ax.set_yticklabels([])
        ax.tick_params(labelsize=6, pad=0)
        ax.set_title(c["title"], fontsize=7, pad=8)
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    fig.suptitle(f"{sid}: polar tuning", x=0.01, ha="left", fontsize=9)
    fig.tight_layout()
    _save(fig, path)
    return ("The same ROIs as polar plots: distance from the centre is the response (negative responses drawn as 0) "
            "to each direction of motion. The orange arrow is the vector-sum preferred direction, its length the "
            "global direction selectivity (gDSI): long arrows point at the direction the ROI prefers.")


def osi_dsi(path: Path, sid: str, gosi, gdsi, osi, dsi, responsive, kind: str) -> str:
    panels = [("gOSI (vector sum)", gosi), ("OSI (peak)", osi)]
    if kind == "direction":
        panels += [("gDSI (vector sum)", gdsi), ("DSI (peak)", dsi)]
    fig, axes = plt.subplots(1, len(panels), figsize=(2.6 * len(panels), 2.4))
    bins = np.linspace(0, 1.5, 31)
    for ax, (name, v) in zip(np.atleast_1d(axes), panels):
        v = np.asarray(v, float)
        a = v[responsive & np.isfinite(v)]
        b = v[~responsive & np.isfinite(v)]
        ax.hist([a, b], bins=bins, stacked=True, color=[BLUE, GREY], label=["responsive", "not responsive"],
                edgecolor="white", linewidth=0.4)
        if len(a):
            ax.axvline(np.median(a), color=INK, lw=0.8, ls="--")
        ax.set_xlabel(name)
        ax.set_title(f"median {np.median(a):.2f} (n={len(a)})" if len(a) else "no responsive ROIs", fontsize=7, loc="left")
    np.atleast_1d(axes)[0].set_ylabel("ROIs")
    np.atleast_1d(axes)[0].legend(frameon=False, fontsize=6)
    fig.suptitle(f"{sid}: orientation and direction selectivity", x=0.01, ha="left", fontsize=9, y=1.04)
    _save(fig, path)
    return ("Distributions of orientation (OSI) and direction (DSI) selectivity: blue = ROIs that respond to the "
            "stimulus, grey = the rest (their indices are mostly noise). 0 = no preference, 1 = responds to one "
            "orientation/direction only. Vector-sum indices (gOSI, gDSI) use the whole tuning curve and are robust "
            "to noise; peak indices compare the best direction with the orthogonal / opposite one, as the Allen "
            "Brain Observatory reports them, and can exceed 1 when those responses are negative. Dashed: median of "
            "responsive ROIs.")


def roi_map(path: Path, sid: str, masks, shape, image, values, valid, mode: str, image_name: str | None) -> str:
    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    if image is not None:
        lo, hi = np.percentile(image, [1, 99.5])
        ax.imshow(image, cmap="gray", vmin=lo, vmax=hi * 1.1, interpolation="nearest")
    else:
        ax.imshow(np.zeros(shape), cmap="gray", vmin=0, vmax=1)
    rgba = np.zeros(shape + (4,), dtype=np.float32)
    if mode == "direction" or mode == "orientation":
        period = 360.0 if mode == "direction" else 180.0
        cmap = plt.get_cmap(CYCLIC)
        norm = Normalize(0, period)
    else:
        cmap = plt.get_cmap("viridis")
        fin = np.asarray(values, float)[np.asarray(valid, bool)]
        norm = Normalize(*(np.nanpercentile(fin, [2, 98]) if len(fin) else (0, 1)))
    for (rr, cc, w), v, ok in zip(masks, values, valid):
        m = (rr >= 0) & (rr < shape[0]) & (cc >= 0) & (cc < shape[1])
        rr, cc = rr[m], cc[m]
        if not len(rr):
            continue
        col = cmap(norm(v)) if ok and np.isfinite(v) else (0.75, 0.75, 0.75, 1.0)
        rgba[rr, cc, :3] = col[:3]
        rgba[rr, cc, 3] = 0.85 if ok else 0.35
    ax.imshow(rgba, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    if mode in ("direction", "orientation"):
        what = "preferred direction (deg)" if mode == "direction" else "preferred orientation (deg)"
    else:
        what = mode
    fig.colorbar(sm, ax=ax, shrink=0.7, label=what)
    ax.set_title(f"{sid}: ROI map coloured by {what.split(' (')[0]}", loc="left")
    _save(fig, path)
    bg = f" on the {image_name.replace('_', ' ')} image" if image_name else ""
    if mode in ("direction", "orientation"):
        return (f"Each analysed ROI drawn at its place in the field of view{bg}, coloured by its vector-sum "
                f"{what.split(' (')[0]} (the colour wheel wraps around: 0 and {360 if mode == 'direction' else 180} "
                "deg are the same colour); grey ROIs are not tuned. In mouse V1, preferences are mixed "
                "'salt-and-pepper' rather than in maps; clusters of one colour would suggest shared input or a "
                "contaminating neuropil signal.")
    return (f"Each analysed ROI drawn at its place in the field of view{bg}, coloured by {what}; grey ROIs have no "
            "value. Look for spatial clustering of activity and for ROIs at the edge of the field of view.")


def correlations(path: Path, sid: str, mats: list[tuple[str, np.ndarray]], order: np.ndarray, order_label: str) -> str:
    fig, axes = plt.subplots(1, len(mats), figsize=(4.0 * len(mats), 3.6), squeeze=False)
    for ax, (name, C) in zip(axes[0], mats):
        Cs = C[np.ix_(order, order)].copy()
        np.fill_diagonal(Cs, np.nan)
        lim = np.nanpercentile(np.abs(Cs), 99) if np.isfinite(Cs).any() else 1
        im = ax.imshow(Cs, cmap="RdBu_r", vmin=-lim, vmax=lim, interpolation="nearest")
        ax.set_title(name, loc="left")
        ax.set_xlabel(f"ROI ({order_label})")
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, shrink=0.75, label="Pearson r")
    axes[0][0].set_ylabel(f"ROI ({order_label})")
    fig.suptitle(f"{sid}: pairwise correlations", x=0.01, ha="left", fontsize=9, y=1.02)
    fig.tight_layout()
    _save(fig, path)
    names = [m[0] for m in mats]
    cap = f"Pairwise Pearson correlations between ROIs, ordered by {order_label} (red = positive, blue = negative). "
    if any("signal" in n for n in names):
        cap += ("Signal correlation: how similar two ROIs' tuning is (correlation of their mean responses over "
                "stimulus conditions). Noise correlation: how much their trial-to-trial fluctuations around those means "
                "co-vary, i.e. shared variability not explained by the stimulus; usually higher for ROIs with similar "
                "tuning. Positive noise correlation across almost all pairs points to a fluctuation shared by the whole "
                "population (running, arousal, slow drift) rather than to specific connections. ")
    if any("activity" in n for n in names):
        cap += ("Activity correlation: correlation of the binned activity over the whole recording; blocks along the "
                "diagonal are groups of co-active ROIs. Neighbouring ROIs that share neuropil or overlap also correlate.")
    return cap.strip()


def assemblies(path: Path, sid: str, res: dict, tb: np.ndarray, roi_order: np.ndarray, max_show: int = 8) -> str:
    n = res["n"]
    k = min(n, max_show)
    fig = plt.figure(figsize=(10, 1.0 + 0.9 * k))
    gs = fig.add_gridspec(k, 2, width_ratios=[1, 3], hspace=0.35, wspace=0.12)
    for i in range(k):
        w = res["patterns"][i][roi_order]
        ax = fig.add_subplot(gs[i, 0])
        mem = np.zeros(len(w), bool)
        inv = np.empty_like(roi_order)
        inv[roi_order] = np.arange(len(roi_order))
        mem[inv[res["members"][i]]] = True
        ax.bar(np.arange(len(w)), w, color=np.where(mem, ORANGE, GREY), width=1.0)
        ax.set_xticks([])
        ax.set_ylabel(f"A{i + 1}", rotation=0, labelpad=12, va="center")
        if i == 0:
            ax.set_title("weights (orange = members)", loc="left")
        ax2 = fig.add_subplot(gs[i, 1])
        ax2.plot(tb, res["activation"][i], color=BLUE, lw=0.6)
        if i == 0:
            ax2.set_title("activation strength", loc="left")
        if i < k - 1:
            ax2.set_xticks([])
        else:
            ax2.set_xlabel("time (s, session clock)")
            ax.set_xlabel("ROI (Rastermap order)")
    fig.suptitle(f"{sid}: {n} cell assemblies" + (f" (first {k} shown)" if n > k else ""), x=0.01, ha="left",
                 fontsize=9)
    _save(fig, path)
    return ("Cell assemblies (groups of ROIs that are repeatedly active together), found by PCA with the "
            "Marchenko-Pastur bound plus ICA (Lopes-dos-Santos et al. 2013). Left: each assembly's weight per ROI; "
            "orange ROIs are its members. Right: how strongly the assembly's pattern is expressed over time; sharp "
            "peaks are moments when its members fire together. Assemblies made of neighbouring ROIs in the field "
            "of view can also come from shared neuropil or motion.")
