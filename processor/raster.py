"""
Rastermap sorting of the population (Stringer et al. 2024, Nat Neurosci; MouseLand/rastermap,
GPL-3.0): ROIs ordered so that neighbours in the raster have similar activity, which turns a
noisy population raster into visible sequences and co-active groups.
"""
from __future__ import annotations

import logging
import warnings

import numpy as np

log = logging.getLogger("ophys-activity")


def rastermap_sort(Z: np.ndarray, seed: int) -> tuple[np.ndarray, dict]:
    """Z: (bins, rois) z-scored activity. Returns (isort, settings). Falls back to the order of
    the first principal component when Rastermap cannot run (fewer than 3 ROIs)."""
    T, N = Z.shape
    if N < 3 or T < 10:
        return np.arange(N), {"method": "none (fewer than 3 ROIs)"}
    from rastermap import Rastermap

    # Rastermap's defaults (100 clusters, 200 PCs) assume hundreds of neurons or more. With fewer,
    # fewer clusters (clustering is skipped by Rastermap itself at <= 50 ROIs).
    n_pcs = int(min(200, N - 1, T - 1))
    n_clusters = None if N <= 50 else int(min(100, max(10, N // 3)))
    settings = {"method": "rastermap", "n_PCs": n_pcs, "n_clusters": n_clusters, "locality": 0.0,
                "time_lag_window": 0, "random_state": seed}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = Rastermap(n_clusters=n_clusters, n_PCs=n_pcs, locality=0.0, time_lag_window=0,
                          verbose=False, random_state=seed)
        model.fit(np.ascontiguousarray(Z.T, dtype=np.float32), compute_X_embedding=False)
    return np.asarray(model.isort, dtype=np.int64), settings
