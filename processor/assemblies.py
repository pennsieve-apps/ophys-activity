"""
Cell assemblies: groups of ROIs that are co-active more often than chance.

Method of Lopes-dos-Santos, Ribeiro & Tort (2013, J Neurosci Methods), "PCA + ICA":
  1. z-score each ROI's binned activity (bins x ROIs);
  2. the number of assemblies is the number of eigenvalues of the ROI correlation matrix above
     the largest eigenvalue expected for independent ROIs. Lopes-dos-Santos et al. use the
     Marchenko-Pastur upper bound (1 + sqrt(n_rois / n_bins))^2 (ASSEMBLY_NULL=mp), which assumes
     independent time bins. Calcium signals are strongly autocorrelated (a transient lasts several
     bins), which inflates the eigenvalues of even independent ROIs and makes the bound find
     dozens of spurious assemblies. So by default (ASSEMBLY_NULL=shift) the bound is the largest
     eigenvalue over 20 surrogates in which every ROI's time series is circularly shifted by an
     independent random amount: each ROI keeps its own autocorrelation, the co-activation between
     ROIs is destroyed. (Our adaptation; the higher of the two bounds is used.)
  3. independent component analysis (FastICA) of the activity projected onto those principal
     components gives one weight vector (pattern) per assembly, scaled to unit length and signed so
     that its largest weight is positive;
  4. activation strength over time: R_k(t) = z(t)' P_k z(t) with P_k the outer product of the
     pattern and its diagonal set to zero, so one ROI alone cannot activate an assembly.
Members are the ROIs whose weight is an outlier of the pattern, robust z > 3 (median + 3 x 1.4826 x
MAD; our convention, the weights themselves are saved). Patterns with fewer than 2 members are dropped (one ROI's own
variance, not co-activation).
"""
from __future__ import annotations

import logging
import warnings

import numpy as np

log = logging.getLogger("ophys-activity")


def _member_threshold(w: np.ndarray) -> float:
    """Robust z > 3: median + 3 * 1.4826 * MAD of the weights. (Mean + 2 SD misses members when an
    assembly is a large fraction of the ROIs, because the members themselves inflate the SD.)"""
    med = np.median(w)
    return float(med + 3 * 1.4826 * np.median(np.abs(w - med)))


def detect_assemblies(Z: np.ndarray, seed: int, null: str = "shift", n_surrogates: int = 20,
                      max_assemblies: int = 50) -> dict:
    """Z: (bins, rois) z-scored activity. Returns n, patterns (n, rois), members (list of index
    arrays), activation (n, bins), eigenvalues, threshold."""
    T, N = Z.shape
    empty = {"n": 0, "patterns": np.zeros((0, N), np.float32), "members": [], "activation": np.zeros((0, T), np.float32),
             "eigenvalues": np.array([]), "mp_threshold": float("nan"),
             "threshold": float("nan"), "null": null}
    if N < 3 or T < N:
        return empty
    C = (Z.T @ Z) / T
    evals, evecs = np.linalg.eigh(C)
    order = np.argsort(evals)[::-1]
    evals, evecs = evals[order], evecs[:, order]
    lam_max = (1 + np.sqrt(N / T)) ** 2
    lam_mp = lam_max
    if null == "shift":
        rng = np.random.default_rng(seed)
        top = []
        rows = np.arange(T)
        for _ in range(n_surrogates):
            shifts = rng.integers(0, T, size=N)
            Zs = Z[(rows[:, None] + shifts[None, :]) % T, np.arange(N)[None, :]]
            top.append(np.linalg.eigvalsh((Zs.T @ Zs) / T)[-1])
        lam_max = max(lam_max, float(np.max(top)))
    n = int((evals > lam_max).sum())
    empty["eigenvalues"], empty["mp_threshold"], empty["threshold"] = evals, float(lam_mp), float(lam_max)
    if n == 0:
        return empty
    n = min(n, max_assemblies)
    P = evecs[:, :n]
    from sklearn.decomposition import FastICA

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ica = FastICA(n_components=n, random_state=seed, whiten="unit-variance", max_iter=2000, tol=1e-4)
        ica.fit(Z @ P)
    W = ica.components_ @ P.T                    # (n, N) patterns in ROI space
    W /= np.linalg.norm(W, axis=1, keepdims=True)
    flip = np.abs(W.max(axis=1)) < np.abs(W.min(axis=1))
    W[flip] *= -1
    members = [np.flatnonzero(w > _member_threshold(w)) for w in W]
    # A pattern dominated by a single ROI is that ROI's own variance, not co-activation.
    keep = [i for i, m in enumerate(members) if len(m) >= 2]
    W, members = W[keep], [members[i] for i in keep]
    n_single = n - len(keep)
    n = len(keep)
    if n == 0:
        empty["n_single"] = n_single
        return empty
    proj = Z @ W.T                                # (T, n)
    act = proj ** 2 - (Z ** 2) @ (W.T ** 2)       # zero-diagonal quadratic form
    # Number the assemblies by the variance of their activation (largest first).
    order = np.argsort(-act.var(axis=0))
    return {"n": n, "patterns": W[order].astype(np.float32), "members": [members[i] for i in order],
            "activation": act[:, order].T.astype(np.float32), "eigenvalues": evals, "mp_threshold": float(lam_mp),
            "threshold": float(lam_max), "null": null, "n_single": n_single}
