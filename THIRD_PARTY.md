# Third-party software

This app's own code is licensed **GPL-3.0-or-later** (see `LICENSE`), because the image imports
**Rastermap**, which is GPL-3.0, in-process (`processor/raster.py` calls `rastermap.Rastermap`).
The other bundled components are under permissive licences compatible with that. The image is
built from public sources: the upstream repositories below, plus this repository's Dockerfile
and code, which is the corresponding source for the GPL.

Installed from PyPI at the pinned versions in `requirements.txt`; licences read from each
installed package's metadata and LICENSE file.

| Component | Version | What it does here | Authors / copyright holder | Licence | Source |
|---|---|---|---|---|---|
| Rastermap | 1.0 | Sorting of ROIs for the population raster. | Carsen Stringer and Marius Pachitariu (MouseLand, HHMI Janelia Research Campus) | GPL-3.0 | https://github.com/MouseLand/rastermap |
| scikit-learn | 1.9.1 | FastICA for cell assemblies; used by Rastermap (PCA, k-means). | The scikit-learn developers | BSD-3-Clause | https://github.com/scikit-learn/scikit-learn |
| NumPy | 2.5.4 | Arrays, linear algebra. | NumPy Developers | BSD-3-Clause (bundled parts: 0BSD, MIT, Zlib, CC0-1.0) | https://numpy.org |
| SciPy | 1.18.1 | Kruskal-Wallis test, statistics. | Enthought, Inc.; SciPy Developers | BSD-3-Clause | https://scipy.org |
| pandas | 3.1.1 | Tables. | AQR Capital Management, LLC, Lambda Foundry, Inc. and PyData Development Team | BSD-3-Clause | https://pandas.pydata.org |
| h5py | 3.16.0 | Reading NWB (HDF5) files. | Andrew Collette and contributors | BSD-3-Clause | https://www.h5py.org |
| Matplotlib | 3.11.2 | Figures. | Matplotlib Development Team | Matplotlib licence (PSF-based, BSD-compatible) | https://matplotlib.org |
| numba 0.68.1, llvmlite 0.50.0 | | JIT used by Rastermap. | Anaconda, Inc. and contributors | BSD-2-Clause; llvmlite BSD-2-Clause AND Apache-2.0 WITH LLVM-exception | https://numba.pydata.org |
| natsort 8.4.0, tqdm 4.70.1 | | Dependencies of Rastermap. | Seth M. Morton; tqdm developers | MIT; MPL-2.0 AND MIT (unmodified) | PyPI |
| joblib 1.6.0, threadpoolctl 3.8.0, narwhals 2.28.0 | | Dependencies of scikit-learn. | The joblib developers; threadpoolctl contributors; Marco Gorelli | BSD-3-Clause; BSD-3-Clause; MIT | PyPI |
| contourpy 1.4.0, cycler 0.12.1, fonttools 4.68.0, kiwisolver 1.5.1, pillow 13.0.0, pyparsing 3.3.3, packaging 26.3, python-dateutil 4.0.0 | | Dependencies of Matplotlib. | their authors | BSD-3-Clause; BSD-3-Clause; MIT; BSD-3-Clause; MIT-CMU; MIT; Apache-2.0 OR BSD-2-Clause; BSD-3-Clause | PyPI |
| cloudpickle 3.1.2, tzdata 2026.6 | | Transitive dependencies. | Cloudpickle contributors; Paul Ganssle | BSD-3-Clause; Apache-2.0 | PyPI |
| python:3.12.11-slim-bookworm | | Base image (Debian 12 + CPython 3.12). | Python Software Foundation; Debian | PSF; Debian packages under their own free licences | https://hub.docker.com/_/python |

Test-only (not in the image): pytest (MIT), pynwb 4.2.0 (BSD-3-Clause, used to check that a
file written by pynwb itself is read correctly).

## Methods implemented here (no code copied)

- Cell assemblies: the PCA + ICA method of Lopes-dos-Santos, Ribeiro & Tort (2013), written
  from the paper's description with scikit-learn's FastICA.
- Orientation / direction selectivity: vector-sum indices as in Mazurek, Kager & Van Hooser
  (2014); peak indices as the Allen Brain Observatory reports them (de Vries et al. 2020).
- Sparseness: Vinje & Gallant (2000). FDR: Benjamini & Hochberg (1995).

## Not bundled

- **Demo data** (not in the image or the repository): Allen Brain Observatory Visual Coding 2p,
  DANDI:000728 (CC-BY-4.0); hippocampal CA1 one-photon imaging in wild-type and 5xFAD mice,
  DANDI:002047 (CC-BY-4.0).
- The Allen Brain Observatory's published cell metrics, used only by `scripts/compare_allen.py`
  at development time (fetched from api.brain-map.org).
