# ophys-activity

This app packages **Rastermap**, developed by Carsen Stringer and Marius Pachitariu at the HHMI
Janelia Research Campus (MouseLand, https://github.com/MouseLand/rastermap; Stringer et al. 2024,
*Nature Neuroscience*, https://doi.org/10.1038/s41593-024-01783-4), for the sorted population
raster, and implements published analysis methods: cell-assembly detection by PCA and ICA from
Vítor Lopes-dos-Santos, Sidarta Ribeiro and Adriano Tort (Brain Institute, Federal University of
Rio Grande do Norte; *J Neurosci Methods* 2013), vector-sum orientation and direction selectivity
from Mark Mazurek, Marisa Kager and Stephen Van Hooser (Brandeis University and Metropolitan State
University of Denver; *Front Neural Circuits* 2014), and the peak-based tuning metrics of the
Allen Brain Observatory (de Vries et al. 2020). **Pennsieve packages these methods; it did not develop them.** The demo data are the Allen
Institute's Visual Coding 2p survey (DANDI:000728, de Vries et al. 2020) and hippocampal CA1
one-photon recordings shared by Viktor Plusnin and Nikita Pospelov (DANDI:002047), both CC-BY-4.0.

**What does a population of imaged neurons do, and what does it respond to?** For every
recording the app summarises activity (event rates), sorts the population with Rastermap into
the iconic raster, measures pairwise correlations and finds **cell assemblies** (groups of ROIs
that are repeatedly co-active). When a stimulus or trials table is available it aligns the
activity to it and asks, per ROI: does it respond (against what it does at random times, with a
false-discovery-rate correction)? To what (tuning curve, preferred direction, **OSI / DSI**, or a
selectivity index for any categorical condition such as odours)? How reliably, trial after
trial? Which ROIs share tuning (**signal correlation**) or trial-to-trial fluctuations (**noise
correlation**)? It writes the **trials × ROIs response matrix** that decoding starts from.

For a neuroscientist: point it at an NWB file with ROI traces (our `ophys-traces` output, or any
DANDI ophys file) and, if you have one, the stimulus table; the figures come with captions that
say what to look at. For a computational analyst: everything behind the figures is in tables
(`rois.csv`, `tuning.csv`, `trials.csv`) and arrays (`responses.npz`, `correlations.npz`,
`assemblies.npz`), with every choice listed below and in `provenance.json`.

It is the third step of the **Two-photon Calcium Imaging** workflow:
`suite2p → ophys-traces → ophys-activity → neuro-report`, and runs as well on any NWB file with
`processing/ophys` (it needs no other step).

## Method

Per session (one traces NWB), streaming one session at a time:

1. **Signal.** The `RoiResponseSeries` under `processing/` (`SIGNAL=auto`): `Deconvolved`
   (ophys-traces' CASCADE spike rate; an OASIS / event series only if nothing else is there) >
   `DfOverF` > `Fluorescence` (never `Neuropil`, residual or background series) > event series
   (Allen's `DfOverFEvents`, CNMF `events`). Timestamps from the series (`timestamps`, else
   `starting_time` + `rate`), on the session clock, never re-zeroed; `T_START`/`T_STOP` cut a
   window. ROIs with `accepted = false` (or `iscell` probability < 0.5 when there is no
   `accepted` column) are left out (`ACCEPTED_ONLY`).
2. **Activity summary** (every recording). Per ROI the mean, the robust noise level (1.4826 ×
   MAD) and **events**: upward crossings of median + `EVENT_Z` × noise, re-armed only after the
   trace falls below median + `EVENT_Z`/2 × noise, so one transient is one event (events per
   second over the window). For deconvolved spike rates `mean_signal` is the mean inferred rate.
3. **Population** (every recording), on activity averaged in `BIN_S` bins and z-scored per ROI:
   - **Rastermap** (Stringer et al. 2024) sorts the ROIs so that neighbours have similar activity
     (Rastermap's defaults, 100 clusters and 200 PCs, scaled down below 300 ROIs; `locality` 0).
   - **Activity correlation**: Pearson correlation of every ROI pair.
   - **Cell assemblies** (Lopes-dos-Santos, Ribeiro & Tort 2013): the number of assemblies is
     the number of eigenvalues of the ROI correlation matrix above a bound for independent
     ROIs; FastICA in the space of those components gives one weight pattern per assembly;
     members are ROIs with robust z > 3 of the weights; activation strength over time is the
     pattern's quadratic form with the diagonal removed. The paper's bound is Marchenko-Pastur
     (`ASSEMBLY_NULL=mp`), which assumes independent time bins; calcium signals are strongly
     autocorrelated, and on the Allen session it finds 50+ "assemblies". The default
     (`ASSEMBLY_NULL=shift`, our adaptation) uses the largest eigenvalue of 20 surrogates in
     which each ROI is circularly shifted by its own random lag (autocorrelation kept,
     co-activation destroyed). Patterns with fewer than 2 members are dropped.
4. **Stimulus table** (when there is one). `STIM_TABLE=auto` takes, from the `session` port
   first, then a stimulus CSV, then the traces file itself: `intervals/trials`, else the
   `TimeIntervals` table under `intervals/` or `stimulus/presentation/` with a condition column
   and the most rows (never `epochs` or `spontaneous*`); a name picks any TimeIntervals table,
   including behavioural ones under `processing/` (event-aligned analysis, e.g. `locomotion`).
   **Conditions** (`CONDITION=auto`): an angle column (`direction`, `orientation`, `ori`, ...)
   is the primary factor, with grating parameters (temporal / spatial frequency, contrast,
   size, phase) as secondary factors; otherwise a column named `condition`, `stimulus`,
   `trial_type`, `odor`, ...; columns with more than `MAX_LEVELS` values are not conditions.
   Angles spanning more than 180° are directions of motion, else orientations. **Blank** trials
   are those with a `*blank*` column true or no primary value. A condition with fewer than half
   the median number of trials per condition (at least 2) is left out (it would win "preferred"
   by noise alone).
5. **Aligned responses.** Every trial is interpolated onto a time grid at the frame rate from
   −`BASELINE_S` to the response window + `POST_S`; the mean of the baseline window is
   subtracted (per trial, per ROI). The **trial response** is the mean over the response window
   [0, `RESPONSE_S`] (default: the median trial duration) minus the baseline. These form
   `responses` (trials × ROIs) and `aligned` (trials × ROIs × time) in `responses.npz`.
6. **Responsiveness.** Statistic: the largest mean trial response over the conditions (and over
   each level of each factor pooled over the others, which gives power when conditions have few
   trials). Null: the same statistic after moving every trial's onset to a random frame of the
   recording, `N_SHUFFLES` times; this keeps each ROI's autocorrelation and slow drift, and the
   maximum over conditions in both corrects for testing every condition. Random frames include
   stimulus periods, so the test is conservative for strongly driven ROIs. p = (1 + #null ≥
   observed)/(N + 1), then Benjamini-Hochberg over ROIs: **responsive** = q < `FDR`.
   When the table has ≥ 10 blank trials, the app also reports a blank-sweep criterion in the
   spirit of de Vries et al. 2020 (`responsive_blank`: at least 25 % of the trials of the
   preferred condition above the 95th percentile of the blank-trial responses; our
   implementation, for comparison).
7. **Tuning.** Condition means ± SEM; the **preferred condition** is the largest mean.
   **Selective** (`tuned`) = responsive and Kruskal-Wallis across conditions q < `FDR`.
   Angles: the tuning curve over the primary factor at the ROI's preferred value of the other
   factors (as the Allen Brain Observatory does; pooled over them when a condition there is
   missing), then **vector-sum** gOSI = |Σ R e^{2iθ}| / Σ R and gDSI = |Σ R e^{iθ}| / Σ R on the
   rectified curve with the preferred direction / orientation at the vector's angle (Mazurek et
   al. 2014), and **peak** OSI = (R_pref − R_orth)/(R_pref + R_orth), DSI = (R_pref −
   R_null)/(R_pref + R_null) as reported by the Allen Brain Observatory (orth = mean of ±90°,
   null = +180°; these exceed 1 when R_orth or R_null is negative). Categorical conditions: the
   **selectivity index** (R_pref − mean R_other)/(R_pref + mean R_other) on the rectified
   condition means.
8. **Reliability.** Split-half: correlation over conditions between the condition means of odd
   and even trials, Spearman-Brown corrected. Trial-to-trial: the mean pairwise correlation of
   single-trial time courses at the preferred condition (the Allen `reliability_dg` idea).
9. **Correlations** (Cohen & Kohn 2011). **Signal**: correlation over conditions of two ROIs'
   condition means. **Noise**: correlation of their trial responses after subtracting each
   condition's mean and dividing by its SD (conditions with ≥ 3 trials). Means are over the
   responsive ROIs (all ROIs when fewer than 2 respond), plus the correlation between signal and
   noise correlation across pairs.
10. **Sparseness** (Vinje & Gallant 2000) of the rectified condition means: **lifetime** (per ROI,
    over conditions) and **population** (per condition, over ROIs; the median is reported).

## Ports

| Port | Kind | Files | Notes |
|---|---|---|---|
| `traces` | input (subfolder) | `*.nwb` with `processing/*/RoiResponseSeries` | One session per file: ophys-traces' `<session_id>_ophys.nwb` or any DANDI ophys NWB. |
| `session` | input (subfolder, optional) | an NWB with `TimeIntervals`, or `<session_id>_stimulus.csv` (`start_time, stop_time, condition`, ...; or a `session_id` column) | Matched to the traces by session_id; with a single session any stimulus file applies. Without it, tables in the traces NWB are used. |
| `metadata` | input (subfolder, optional) | `sessions.csv` | `session_id, path, subject_id, condition, group, ...`; carried into the tables and summary.json. |
| `activity` | output | `ophys-activity/...` (below) | The only output, so it is the whole of `OUTPUT_DIR`. |

Without links the same files are found by name and content anywhere under `INPUT_DIR`
(symlinks followed). Session ids follow conventions.md (sessions.csv, else `sub-X_ses-Y` in the
file or folder name, else the file stem).

## Parameters

| Parameter | Default | What it does |
|---|---|---|
| `SIGNAL` | `auto` | `auto`, `deconvolved`, `dff`, `fluorescence` or `events` (step 1). |
| `SERIES` | (empty) | Pick the RoiResponseSeries whose NWB path contains this text. |
| `ACCEPTED_ONLY` | `true` | Only ROIs with `accepted = true` (or iscell ≥ 0.5). |
| `T_START`, `T_STOP` | (empty) | Analysed window in session-clock seconds; empty = the whole recording. |
| `STIM_TABLE` | `auto` | `auto`, a TimeIntervals table name, or `none` (spontaneous analysis only). |
| `CONDITION` | `auto` | Condition column(s), primary first, comma-separated; `none` = all trials one condition. |
| `MAX_LEVELS` | `24` | Columns with more distinct values are not conditions. |
| `MIN_TRIALS` | `5` | Fewest trials (in the window) for a stimulus analysis. |
| `BASELINE_S` | `1.0` | Pre-stimulus baseline (s). |
| `RESPONSE_S` | `0` | Response window (s); 0 = median trial duration. |
| `POST_S` | `1.0` | Seconds after the response window in the aligned traces. |
| `N_SHUFFLES` | `10000` | Random-time shuffles; the smallest p is 1/(N+1), so N must be large for BH over hundreds of ROIs. |
| `FDR` | `0.05` | Benjamini-Hochberg level for `responsive` and `tuned`. |
| `BIN_S` | `0.2` | Bin (s) for Rastermap, correlations and assemblies. |
| `EVENT_Z` | `3.0` | Event threshold in robust SDs. |
| `ASSEMBLY_NULL` | `shift` | `shift` (circular-shift surrogates) or `mp` (Marchenko-Pastur). |
| `TOP_N` | `12` | ROIs in the tuning-curve and polar figures (the most reliable responsive ones). |
| `SEED` | `42` | Seed of the shuffles, surrogates, ICA and Rastermap. |
| `THREADS` | `0` | BLAS/numba threads; 0 = from the cgroup CPU quota. |

Bad parameters exit 2; no usable input or every session failed exit 1; a session that fails is
logged and skipped.

## Outputs (`OUTPUT_DIR/ophys-activity/`)

| File | Contents |
|---|---|
| `<sid>/<sid>_rois.csv` | One row per analysed ROI: `session_id, roi_id` (PlaneSegmentation id), `roi_row, subject_id, condition, group, mean_signal, sd_signal, noise_level, n_events, event_rate_hz, rastermap_rank, x, y` (mask centroid, image rows/columns), `assemblies`; with a stimulus: `responsive, response_stat, p_responsive, q_responsive, pref_condition, peak_response, kw_p, kw_q, tuned, pref_<factor>…`, `frac_pref_trials_above_blank, responsive_blank, blank_mean` (≥ 10 blanks), `reliability_split_half, reliability_trials, lifetime_sparseness`; angles: `tuning_curve, osi, dsi, gosi, gdsi, pref_dir_vs, pref_ori_vs`; categorical: `selectivity_index`. |
| `<sid>/<sid>_tuning.csv` | Long format: `session_id, roi_id, factor, level, kind` (`marginal` = pooled over the other factors, `at_preferred` = at the ROI's preferred value of the others), `mean_response, sem, n_trials`. |
| `<sid>/<sid>_trials.csv` | The analysed trials: the table's columns plus `trial_index`, `condition_label`, `blank`. |
| `<sid>/<sid>_responses.npz` | `responses` (trials × ROIs), `baseline`, `aligned` (trials × ROIs × time), `t`, `roi_ids`, `trial_index`, `start_time`, `condition`, `condition_index`, `blank`, windows, `signal`. |
| `<sid>/<sid>_correlations.npz` | `activity` (and `signal`, `noise`) ROI × ROI matrices, `roi_ids`. |
| `<sid>/<sid>_assemblies.csv`, `.npz` | Weight of every ROI in every assembly and membership; patterns, activation (assemblies × bins), bin times, eigenvalues, thresholds. |
| `<sid>/<sid>_*.png` | Rastermap raster (with the trials as a coloured bar), mean responses at the preferred stimulus, tuning curves, polar plots, OSI/DSI distributions, ROI map (preferred direction, or event rate without a stimulus), correlation matrices, assemblies. |
| `rois.csv` | All sessions' ROI tables. |
| `summary.json` | Per session: signal, ROI and trial counts, condition columns, `n_responsive`, `fraction_responsive`, `n_tuned`, median gOSI/OSI/gDSI/DSI, preferred-direction counts, median reliability and sparseness, mean signal/noise/activity correlation, event rates, assemblies; the figures with captions, tables, warnings. |
| `provenance.json` | Conventions keys: versions, parameters, inputs, timings, peak memory, DOIs to cite. |

## Results on the demo data

Outputs are in `workflow-systems-neuroscience/data/work/<demo>/ophys-activity/`.

**Allen Brain Observatory, VISp layer 2/3** (DANDI:000728, session
`sub-496001363_ses-503109347-StimA`, Allen's own ΔF/F of 353 ROIs, whole session, 115,497 frames
at 30 Hz; the three drifting-grating blocks, 628 trials = 8 directions × 5 temporal frequencies ×
15 + 30 blanks; defaults). The stimulus table is found in the same file
(`stimulus/presentation/drifting_gratings`), conditions `orientation_in_degrees` (directions) ×
`temporal_frequency_in_hz`, response window 2 s (the grating duration), baseline 1 s.

- **53 of 353 ROIs responsive** (15 %; random-time null, 10,000 shuffles, BH q < 0.05), 49 of
  them selective (Kruskal-Wallis). The more liberal blank-sweep criterion calls 137 (39 %).
  Preferred directions of responsive ROIs are spread over all eight (0°: 10, 45°: 9, 90°: 13,
  135°: 3, 180°: 9, 225°: 4, 270°: 3, 315°: 2).
- Medians over responsive ROIs: gOSI 0.85, gDSI 0.80, peak OSI 1.03, DSI 0.91; split-half
  reliability 0.82, trial-to-trial reliability 0.18; lifetime sparseness 0.90, population
  sparseness 0.97. Mean signal correlation 0.05, mean noise correlation 0.18 (a population-wide
  shared fluctuation; signal vs noise correlation across pairs r = 0.37). Activity correlation
  0.07; 28 assemblies (circular-shift bound 1.58; the Marchenko-Pastur bound finds 50+);
  0.11 events/s per ROI.
- **Against the Allen Brain Observatory's published per-cell metrics** (de Vries et al. 2020;
  `scripts/compare_allen.py`, 241 of the 353 ROIs have published drifting-grating metrics, matched
  by cell_specimen_id): for our 53 responsive ROIs the preferred direction is the same as Allen's
  `pref_dir_dg` in 94 % (vector-sum direction within 4.2° median), the preferred temporal frequency
  in 89 %; Pearson r with Allen's values: gDSI 0.92, gOSI 0.88, DSI 0.90, trial reliability
  (`reliability_dg`) 0.95, peak OSI 0.54 (peak OSI is the noisiest index). Medians agree (gOSI 0.85
  vs 0.90, gDSI 0.80 vs 0.82, DSI 0.91 vs 0.93). Over all 241 matched ROIs, preferred direction
  agrees in 83 %, gDSI r = 0.76.
- **First grating block only** (`T_STOP=631`, the window of the raw movie that `suite2p` and
  `ophys-traces` process; 199 trials, ~5 per condition, 6 of the 39 conditions in the window with < 3 trials left
  out): 19 responsive ROIs (5 %); the preferred direction matches Allen's (computed from all
  three blocks) in 90 % of them, gDSI r = 0.61, so one 10-minute block is enough for strongly
  tuned cells but not for a census. ophys-traces output of our own Suite2p run was not yet in
  `data/work/allen-2p/ophys-traces/` when this was built; rerun on it then (and match ROIs to
  Allen's by mask overlap before using `scripts/compare_allen.py`).

**Spontaneous path, hippocampal CA1 one-photon** (DANDI:002047, six open-field sessions, 12 min,
CNMF-E `Fluorescence/C`, the authors' `accepted` ROIs, no stimulus table): accepted ROIs (of all)
F07 42/118, F14 265/321, F20 239/331 (5xFAD), F35 148/441, F40 247/379, F48 245/593 (WT); median
event rate 1.2–4.1 events/min; mean pairwise correlation 0.002–0.016; assemblies 1, 8, 3 (5xFAD)
and 0, 4, 4 (WT). Three animals per genotype are far too few for a group comparison; these are
descriptive.

**Synthetic fixture** (`tests/`): all 36 driven ROIs responsive and none of the 4 undriven ones;
planted preferred directions recovered within 22.5° (peak condition exact in ≥ 95 %); the
orientation-tuned ROIs have low gDSI; both planted assemblies recovered (Jaccard ≥ 0.8); the
preferred odour of every odour-tuned ROI recovered from a stimulus.csv.

## Runtime

`computeTypes: [standard]`, 2 vCPU / 3072 MiB, 900 s timeout. These fit the platform's Lambda
limits, so the platform may run it as a Lambda step (it decides from these numbers). The image
is 1.1 GB.

Measured in the linux/amd64 image under emulation on an M-series Mac (Docker `--memory 3g
--cpus 2 --network none --read-only --user 65534:65534 --tmpfs /tmp`, defaults), on a shared
machine with other builds running:

| Data | Size | Wall time | cgroup `memory.peak` |
|---|---|---|---|
| Allen session (ΔF/F) | 353 ROIs × 115,497 frames (64 min), 628 trials | 149 s | 1.24 GB |
| Open field, 6 sessions | 42–265 ROIs × ~21,400 frames (12 min) each | 127 s | 0.90 GB |
| Synthetic fixture, 3 sessions | 20–40 ROIs | 61 s | 0.79 GB |

Memory scales with one session's traces (frames × ROIs × 4 bytes, 163 MB here) plus the aligned
trials × ROIs × time array (107 MB here; not stored when above 150 M values) and ~0.4 GB of
imported libraries. Time is dominated by the responsiveness test (N_SHUFFLES × trials × ROIs)
and Rastermap; a session with thousands of ROIs and trials may need a longer `timeoutSeconds` or
fewer `N_SHUFFLES`. No GPU is used.

## Running it locally

```sh
python3.12 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
make test PYTHON=.venv/bin/python    # pytest on the synthetic NWB fixture, no Docker
make run PYTHON=.venv/bin/python     # Docker over data/fixture/linked -> data/output/
```

By hand:

```sh
docker build --platform linux/amd64 -t ophys-activity .
docker run --rm --platform linux/amd64 --network none --read-only --tmpfs /tmp --memory 3g \
  -e INPUT_DIR=/data/input -e OUTPUT_DIR=/data/output \
  -v "$PWD/data/fixture/linked:/data/input:ro" -v "$PWD/data/output:/data/output" ophys-activity
```

`scripts/compare_allen.py` compares the tuning metrics with the Allen Brain Observatory's
published per-cell metrics (development tool; fetches them from api.brain-map.org).

## Limitations

- **Calcium is not spikes.** ΔF/F responses are filtered by the indicator (GCaMP6f decays over
  ~0.5–1 s); a 1 s baseline right after a previous trial still contains its tail, which lowers
  responses at short inter-trial intervals. Deconvolved rates sharpen this but inherit the
  spike-inference model's errors.
- **Responsiveness is conservative.** The random-time null includes evoked activity and the
  maximum over all conditions; with few trials per condition (e.g. one 10-minute grating block,
  ~5 trials each) only strongly and reliably driven ROIs pass. `responsive_blank` is a more
  liberal, Allen-style alternative. A ROI that is only *suppressed* by the stimulus is not called
  responsive.
- **Tuning indices are noisy for weak cells.** OSI/DSI of non-responsive ROIs are mostly noise
  (the figures show them in grey); peak OSI/DSI exceed 1 when the orthogonal or opposite
  response is negative. Use gOSI/gDSI for comparisons, and only over responsive ROIs.
- **Noise correlations need many trials** per condition and are inflated by anything shared
  across the field of view: running and arousal, slow drift, and neuropil contamination
  between neighbouring ROIs. The app does not regress out behaviour.
- **Assemblies** are statistical co-activation patterns, not anatomical circuits: overlapping or
  adjacent ROIs, shared neuropil and motion produce them too; stimulus-locked co-activation
  (ROIs tuned alike) also counts. The circular-shift bound is our choice for autocorrelated
  signals; `ASSEMBLY_NULL=mp` gives the paper's bound.
- One plane per file; a multi-plane recording is analysed as separate sessions (one NWB each).
- No running-modulation, receptive-field or natural-movie analyses.

## Licences

**GPL-3.0-or-later** for this app (see `LICENSE`): the image imports Rastermap, which is
GPL-3.0, in-process, so the app is distributed under a compatible licence. The other bundled
components are permissively licensed: `THIRD_PARTY.md`.

## Citations

Please cite these works if you use this app. Verified on Crossref/DataCite and PubMed; full
author lists in `CITATION.cff`.

- Stringer C, Zhong L, Syeda A, Du F, Kesa M, Pachitariu M. Rastermap: a discovery method for
  neural population recordings. *Nature Neuroscience* 28, 201–212 (2025; online 2024).
  https://doi.org/10.1038/s41593-024-01783-4, PMID 39414974.
- Lopes-dos-Santos V, Ribeiro S, Tort ABL. Detecting cell assemblies in large neuronal
  populations. *Journal of Neuroscience Methods* 220, 149–166 (2013).
  https://doi.org/10.1016/j.jneumeth.2013.04.010, PMID 23639919.
- Mazurek M, Kager M, Van Hooser SD. Robust quantification of orientation selectivity and
  direction selectivity. *Frontiers in Neural Circuits* 8 (2014).
  https://doi.org/10.3389/fncir.2014.00092, PMID 25147504.
- de Vries SEJ, Lecoq JA, Buice MA, et al. A large-scale standardized physiological survey
  reveals functional organization of the mouse visual cortex. *Nature Neuroscience* 23, 138–151
  (2020). https://doi.org/10.1038/s41593-019-0550-9, PMID 31844315.
- Cohen MR, Kohn A. Measuring and interpreting neuronal correlations. *Nature Neuroscience* 14,
  811–819 (2011). https://doi.org/10.1038/nn.2842, PMID 21709677.
- Vinje WE, Gallant JL. Sparse coding and decorrelation in primary visual cortex during natural
  vision. *Science* 287, 1273–1276 (2000). https://doi.org/10.1126/science.287.5456.1273,
  PMID 10678835.
- Benjamini Y, Hochberg Y. Controlling the false discovery rate: a practical and powerful
  approach to multiple testing. *Journal of the Royal Statistical Society Series B* 57, 289–300
  (1995). https://doi.org/10.1111/j.2517-6161.1995.tb02031.x
- Rübel O, Tritt A, Ly R, et al. The Neurodata Without Borders ecosystem for neurophysiological
  data science. *eLife* 11, e78362 (2022). https://doi.org/10.7554/eLife.78362, PMID 36193886.
- Harris CR, et al. Array programming with NumPy. *Nature* 585, 357–362 (2020).
  https://doi.org/10.1038/s41586-020-2649-2, PMID 32939066; Virtanen P, et al. SciPy 1.0.
  *Nature Methods* 17, 261–272 (2020). https://doi.org/10.1038/s41592-019-0686-2,
  PMID 32015543; Hunter JD. Matplotlib: a 2D graphics environment. *Computing in Science &
  Engineering* 9, 90–95 (2007). https://doi.org/10.1109/MCSE.2007.55; scikit-learn (Pedregosa F, et al. Scikit-learn: machine
  learning in Python. *Journal of Machine Learning Research* 12, 2825–2830 (2011); no DOI;
  https://jmlr.org/papers/v12/pedregosa11a.html), pandas, h5py.
- Data: Baker C. Allen Institute – Visual Coding – Optical Physiology, DANDI:000728 v0.240827.1809,
  https://doi.org/10.48324/dandi.000728/0.240827.1809 (CC-BY-4.0); Plusnin V, Pospelov N.
  Hippocampal CA1 one-photon calcium imaging and pose-tracked behavior in wild-type and 5xFAD
  mice across four behavioral paradigms, DANDI:002047 v0.261106.0903,
  https://doi.org/10.48324/dandi.002047/0.261106.0903 (CC-BY-4.0).

## Acknowledgements

Rastermap is developed by the MouseLand team (Carsen Stringer and Marius Pachitariu) at HHMI
Janelia Research Campus. The Allen Brain Observatory data and published cell metrics are from
the Allen Institute (de Vries et al. 2020); we thank the Allen Institute and DANDI for making
them openly available. The NWB format and ecosystem are developed by the Neurodata Without
Borders community (Rübel et al. 2022).
