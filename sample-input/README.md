# sample-input

This app reads NWB files with `processing/ophys` ROI traces (port `traces`), optionally a
stimulus / trials table (port `session`: an NWB with TimeIntervals, or a `stimulus.csv`) and
optionally a `sessions.csv` (port `metadata`).

## Synthetic fixture (no download)

```sh
make fixture PYTHON=.venv/bin/python   # data/fixture/{linked,flat}/ and data/fixture/truth.json
make run PYTHON=.venv/bin/python       # Docker over data/fixture/linked/ -> data/output/
```

Three sessions at 15 Hz on a 64 x 64 px field of view (see `tests/make_fixture.py`):

- `sub-syn01_ses-grating`: 40 ROIs, drifting gratings (8 directions x 2 contrasts x 6 repeats +
  12 blanks) in a separate session NWB (`stimulus/presentation/drifting_gratings`). ROIs 0-31 are
  direction tuned (ROI k prefers 45 x (k mod 8) deg), 32-35 orientation tuned, 36-39 not driven.
  Expected: ROIs 0-35 responsive, 36-39 not; vector-sum preferred direction within 22.5 deg of
  the planted one; gDSI of direction-tuned ROIs far above that of orientation-tuned ones.
- `sub-syn02_ses-spont`: 30 ROIs, no stimulus, two planted assemblies (ROIs 0-5, 10-15); ROIs
  28-29 have `accepted = false`. Expected: 28 ROIs analysed, 2 assemblies with those members.
- `sub-syn03_ses-odor`: 20 ROIs, `sub-syn03_ses-odor_stimulus.csv` with three odours; ROI k
  responds to odour k mod 3, ROIs 15-19 to none. Expected: preferred odour recovered.

## Real data

**Allen Brain Observatory, Visual Coding 2p** (DANDI:000728, CC-BY-4.0), session
`sub-496001363_ses-503109347-StimA` (VISp layer 2/3, GCaMP6f, 353 ROIs): the processed file
`sub-496001363/sub-496001363_ses-503109347-StimA_behavior+image+ophys.nwb` (asset
`d8e398ec-9b42-4369-a232-8d1ac7a0f80d`, 0.83 GB) holds Allen's ROIs and ΔF/F and the
`drifting_gratings` stimulus table, so it works as both `traces` and `session`:

```sh
dandi download "https://api.dandiarchive.org/api/assets/d8e398ec-9b42-4369-a232-8d1ac7a0f80d/download/"
```

In the workflow, `traces` is ophys-traces' `<session_id>_ophys.nwb` made from the raw movie and
`session` is the same companion NWB (for the stimulus table); the app matches them by
`session_id`.

**Spontaneous activity** (no stimulus): any of the six open-field sessions of DANDI:002047
(CC-BY-4.0; hippocampal CA1 one-photon imaging, CNMF-E traces in `processing/ophys/Fluorescence/C`,
`accepted` column), e.g. `sub-F35/sub-F35_ses-OF-F35-S1_behavior+image+ophys.nwb`. Its behaviour
`TimeIntervals` (`locomotion`, `rear`, ...) can be used for event-aligned responses with
`STIM_TABLE=locomotion`.

Any other DANDI ophys NWB works the same way: the app takes `Deconvolved`, else `DfOverF`, else
`Fluorescence` and `intervals/trials` or a stimulus table under `stimulus/presentation/`.
