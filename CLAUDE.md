# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

FusiConverter is a Napari-based tool for FUSi (functional ultrasound) recordings. It aligns `.mat` recordings to a vascular reference atlas and the Allen CCF atlas, then labels each voxel with the brain structure it sits in. It's a small internal lab tool: the reusable code lives in the `fusiconverter` package under `src/`, driven by one thin entry-point script in `scripts/`. `README.md` is the end-user guide (written for non-technical users); this file is for developers.

The old `.mat` → `.h5` conversion script (`convert_mat_to_h5.py`) and its `.bat` wrapper no longer exist. The viewer reads `.mat` files directly. The only `.h5` files left in `../data` are the `burst_corrected/` ones, which FusiConverter doesn't use.

## Environment & running

Dependency management is via [uv](https://docs.astral.sh/uv/) (`pyproject.toml` / `uv.lock`), which drives a `.venv`. Python is pinned to 3.13 (`>=3.13,<3.14`), and `uv.lock` only targets Windows AMD64. Main packages: napari[pyqt5] (<0.6), nibabel, qtpy, numpy, scipy, pandas, xarray, h5py, matplotlib, plotly, ipywidgets, ipykernel, tqdm. `uv sync` also installs the project itself in editable mode (`.venv/Lib/site-packages/fusiconverter.pth` points at `src/`), which is what registers the Napari reader plugin and makes `import fusiconverter` work from anywhere.

```
winget install --id=astral-sh.uv -e
uv sync
uv run python scripts/launch_napari_with_widgets.py
```
`install_uv.bat` and `launch_napari_with_widgets.bat` wrap the install and launch steps as double-clickable files for non-technical (Windows) users. The launch `.bat` runs the script, then `pause`.

For notebook work, the ipykernel may need registering once:
```
uv run python -m ipykernel install --name fusi
```

Tests are in `tests/` (pytest, in the `dev` dependency group; run with `uv run python -m pytest` from the repo root, configured in `pyproject.toml`):
- `tests/unit/`: one file per package module (`test_loading.py`, `test_landmarks.py`, `test_transforms.py`, `test_areas.py`), on synthetic data built by the fixtures in `tests/conftest.py`.
- `tests/integration/`: the reader, the widgets (in a napari viewer), and the whole landmarks → alignment → area map chain on synthetic data with a known answer. `test_readers_real_data.py` reads every recording and atlas in `../data`.
- `tests/e2e/`: `test_golden.py` reruns the pipeline on one real session and compares it with `tests/e2e/golden/golden.json`, the output recorded from the code by `tests/e2e/make_golden.py` ("golden" means what the code produced then, not checked to be correct). Rerun `make_golden.py` only when a change in the outputs is intended. `test_launch_script.py` runs the launch script with `napari.run` patched out.
- Markers: `data` (needs `../data`; every such test is skipped, with its reason, when the folder is missing) and `gui` (needs a Qt viewer). `-m "not data"` runs only the quick synthetic tests.
- Known bugs are strict `xfail` tests that expect the correct behaviour. Fixing a bug means removing its `xfail`.

No linter or build step is set up.

## Data

- The data folder sits at the same level as the FusiConverter folder (`../data`), not inside it.
- The Allen atlases live in `data/00_ReferenceAtlas/Allen_CCF/`, and `reader.py` relies on that folder name (see below). The Keliris vascular atlases (the reference frame) live in `data/00_ReferenceAtlas/Keliris_vasc_atlas/`.
- Recordings live in `data/Raw/<animal>/<date>_Session_XX[_reg]/` (flat, e.g. `Raw/GEYP659/20251211_Session_14_reg/GEYP659_20251211_1406_fUS_HQ.mat`).

### Read the README.md in the data folder before handling/reading the data itself to get an overview of its content.

## Architecture

The entry point is an interactive GUI tool (Napari + Qt dialogs), not a CLI. Don't try to script or pipe it.

**`scripts/launch_napari_with_widgets.py`**: changes into the repo root and imports the installed package as `fusiconverter.*`. It opens a Napari viewer with four Qt dock widgets from `widgets.py`. On startup it then loads `VascularProbabilisticAtlas.nii` and `Allen_CCF/average_template_50.nii.gz` from `../data` (skipping any that are missing) via `read_atlas(..., ask=True)`, so the Allen atlas's saved alignment triggers the apply-alignment prompt. Recordings (`.mat`) and other atlases (`.nii`/`.nii.gz`) are opened by the user via File → Open File(s), handled by `reader.py`.

Always import the package as `fusiconverter` (scripts, tests, notebooks) and use relative imports inside it, never `src.fusiconverter`. `napari.yaml` registers the reader as `fusiconverter.reader:napari_get_reader`, so a `src.` import would load a second, separate copy of every module, and monkeypatching one copy wouldn't affect the other.

**`src/fusiconverter/`** is the package holding everything reusable:
- **`reader.py`** + **`napari.yaml`**: Napari plugin reader for `*.mat`, `*.nii`, `*.nii.gz`.
  - `read_mat`: takes `np.abs` of the image (ultrasound data can be complex) and passes it through `loading.prepare_image_for_layer`, which moves a trailing time axis to the front so the spatial axes line up with the 3D atlas. It stores `source_path` and `spatial_axes` in the layer metadata.
  - `read_atlas`: scales the atlas to mm and uses additive blending so a recording inside the atlas volume stays visible in 3D.
  - **Auto-align on open**: if `<source>.alignment.json` exists, a Qt prompt asks whether to apply it, and if so the saved transform is set as the layer affine. The **vascular atlas is the reference frame**. Only atlases whose parent folder is named `Allen_CCF` get moved onto it, and recordings are aligned to it too.
- **`loading.py`**: reading recordings and atlases from disk: `load_mat_image` (see the `.mat` quirks below), `load_atlas_image` and `prepare_image_for_layer`. `load_atlas_image` converts voxel sizes to mm by guessing the unit from the header zoom, because the headers all say "mm" but don't all mean it: Allen CCF is in µm, the 78 µm Keliris vascular atlas is 10× inflated, and the Keliris VascularProbabilisticAtlas is real mm. The comment at the top of the module lists which atlas files are which.
- **`landmarks.py`**: save/load (`<source>.landmarks.json`), `get_or_create_landmarks_layer` and `match_landmarks`.
- **`transforms.py`**: the alignment maths, napari-independent and testable on arrays.
  - Fitting: `fit_similarity_transform` (Kabsch-Umeyama: rotation + translation, plus a uniform scale with `allow_scaling=True`), `transform_residuals`, `compose_transform_matrix`, `rotate_about_axis_matrix`.
  - Saving and loading `<source>.alignment.json`, and conversion between a 4x4 transform matrix and a Napari layer affine (`put_transform_matrix_in_layer_affine` / `get_transform_matrix_from_layer_affine`, which leave non-spatial axes such as time untouched). `atlas_index_transform` maps recording voxel indices to atlas voxel indices.
- **`areas.py`**: voxel-level registration. `annotate_volume`, `summarize_areas`, `save_area_map` → `<source>.areas.json` plus a `.areas.npz` of per-structure voxel indices, and `load_structure_graph` (maps Allen structure ids to acronym/name/colour/major division from the structure-graph CSV).
- **`widgets.py`**: the Qt dock widgets, plus `prompt_load_annotation`, the Tk file dialog `RegisterToAreasWidget` uses to ask for the annotation file.
  - `AddLandmark`: place/name/save/load landmarks on a layer.
  - `AlignmentWidget`: fit a transform from landmarks matched between an HQ volume and the atlas, then save it.
  - `ManualAlignmentWidget`: spinboxes (translate/rotate/scale) and a 3D mouse drag mode to adjust a layer by hand on top of whatever transform it already has, pivoting about the recording's centre, with reset and save to the same `.alignment.json`.
  - `RegisterToAreasWidget`: label every voxel of an aligned recording with its Allen structure (uses `../data/00_ReferenceAtlas/Allen_CCF/annotation/annotation_50.nii.gz` automatically, or asks the user for the annotation file if it's missing; `allen_mouse_connectivity_structure_graph.csv` must sit next to it) and add it as a Labels layer. Results are saved automatically.
- **`__init__.py`**: only a placeholder `main()`, which `pyproject.toml` wires up as the `fusiconverter` script and as the `napari.manifest` entry point.

Coordinate conventions worth keeping straight when editing: images are (Z, X, Y) with an optional trailing time axis in the file, which becomes a *leading* time axis in the layer. `scale` and voxel sizes are in mm. Transform matrices are 4x4 homogeneous and map recording mm → vascular reference atlas mm. A Napari layer must have its spatial axes last to line up with 3D layers.

Results are saved as `<source>.landmarks.json`, `.alignment.json`, `.areas.json` and `.areas.npz` (`paths.registration_file_path`). For a source in `Raw/<animal>/<session>/` they go to `Processed/<animal>/<session>/registration_files/`; any other source (e.g. the atlases) gets them next to the file.

**`inspect-data-online/`**: notebooks for looking at recordings after registration (currently `visualize-recording-after-registration.ipynb`, which reads the `*areas*.json` / `.npz` files from `../data/<session>`). It expects to run from the repo root (it `chdir`s there).

**`fUSI_analysis_from_thoralf/`**: figures from an outside analysis, not part of the code.

**`scratch/`**: exploratory, cell-based (`# %%`) scripts that reverse-engineer the structure of raw `.mat` files from different acquisition modes (`main_bmode.py`, `main_doppler.py`, `main_functional_doppler.py`). They reference local data paths and aren't meant to be run end-to-end; treat them as reference notes. The folder is listed in `.gitignore`, but these three files are tracked.

**`notes/`**: dated working notes (`YYYY-MM-DD.md`), including a to-do list of planned changes. Gitignored.

## Notes on the .mat format quirks

The FUSi-exported MATLAB files are inconsistent. The loader in `loading.load_mat_image` copes with this (the `scratch/` scripts show how these were discovered):
- The main image variable's name differs by acquisition mode: `bmode`, `doppler`, `Ihq`, or (in older/other files) `I`.
- `metadata` is a MATLAB struct-array; each field must be accessed as `data['metadata'][name].item().flatten()`.
- `origen` is a misspelling of `origin` in some files.
- Some files (e.g. in `../data/Raw/BECU184/20260624_Session_01`) are MATLAB v7.3 (HDF5), which `scipy.io.loadmat` can't read, so they're read with h5py. Which branch runs is decided by `'2026'` appearing in the path, not by the file format. This branch is untested and has known problems (see `notes/claude_notes/2026-11-09-follow-ups.md`).
- `tag` metadata exists in tagged/stimulus recordings but is not handled. `load_mat_image` reads it with the rest of the metadata and then ignores it.
