
# Functional Ultrasound Image Registration

This program lines up functional ultrasound (fUSI) recordings with the Allen mouse brain atlas, and then labels every point in the recording with the brain area it falls in.

Everything happens in [Napari](https://napari.org), an image viewer, with a few extra panels added on the right-hand side.

## Setup (Windows, one time only)

1. Double-click `install_uv.bat`. This installs **uv**, the tool that downloads everything else the program needs.
2. Close the window when it says it's done.

## Start the program

Double-click `launch_napari_with_widgets.bat`.

## Load files

In Napari, use **File → Open File(s)** to open:

- **Recordings**: the `.mat` files from the fUSI system.
- **Atlases**: `.nii` or `.nii.gz` files, e.g. the Allen atlas. The Allen and vascular atlases load automatically on startup if the data folder is next to the FusiConverter folder (see step 2 below).

## Typical workflow for registration

### 1. Download atlases and recording files from Sciebo

All files you need are currently in fusi/03_Data. You can place the data wherever you want on your computer, but the recommended location is next to (not inside) the FusiConverter repository.

1. Atlases are found in 00_ReferenceAtlas
    - Download the folder named Allen_CCF to get the Allen Atlas.
    - Download the folder named Keliris_vasc_atlas to get the vascular atlas.
2. Download the data from the session you want to register. Registration is done on the HQ (high quality - higher resolution) file when it is available.
    - Recommended organization: Put metadata (.csv files) in a separate folder and put HQ files and trial recordings (T*.mat files) in another folder inside each session folder.


### 2. Load files

1. The atlases load automatically when Napari starts, if the data folder is placed next to the FusiConverter folder (the recommended location in step 1).
    - The Allen atlas (**average_template_50.nii.gz**) and the vascular atlas (**VascularProbabilisticAtlas.nii**) are loaded.
    - If you downloaded the whole atlas folders from Sciebo, an `alignment.json` and `landmarks.json` should be included in the folders, and the saved alignment is applied to the Allen atlas automatically.
    - If your data is somewhere else, load the atlases yourself using **File → Open File(s)**. When you load the Allen atlas this way, you will be asked whether you want to apply the saved alignment. Click "Yes".
      - The only reason not to apply the saved alignment is if you need to redo it. To redo it, remove the Allen atlas layer and open it again with **File → Open File(s)**, then click "No".
2. Load the HQ file recorded in the session using **File → Open File(s)** in Napari.
    - If an alignment has already been done and an `alignment.json` has been saved for this session, you will be asked if you want to apply the saved alignment.

**Note: If the recording has already been aligned for the session you want to register, you can skip to point 6.**

### 3. Place landmarks (*Add Landmark* panel)

Landmarks are points you can recognise in both the recording and the atlas (e.g. a specific blood vessel or structure).

1. Pick the layer (recording or atlas) under **Target Image Layer**.
2. Type a name under **Landmark Name** and click **Add Landmark**. 
    - Currently used landmark namems are **"DA", "DP", "VL", and "VR"**, which stand for "Dorsal Anterior", "Dorsal Posterior", "Ventral Left", and "Ventral Right". If you set the VascularProbabilisticAtlas to be the **Target Image Layer** and click **Load Existing Landmarks**, you can see where they should be placed.
3. Click on the image to place the landmark.
4. Repeat for the same landmark on the other layer, **using the exact same name** (capitalization matters). If you have already placed landmarks on the recording, the other layer means the atlas, and vice versa.
5. Click **Save Landmarks**. They're stored next to the original file, so you can reload them later with **Load Existing Landmarks**.

You need **at least 3** landmarks with matching names on both layers. More landmarks give a better fit.

### 4. Align to the atlas (*Alignment* panel)

1. Choose the recording as **Layer to be moved** and the atlas as **Fixed layer**.
2. Click **Align to Atlas**. The recording moves, turns and resizes so its landmarks line up with the atlas landmarks.
3. Click **Save Alignment Matrix** to keep the resulting transformation matrix, which can be applied to other images in the future.

### 5. Fine-tune by hand (*Manual Alignment* panel), optional

Use this to fine-tune the alignment after the automatic fit using landmarks.

- **Translate**: shifts the recording (in mm).
- **Rotate**: turns it (in degrees).
- **Scale**: stretches or squeezes it (1 = unchanged).
- ** (OPTIONAL) Drag mode (3D)**: move the recording with the mouse instead. **Note**: You need to click it again to deactivate drag mode when you don't want to move the recording anymore.
  - Drag: shift
  - Shift + drag: tilt
  - Ctrl + drag: spin
- **Reset**: undoes your manual changes.
- **Save Alignment Matrix**: saves the result (this replaces any earlier saved transformation matrix).

### 6. Label brain areas (*Register to Atlas Areas* panel)

1. Choose the aligned recording and click **Register to Atlas Areas**.
2. The Allen **annotation** file (`data/00_ReferenceAtlas/Allen_CCF/annotation/annotation_50.nii.gz`) is picked automatically. Only if it can't be found are you asked to pick it yourself (click on the folder named "annotation"). The file `allen_mouse_connectivity_structure_graph.csv` must be in the same folder.
3. A coloured layer showing the brain areas appears on top of the recording. Hover over it to see each area's name in the bar at the bottom of the window.

The panel shows which areas the recording covers most. The full list is printed in the terminal window.

## Files the program creates

For a recording in `data/Raw/<animal>/<session>/`, results are saved in **`data/Processed/<animal>/<session>/registration_files/`**, which is created if it doesn't exist. Each file is named after the recording, with an extra ending added:

| File | Contains |
|---|---|
| `<recording>.landmarks.json` | Your landmarks |
| `<recording>.alignment.json` | How to move the recording onto the atlas |
| `<recording>.areas.json` | Which brain areas the recording covers, and how much of each |
| `<recording>.areas.npz` | Which brain area each point in the recording belongs to |

## Troubleshooting

- **"Need >=3 matching landmark names"**: check that the landmark names are spelled exactly the same on both layers.
- **"has not been aligned to the atlas yet"**: run step 2 first, or reopen the recording and choose **Yes** to apply its saved alignment.
- **"No voxel ... landed on a labelled atlas structure"**: the alignment is off. Check it in 3D and adjust it.

## For developers

To run from a terminal:

```
uv sync
uv run python scripts/launch_napari_with_widgets.py
```