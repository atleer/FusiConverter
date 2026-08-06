import json
from pathlib import Path

import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
import napari
import nibabel as nib


def select_files_from_gui(title=None, defaultextension='.mat') -> tuple[str, ...]:
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    file_path = filedialog.askopenfilenames(title=title, defaultextension=defaultextension)  # Open file dialog
    if not file_path:
        return ()
    return file_path


def select_save_dir_from_gui(title="Save Directory") -> str | None:
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    file_path = filedialog.askdirectory(title=title)  # Open folder dialog
    if not file_path:
        return None
    return file_path

def prompt_load_atlas() -> str | None:
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    load = messagebox.askyesno(
        "Load Atlas",
        "Would you like to load the Allen CCF atlas as an additional layer on top of the recorded image?\n\n"
        "If so, navigate to the folder Allen_CCF after clicking \"yes\" and select the atlas to load. Recommended: 50um resolution (e.g. 'average_template_50.nii.gz') in Allen_CCF."
    )
    if not load:
        root.destroy()
        return None
    atlas_path = filedialog.askopenfilename(
        title="Select Allen CCF Atlas File (recommended: 50um, e.g. average_template_50.nii.gz)",
        filetypes=[("NIfTI files", "*.nii.gz *.nii"), ("All files", "*.*")],
    )
    root.destroy()
    return atlas_path or None


def load_atlas_image(atlas_path) -> tuple[np.ndarray, tuple[float, float, float]]:
    img = nib.load(atlas_path)
    # Allen CCF .nii.gz files declare their units as "mm" but the zoom values
    # are actually written in um (e.g. 50.0 for the 50um atlas), so divide by
    # 1000 to get a real mm voxel size comparable to the h5 files' voxelSize.
    scale_mm = tuple(z / 1000 for z in img.header.get_zooms()[:3])
    return np.asarray(img.get_fdata()), scale_mm


def get_voxel_size_mm(h5_file) -> tuple[float, ...] | None:
    if 'voxelSize' not in h5_file:
        return None
    return tuple(h5_file['voxelSize'][()])


def get_landmarks_json_path(source_path) -> Path:
    return Path(f'{source_path}.landmarks.json')


def load_landmarks(source_path) -> dict[str, list[float]]:
    json_path = get_landmarks_json_path(source_path)
    if not json_path.exists():
        return {}
    with open(json_path) as f:
        return json.load(f)


def save_landmarks(points_layer) -> None:
    source_path = points_layer.metadata.get('source_path')
    if source_path is None:
        return
    names = points_layer.features['name'].tolist()
    landmarks = {name: coords.tolist() for name, coords in zip(names, points_layer.data)}
    with open(get_landmarks_json_path(source_path), 'w') as f:
        json.dump(landmarks, f, indent=2)


def get_or_create_landmarks_layer(viewer, image_layer) -> napari.layers.Points:
    points_name = f'{image_layer.name}_landmarks'
    if points_name in viewer.layers:
        return viewer.layers[points_name]

    source_path = image_layer.metadata.get('source_path')
    if source_path is None:
        print(f"'{image_layer.name}' has no known source file; landmarks won't be saved to disk.")
    saved_landmarks = load_landmarks(source_path) if source_path else {}

    points_layer = viewer.add_points(
        data=np.array(list(saved_landmarks.values())) if saved_landmarks else None,
        name=points_name,
        ndim=image_layer.ndim,
        scale=image_layer.scale,
        features={'name': np.array(list(saved_landmarks.keys()), dtype=object)},
        text='name',
        metadata={'source_path': source_path},
    )
    points_layer.events.data.connect(lambda event: save_landmarks(points_layer))
    return points_layer



# Function to crop image in spatial dimensions (no temporal)
def crop_image(viewer, z_range, x_range, y_range):
    layer = viewer.layers.selection.active
    if layer is None or not isinstance(layer, napari.layers.Image):
        print("No image layer selected.")
        return
    
    # Get the shape of the image
    z_shape, x_shape, y_shape, _ = layer.data.shape

    # Set default ranges if not provided
    if not z_range:
        z_range = f'0:{z_shape}'
    if not x_range:
        x_range = f'0:{x_shape}'
    if not y_range:
        y_range = f'0:{y_shape}'
        
    # Parse the input ranges
    z_min, z_max = map(int, z_range.split(':'))
    x_min, x_max = map(int, x_range.split(':'))
    y_min, y_max = map(int, y_range.split(':'))

    # Crop the image
    cropped_image = layer.data[z_min:z_max, x_min:x_max, y_min:y_max, :]

    # Add the cropped image as a new layer
    viewer.add_image(cropped_image, name=f'Cropped_{layer.name}')

def apply_log_normalization(viewer):
    layer = viewer.layers.selection.active
    if layer is None or not isinstance(layer, napari.layers.Image):
        print("No image layer selected.")
        return
    # Map the image data to [0, 1] range
    data_min = np.min(layer.data)
    data_max = np.max(layer.data)
    scaled_data = (layer.data - data_min) / (data_max - data_min)

    # Apply logarithmic normalization
    normalized_data = np.log1p(scaled_data)
    
    # Add the normalized image as a new layer
    viewer.add_image(normalized_data, name=f'LogNorm_{layer.name}')