import json
from datetime import datetime, timezone
from pathlib import Path
from warnings import warn

import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
import napari
import nibabel as nib
from scipy import io as sio
from scipy.ndimage import affine_transform


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
    points_layer.events.features.connect(lambda event: save_landmarks(points_layer))
    return points_layer


## Registration: matching landmarks and fitting a similarity transform

def match_named_points(landmarks_a: dict, landmarks_b: dict) -> tuple[list[str], np.ndarray, np.ndarray]:
    names = sorted(set(landmarks_a) & set(landmarks_b))
    pts_a = np.array([landmarks_a[name] for name in names], dtype=float)
    pts_b = np.array([landmarks_b[name] for name in names], dtype=float)
    return names, pts_a, pts_b


def fit_similarity_transform(moving_mm: np.ndarray, fixed_mm: np.ndarray) -> np.ndarray:
    """Closed-form (Umeyama) similarity fit: rotation + uniform scale + translation.

    Returns a 4x4 homogeneous matrix mapping moving_mm points onto fixed_mm points.
    """
    if len(moving_mm) < 3:
        raise ValueError(f"Need at least 3 matched landmark pairs, got {len(moving_mm)}.")

    moving_centroid = moving_mm.mean(axis=0)
    fixed_centroid = fixed_mm.mean(axis=0)
    moving_centered = moving_mm - moving_centroid
    fixed_centered = fixed_mm - fixed_centroid

    covariance = (fixed_centered.T @ moving_centered) / len(moving_mm)
    U, S, Vt = np.linalg.svd(covariance)
    d = np.sign(np.linalg.det(U @ Vt)) or 1.0
    correction = np.diag([1.0, 1.0, d])
    rotation = U @ correction @ Vt

    moving_variance = (moving_centered ** 2).sum(axis=1).mean()
    scale = np.sum(S * np.diag(correction)) / moving_variance
    translation = fixed_centroid - scale * rotation @ moving_centroid

    matrix = np.eye(4)
    matrix[:3, :3] = scale * rotation
    matrix[:3, 3] = translation
    return matrix


def transform_residuals_mm(matrix: np.ndarray, moving_mm: np.ndarray, fixed_mm: np.ndarray) -> np.ndarray:
    moving_homog = np.hstack([moving_mm, np.ones((len(moving_mm), 1))])
    transformed = (matrix @ moving_homog.T).T[:, :3]
    return np.linalg.norm(transformed - fixed_mm, axis=1)


def get_registration_json_path(hq_source_path) -> Path:
    return Path(f'{hq_source_path}.registration.json')


def save_registration(
    hq_source_path, atlas_source_path, matrix_mm, hq_voxel_size_mm, atlas_voxel_size_mm,
    landmark_names, residuals_mm,
) -> Path:
    json_path = get_registration_json_path(hq_source_path)
    registration = {
        'model': 'similarity',
        'hq_source': str(hq_source_path),
        'atlas_source': str(atlas_source_path),
        'hq_voxel_size_mm': [float(v) for v in hq_voxel_size_mm],
        'atlas_voxel_size_mm': [float(v) for v in atlas_voxel_size_mm],
        'landmark_names': list(landmark_names),
        'matrix_mm': np.asarray(matrix_mm).tolist(),
        'residuals_mm': dict(zip(landmark_names, np.asarray(residuals_mm).tolist())),
        'rms_error_mm': float(np.sqrt(np.mean(np.square(residuals_mm)))),
        'created': datetime.now(timezone.utc).isoformat(),
    }
    with open(json_path, 'w') as f:
        json.dump(registration, f, indent=2)
    return json_path


def load_registration(json_path) -> dict:
    with open(json_path) as f:
        registration = json.load(f)
    registration['matrix_mm'] = np.array(registration['matrix_mm'])
    return registration


def resample_volume_to_atlas(
    volume: np.ndarray, source_voxel_size_mm, matrix_mm: np.ndarray, atlas_shape, atlas_voxel_size_mm,
) -> np.ndarray:
    """Resample a 3D volume from its own voxel grid onto the atlas voxel grid via matrix_mm
    (source physical mm -> atlas physical mm), using scipy's backward-mapping convention.
    """
    source_scale = np.diag(list(source_voxel_size_mm) + [1.0])
    atlas_scale_inv = np.diag([1 / s for s in atlas_voxel_size_mm] + [1.0])
    # source voxel index -> atlas voxel index
    voxel_to_voxel = atlas_scale_inv @ matrix_mm @ source_scale
    voxel_to_voxel_inv = np.linalg.inv(voxel_to_voxel)

    return affine_transform(
        volume,
        voxel_to_voxel_inv[:3, :3],
        offset=voxel_to_voxel_inv[:3, 3],
        output_shape=atlas_shape,
        order=1,
        mode='constant',
        cval=0.0,
    )


def load_mat_image(filepath) -> tuple[np.ndarray, dict]:
    """Load a raw (non-HQ) .mat recording's image data + flattened metadata.

    Standalone variant of the loader in scripts/convert_to_h5.py, extended with the
    'I' variable name (seen in this dataset's T_*.mat files) per CLAUDE.md's documented
    .mat quirks. Kept separate to avoid touching convert_to_h5.py's existing behavior.
    """
    data = sio.loadmat(filepath)

    for image_name in ['bmode', 'doppler', 'Ihq', 'I']:
        if image_name in data:
            break
    else:
        raise ValueError(f"Could not find image data in '{filepath}'.")

    mdata = {}
    if 'metadata' in data:
        for name in data['metadata'].dtype.names:
            mdata[name] = data['metadata'][name].item().flatten()

    if 'origen' in mdata:
        mdata['origin'] = mdata.pop('origen')
    if 'tag' in mdata:
        mdata.pop('tag')
        warn('tags not yet supported.')

    return data[image_name], mdata


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