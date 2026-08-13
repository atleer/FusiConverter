import napari
import numpy as np
from pathlib import Path
import json
from datetime import datetime, timezone
from scipy.ndimage import affine_transform


def load_landmarks(source_path):
    """Load landmarks that have already been written to file, if there are any"""
    json_path = Path(f'{source_path}.landmarks.json')
    if not json_path.exists():
        return {}
    with open(json_path) as f:
        return json.load(f)

def save_landmarks(points_layer, mode='append'):
    source_path = points_layer.metadata.get('source_path') # source path stems from name of data that the landmarks are put on (e.g. atlas or the recorded image)

    features = points_layer.features # names of landmarks and a list of booleans ('from_disk') indicating whether the landmark was added in this session or loaded from existing file are in features
    landmarks = {}
    for landmark_name, coords_landmark, from_disk in zip(features['landmark_name'], points_layer.data, features['from_disk']):
        # overwrite means only the landmarks added in this session are kept
        if mode == 'overwrite' and from_disk: # from_disk is just a boolean value
            continue
        landmarks[landmark_name] = coords_landmark.tolist() # added if the landmark doesn't already exist on file or the landmark already exists but the user has chosen to append

    if mode == 'append':
        # add new landmarks to existing landmarks on disk. If new landmarks have same name as landmarks on disk, new landmarks take precedence and will overwrite landmarks on disk
        merged = load_landmarks(source_path)
        merged.update(landmarks)
        landmarks = merged

    with open(Path(f'{source_path}.landmarks.json'), 'w') as f:
        json.dump(landmarks, f, indent=2)

def get_or_create_landmarks_layer(viewer, image_layer) -> napari.layers.Points:
    points_name = f'{image_layer.name}_landmarks'
    if points_name in viewer.layers:
        return viewer.layers[points_name]

    source_path = image_layer.metadata.get('source_path')

    saved_landmarks = load_landmarks(source_path) # this causes new landmarks to be appended to landmarks already set and written to file, if there are any

    points_layer = viewer.add_points(
        data = np.array(list(saved_landmarks.values())),
        name = points_name, # takes its name from the filename - atlas or HQ volume
        ndim=image_layer.ndim,
        scale=image_layer.scale,
        features={
            'landmark_name': np.array(list(saved_landmarks.keys()), dtype=object),
            'from_disk': np.ones(len(saved_landmarks), dtype=bool),
            },
        text='landmark_name',
        metadata={'source_path': source_path},
    )
    return points_layer

def match_landmarks(landmarks_a: dict, landmarks_b: dict):
    """Gets the landmarks in common between layers a and b and their respective coordinates"""
    names = sorted(set(landmarks_a) & set(landmarks_b))
    coords_a = np.array([landmarks_a[name] for name in names], dtype=float)
    coords_b = np.array([landmarks_b[name] for name in names], dtype=float)
    return names, coords_a, coords_b

def fit_similarity_transform(moving: np.ndarray, fixed: np.ndarray):
    """Applies Kabsch-Umeyama Algorithm to transform data to be aligned atlas: rotation + uniform scaling + translation.
    
    Returns a 4x4 homogeneous matrix mapping moving (landmarks on recorded data) points onto fixed (landmarks in atlas) points.
    """

    if len(moving) < 3:
        raise ValueError(f"Need at least 3 matched landmark pairs, got {len(moving)}.")

    moving_centroid = moving.mean(axis=0)
    fixed_centroid = fixed.mean(axis=0)
    moving_centered = moving - moving_centroid
    fixed_centered = fixed - fixed_centroid

    covariance = (fixed_centered.T @ moving_centered) / len(moving)

    U, S, Vt = np.linalg.svd(covariance)
    d = np.sign(np.linalg.det(U @ Vt)) or 1.0
    correction = np.diag([1.0, 1.0, d])
    rotation = U @ correction @ Vt

    moving_variance = (moving_centered ** 2).sum(axis=1).mean()
    scale = np.sum(S * np.diag(correction)) / moving_variance
    translation = fixed_centroid - scale * rotation @ moving_centroid

    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = scale * rotation
    transform_matrix[:3, 3] = translation
    return transform_matrix

def transform_residuals(transform_matrix: np.ndarray, moving: np.ndarray, fixed: np.ndarray) -> np.ndarray:
    """Computes difference between atlas and recorded data after transformation"""
    moving_homog = np.hstack([moving, np.ones((len(moving), 1))])
    transformed = (transform_matrix @ moving_homog.T).T[:, :3]
    return np.linalg.norm(transformed - fixed, axis = 1)

def save_registration(hq_source_path, atlas_source_path, transform_matrix, hq_voxel_size, atlas_voxel_size, landmark_names, residuals):
    json_path = Path(f'{hq_source_path}.registration.json')
    registration = {
        'model': 'similarity',
        'hq_source_path': str(hq_source_path),
        'atlas_source_path': str(atlas_source_path),
        'hq_voxel_size': [float(v) for v in hq_voxel_size],
        'atlas_voxel_size': [float(v) for v in atlas_voxel_size],
        'landmark_names': list(landmark_names),
        'transform_matrix': np.asarray(transform_matrix).tolist(),
        'residuals': dict(zip(landmark_names, np.asarray(residuals).tolist())),
        'rms_error': float(np.sqrt(np.mean(np.square(residuals)))),
        'created': datetime.now(timezone.utc).isoformat()
    }
    with open(json_path, 'w') as f:
        json.dump(registration, f, indent=2)
        return json_path

def load_registration(json_path) -> dict:
    "Load transformation matrix"
    with open(json_path) as f:
        registration = json.load(f)

    registration['transform_matrix'] = np.array(registration['transform_matrix'])
    
    return registration



# def apply_transform_matrix(volume: np.ndarray, source_voxel_size, transform_matrix, atlas_shape, atlas_voxel_size):
#     """Transfom volume onto atlas voxel grid via transform_matrix"""

    



# # Function to crop image in spatial dimensions (no temporal)
# def crop_image(viewer, z_range, x_range, y_range):
#     layer = viewer.layers.selection.active
#     if layer is None or not isinstance(layer, napari.layers.Image):
#         print("No image layer selected.")
#         return
    
#     # Get the shape of the image
#     shape = layer.data.shape
#     z_shape, x_shape, y_shape = shape[0], shape[1], shape[2]

#     # Set default ranges if not provided
#     if not z_range:
#         z_range = f'0:{z_shape}'
#     if not x_range:
#         x_range = f'0:{x_shape}'
#     if not y_range:
#         y_range = f'0:{y_shape}'
        
#     # Parse the input ranges
#     z_min, z_max = map(int, z_range.split(':'))
#     x_min, x_max = map(int, x_range.split(':'))
#     y_min, y_max = map(int, y_range.split(':'))

#     # Crop the image
#     cropped_image = layer.data[z_min:z_max, x_min:x_max, y_min:y_max]

#     # Add the cropped image as a new layer
#     viewer.add_image(cropped_image, name=f'Cropped_{layer.name}')

# def apply_log_normalization(viewer):
#     layer = viewer.layers.selection.active
#     if layer is None or not isinstance(layer, napari.layers.Image):
#         print("No image layer selected.")
#         return
#     # Map the image data to [0, 1] range
#     data_min = np.min(layer.data)
#     data_max = np.max(layer.data)
#     scaled_data = (layer.data - data_min) / (data_max - data_min)

#     # Apply logarithmic normalization
#     normalized_data = np.log1p(scaled_data)
    
#     # Add the normalized image as a new layer
#     viewer.add_image(normalized_data, name=f'LogNorm_{layer.name}')
