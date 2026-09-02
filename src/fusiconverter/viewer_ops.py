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

def save_transform_matrix(hq_source_path, atlas_source_path, transform_matrix, hq_voxel_size_in_mm, atlas_voxel_size_in_mm, landmark_names, residuals_mm):
    """Save transformation matrix used for alignement"""
    json_path = Path(f'{hq_source_path}.alignment.json')
    alignment = {
        'hq_source_path': str(hq_source_path),
        'atlas_source_path': str(atlas_source_path),
        'transform_matrix': np.asarray(transform_matrix).tolist(),
        'hq_voxel_size_in_mm': [float(v) for v in hq_voxel_size_in_mm],
        'atlas_voxel_size_in_mm': [float(v) for v in atlas_voxel_size_in_mm],
        'landmark_names': list(landmark_names),
        'residuals_mm': dict(zip(landmark_names, np.asarray(residuals_mm).tolist())),
        'rms_error': float(np.sqrt(np.mean(np.square(residuals_mm)))),
        'created': datetime.now(timezone.utc).isoformat()
    }
    with open(json_path, 'w') as f:
        json.dump(alignment, f, indent=2)
        return json_path

def load_transform_matrix(json_path) -> dict:
    "Load transformation matrix used for alignement"
    with open(json_path) as f:
        alignment = json.load(f)

    alignment['transform_matrix'] = np.array(alignment['transform_matrix'])
    
    return alignment

def put_transform_matrix_in_layer_affine(transform_matrix, ndim, spatial_axes=(0,1,2)):
    """Embed the transformation matrix in an (ndim+1, ndim+1) matrix for a napari image layer.

    'spatial_axes' say which layer axes are spatial (Z, X, Y); other (usually trailing) axes (e.g. time) are left
    untouched, so the transform is never applied across time.
    """
    axes = list(spatial_axes)
    if ndim == 3 and axes == [0,1,2]:
        return transform_matrix
    affine = np.eye(ndim + 1)
    affine[np.ix_(axes,axes)] = transform_matrix[:3, :3] # TODO: what does np.ix do exactly?
    affine[axes, ndim] = transform_matrix[:3, 3]   # translation goes in the last column
    return affine

def get_transform_matrix_from_layer_affine(layer_affine_matrix, spatial_axes=(0,1,2)):
    """Pull the transformation matrix back out of a napari layer affine. Inverse of put_transform_matrix_in_layer_affine."""
    layer_affine_matrix = np.asarray(layer_affine_matrix)
    ndim = layer_affine_matrix.shape[0] - 1
    axes = list(spatial_axes)
    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = layer_affine_matrix[np.ix_(axes, axes)]
    transform_matrix[:3, 3] = layer_affine_matrix[axes, ndim]
    return transform_matrix

def atlas_index_transform(voxel_size_mm, transform_matrix, atlas_voxel_size_mm):
    """Matrix and offset taking a recording's voxel indices straight to atlas voxel indices.
    """
    to_atlas_index = np.diag(1.0 / np.asarray(atlas_voxel_size_mm, dtype=float))
    matrix = to_atlas_index @ transform_matrix[:3, :3] @ np.diag(np.asarray(voxel_size_mm, dtype=float))
    offset = to_atlas_index @ transform_matrix[:3, 3]
    return matrix, offset

def annotate_volume(spatial_shape, voxel_size_mm, transform_matrix, annotation, annotation_voxel_size_mm):
    """Look up which atlas structure each voxel of a recording sits in.
    """
    matrix, offset = atlas_index_transform(voxel_size_mm, transform_matrix, annotation_voxel_size_mm)
    voxel_indices = np.indices(spatial_shape, dtype=np.float32).reshape(3, -1) # (3, n_voxels)
    atlas_indices = np.rint(
        matrix.astype(np.float32) @ voxel_indices + offset.astype(np.float32)[:, None]
    ).astype(np.intp)

    inside = np.ones(atlas_indices.shape[1], dtype=bool)
    for axis, size in enumerate(annotation.shape):
        inside &= (atlas_indices[axis] >= 0) & (atlas_indices[axis] < size)

    labels = np.zeros(atlas_indices.shape[1], dtype=np.uint32)
    labels[inside] = annotation[tuple(atlas_indices[:, inside])]

    return labels.reshape(spatial_shape), float(1.0 - inside.mean())

def resample_atlas_to_recording(atlas, atlas_voxel_size_mm, transform_matrix, spatial_shape, voxel_size_mm, order=1):
    matrix, offset = atlas_index_transform(voxel_size_mm, transform_matrix, atlas_voxel_size_mm)
    resampled_atlas = affine_transform(atlas, matrix, offset=offset, output_shape = tuple(spatial_shape),
                                 order = order, mode='constant', cval=0)
    return resampled_atlas



# def apply_transform_matrix(volume: np.ndarray, source_voxel_size, transform_matrix, atlas_shape, atlas_voxel_size):
#     """Transfom volume onto atlas voxel grid via transform_matrix"""