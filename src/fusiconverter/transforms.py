import numpy as np
from pathlib import Path
import json
from datetime import datetime, timezone
from scipy.spatial.transform import Rotation
from .paths import registration_file_path


def fit_similarity_transform(moving: np.ndarray, fixed: np.ndarray, allow_scaling: bool = False):
    """Applies Kabsch-Umeyama Algorithm to transform data to be aligned atlas: rotation + translation,
    plus uniform scaling if 'allow_scaling' is True.

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

    if allow_scaling:
        moving_variance = (moving_centered ** 2).sum(axis=1).mean()
        scale = np.sum(S * np.diag(correction)) / moving_variance
    else:
        scale = 1.0
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

def compose_transform_matrix(translation_mm, rotation_deg, scale, centre_mm=(0, 0, 0)):
    """Build a 4x4 transform from a translation (mm), Euler rotation angles about layer axes 0/1/2 (degrees) and
    per-axis scale factors. Rotation and scaling happen about 'centre_mm' so the volume stays put while rotating."""
    centre = np.asarray(centre_mm, dtype=float)
    rotation = Rotation.from_euler('xyz', rotation_deg, degrees=True).as_matrix()  # 'xyz' = layer axes 0, 1, 2
    linear = rotation @ np.diag(np.asarray(scale, dtype=float))
    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = linear
    transform_matrix[:3, 3] = np.asarray(translation_mm, dtype=float) + centre - linear @ centre
    return transform_matrix

def rotate_about_axis_matrix(axis, angle_deg, centre_mm=(0, 0, 0)):
    """4x4 rotation of 'angle_deg' about an arbitrary 'axis' (world mm direction) through 'centre_mm'"""
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    centre = np.asarray(centre_mm, dtype=float)
    rotation = Rotation.from_rotvec(axis * np.deg2rad(angle_deg)).as_matrix()
    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = rotation
    transform_matrix[:3, 3] = centre - rotation @ centre
    return transform_matrix

def save_transform_matrix(hq_source_path, atlas_source_path, transform_matrix, hq_voxel_size_in_mm, atlas_voxel_size_in_mm, landmark_names=(), residuals_mm=(), method='landmarks'):
    """Save transformation matrix used for alignement. Landmarks/residuals are absent for a manual alignment."""
    json_path = registration_file_path(hq_source_path, 'alignment.json')
    json_path.parent.mkdir(parents=True, exist_ok=True)
    residuals_mm = np.asarray(residuals_mm, dtype=float)
    alignment = {
        'hq_source_path': str(hq_source_path),
        'atlas_source_path': str(atlas_source_path),
        'method': method,
        'transform_matrix': np.asarray(transform_matrix).tolist(),
        'hq_voxel_size_in_mm': [float(v) for v in hq_voxel_size_in_mm],
        'atlas_voxel_size_in_mm': [float(v) for v in atlas_voxel_size_in_mm],
        'landmark_names': list(landmark_names),
        'residuals_mm': dict(zip(landmark_names, residuals_mm.tolist())),
        'rms_error': float(np.sqrt(np.mean(np.square(residuals_mm)))) if residuals_mm.size else None,
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
    affine[np.ix_(axes,axes)] = transform_matrix[:3, :3] # np.ix_(axes,axes) accesses the rows and columns defined by spatial_axes
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
