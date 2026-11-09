import napari
import numpy as np
from pathlib import Path
import json
from datetime import datetime, timezone
from .transforms import atlas_index_transform


def load_landmarks(source_path):
    """Load landmarks that have already been written to file, if there are any"""
    json_path = Path(f'{source_path}.landmarks.json')
    if not json_path.exists():
        return {}
    with open(json_path) as f:
        return json.load(f)

def save_landmarks(points_layer, mode='append'):
    source_path = points_layer.metadata.get('source_path') # source path stems from name of data that the landmarks are put on (e.g. atlas or the recorded image)

    features = points_layer.features # names of landmarks are in features
    # overwrite means the file ends up matching the layer exactly: loaded landmarks are kept, landmarks deleted in napari are dropped
    landmarks = {}
    for landmark_name, coords_landmark in zip(features['landmark_name'], points_layer.data):
        landmarks[landmark_name] = coords_landmark.tolist()

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
        affine=image_layer.affine.affine_matrix,
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

def summarize_areas(labels, structures):
    """Count how many voxels of the recording fall in each structure, most covered first."""
    ids, counts = np.unique(labels, return_counts=True)
    summary = []
    for structure_id, n_voxels in zip(ids.tolist(), counts.tolist()):
        if structure_id == 0: # unlabelled / outside the atlas
            continue
        structure = structures.get(structure_id, {})
        summary.append({
            'id': structure_id,
            'acronym': structure.get('acronym', f'unknown_{structure_id}'),
            'name': structure.get('name', f'Structure id {structure_id} is not in the structure graph'),
            'division': structure.get('division', ''),
            'n_voxels': n_voxels,
            'fraction_of_recording': n_voxels / labels.size,
            # flat indices into labels.ravel(); use np.unravel_index(..., labels.shape) for (Z, X, Y)
            'voxel_indices': np.flatnonzero(labels == structure_id),
        })
    summary.sort(key=lambda structure: structure['n_voxels'], reverse=True)
    return summary

def save_area_map(source_path, summary, labels, annotation_source_path, fraction_outside):
    """Write which structures a recording covers to <source_path>.areas.json

    The voxel indices are too bulky for JSON, so they go to <source_path>.areas.npz instead,
    one flat index array per structure, keyed by structure id. 'hq_shape' is stored
    alongside them so the flat indices can be unravelled without the labels volume.
    """
    json_path = Path(f'{source_path}.areas.json')
    npz_path = Path(f'{source_path}.areas.npz')

    voxel_indices = {str(structure['id']): structure['voxel_indices']
                     for structure in summary if 'voxel_indices' in structure}
    # keep the JSON a readable summary - drop the indices without touching the caller's dicts
    structures = [{key: value for key, value in structure.items() if key != 'voxel_indices'}
                  for structure in summary]

    area_map = {
        'source_path': str(source_path),
        'annotation_source_path': str(annotation_source_path),
        'n_voxels_total': int(labels.size),
        'n_voxels_unlabelled': int(np.count_nonzero(labels == 0)),
        'fraction_outside_atlas': fraction_outside,
        'hq_shape': list(labels.shape),
        'voxel_indices_path': str(npz_path) if voxel_indices else None,
        'structures': structures,
        'created': datetime.now(timezone.utc).isoformat()
    }
    with open(json_path, 'w') as f:
        json.dump(area_map, f, indent=2)

    if voxel_indices:
        np.savez_compressed(npz_path, labels_shape=np.array(labels.shape), **voxel_indices)

    return json_path