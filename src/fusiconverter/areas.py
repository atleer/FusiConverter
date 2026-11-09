import ast
import numpy as np
import pandas as pd
from pathlib import Path
import json
from datetime import datetime, timezone
from .transforms import atlas_index_transform


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

def load_structure_graph(structure_graph_csv, coarse_structures_csv=None) -> dict[int, dict]:
    """Map Allen structure ids onto their names, colours and major brain division.

    'division' is the coarsest ancestor of a structure (Isocortex, HPF, TH, etc.)
    """
    structure_graph_csv = Path(structure_graph_csv)
    if coarse_structures_csv is None:
        coarse_structures_csv = structure_graph_csv.parent / 'allen_mouse_connectivity_coarse_structures.csv'

    graph = pd.read_csv(structure_graph_csv)

    coarse_acronyms = {}
    if Path(coarse_structures_csv).exists():
        coarse = pd.read_csv(coarse_structures_csv)
        coarse_acronyms = dict(zip(coarse['id'], coarse['acronym']))

    structures = {}
    for row in graph.itertuples(index=False):
        # both columns are stored as strings that look like lists, e.g. "[997, 8, 567]"
        ancestors = ast.literal_eval(row.structure_id_path)
        rgb = ast.literal_eval(row.rgb_triplet)
        # deepest ancestor that is one of the major divisions; structures above them (root, grey) have none
        division = next((coarse_acronyms[i] for i in reversed(ancestors) if i in coarse_acronyms), '')
        structures[int(row.id)] = {
            'acronym': str(row.acronym),
            'name': str(row.name),
            'division': division,
            'rgb': tuple(int(c) for c in rgb),
        }
    return structures
