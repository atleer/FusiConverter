"""Napari reader for Allen CCF NIfTI atlases.

Registered as a napari plugin (see `napari.yaml`) so that atlases can be opened from
File -> Open File(s) and scaled to match the recorded fUSI data, if needed.
"""

import numpy as np
from pathlib import Path
from qtpy.QtWidgets import QMessageBox

from .utils import load_atlas_image, load_mat_image, prepare_image_for_layer
from .viewer_ops import load_transform_matrix, put_transform_matrix_in_layer_affine


def napari_get_reader(path):
    """return the reader itself, not the data."""
    path =  str(path).lower()
    if path.endswith(('.nii', '.nii.gz')):
        return read_atlas
    if path.endswith('.mat'):
        return read_mat
    return None

def _saved_alignment_affine(source_path, ndim, spatial_axes=(0, 1, 2), ask=True):
    """Affine from <source>.alignment.json (maps to the vascular reference atlas's mm), or None if there is none or the user declines."""
    alignment_path = Path(f'{source_path}.alignment.json')
    if not alignment_path.exists():
        return None
    if ask:
        answer = QMessageBox.question(
            None, 'Apply saved alignment?',
            f'{Path(source_path).name} has a saved alignment to the vascular reference atlas.\n\n'
            'Apply it now? The layer will line up with the vascular reference atlas in 3D, but it will be '
            'rotated, so it can no longer be viewed slice-by-slice in 2D.\n\n'
            'Choose No to view it unaligned (2D slices work as normal).')
        if answer != QMessageBox.Yes:
            return None
    alignment = load_transform_matrix(alignment_path)
    print(f'Applied saved alignment to {Path(source_path).name}')
    return put_transform_matrix_in_layer_affine(alignment['transform_matrix'], ndim, spatial_axes)

def read_mat(mat_path, ask = True):
    image, voxel_size, _origin, image_name = load_mat_image(mat_path)

    image, scale, spatial_axes = prepare_image_for_layer(np.abs(image), voxel_size)
    kwargs = {
        'name': f'{image_name}_{Path(mat_path).stem}',
        'scale': scale,
        'metadata': {'source_path': str(mat_path), 'spatial_axes': spatial_axes},
    }
    affine = _saved_alignment_affine(mat_path, image.ndim, spatial_axes, ask=ask)
    if affine is not None:
        kwargs['affine'] = affine

    return [(image, kwargs, 'image')]

def read_atlas(atlas_path, ask=True):
    image, scale = load_atlas_image(atlas_path)
    kwargs = {
        'name': Path(atlas_path).name.rsplit('.nii', 1)[0],
        'scale': scale,
        # make it so a recording sitting inside the atlas volume stays visible in 3D
        'blending': 'additive',
        'metadata': {'source_path': str(atlas_path)},
    }
    # the vascular atlas is the reference frame, so only the Allen atlas gets moved onto it
    if Path(atlas_path).parent.name == 'Allen_CCF':
        affine = _saved_alignment_affine(atlas_path, image.ndim, ask=ask)
        if affine is not None:
            kwargs['affine'] = affine
    return [(image, kwargs, 'image')]  # napari expects a list of layers
