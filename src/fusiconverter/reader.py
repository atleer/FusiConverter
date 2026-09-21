"""Napari reader for Allen CCF NIfTI atlases.

Registered as a napari plugin (see `napari.yaml`) so that opening an atlas from
File -> Open File(s) scales it the same way `view_mat_in_napari.py` does, i.e.
in mm, on the same grid as the recording.
"""

import numpy as np
from pathlib import Path

from .utils import load_atlas_image, load_mat_image, prepare_image_for_layer


def napari_get_reader(path):
    """return the reader itself, not the data."""
    path =  str(path).lower()
    if path.endswith(('.nii', '.nii.gz')):
        return read_atlas
    if path.endswith('.mat'):
        return read_mat
    return None

def read_mat(mat_path):
    image, voxel_size, _origin, image_name = load_mat_image(mat_path)

    image, scale, spatial_axes = prepare_image_for_layer(np.abs(image), voxel_size)
    kwargs = {
        'name': f'{image_name}_{Path(mat_path).stem}',
        'scale': scale,
        'metadata': {'source_path': str(mat_path), 'spatial_axes': spatial_axes},
    }

    return [(image, kwargs, 'image')]

def read_atlas(atlas_path):
    image, scale = load_atlas_image(atlas_path)
    kwargs = {
        'name': Path(atlas_path).name.rsplit('.nii', 1)[0],
        'scale': scale,
        # make it so a recording sitting inside the atlas volume stays visible in 3D
        'blending': 'additive',
        'metadata': {'source_path': str(atlas_path)},
    }
    return [(image, kwargs, 'image')]  # napari expects a list of layers
