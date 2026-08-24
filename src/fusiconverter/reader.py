"""Napari reader for Allen CCF NIfTI atlases.

Registered as a napari plugin (see `napari.yaml`) so that opening an atlas from
File -> Open File(s) scales it the same way `view_mat_in_napari.py` does, i.e.
in mm, on the same grid as the recording.
"""

from pathlib import Path

from .utils import load_atlas_image


def napari_get_reader(path):
    """return the reader itself, not the data."""
    if str(path).lower().endswith(('.nii', '.nii.gz')):
        return read_atlas
    return None


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
