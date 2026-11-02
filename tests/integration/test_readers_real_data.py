"""
Test readers created to enable loading files from napari via File -> Open File(s)
"""
# %% import libraries
import pytest
from pathlib import Path
import numpy as np

# %% Import functions to test

from fusiconverter.reader import read_mat, read_atlas

pytestmark = pytest.mark.data # every test here reads real files from ../data

# %% Get paths to data files

session_date_id = '20251211-2'
recorded_data_dir = Path(__file__).parents[3] / Path('data') / session_date_id
paths_mat = sorted([path for path in recorded_data_dir.glob('*/*.mat') if '_BM' not in path.stem and '_DP' not in path.stem and '_HQ' not in path.stem])

def test_mat_files_found():
    if not paths_mat:
        pytest.skip(f'no .mat recordings found in {recorded_data_dir}')

@pytest.fixture(scope='module', params=paths_mat, ids=lambda p: p.name)
def read_mat_file(request):
    return read_mat(str(request.param), ask = False)

# %%

atlas_data_dir = Path(__file__).parents[3] / Path('data')

# inside each tuple in the list of atlases: (path to atlas, expected voxel size in mm, whether an alignment.json file exists and the atlas should be moved onto the vascular reference atlas)
atlases = [
    (atlas_data_dir / 'Keliris_vasc_atlas' / 'VascularProbabilisticAtlas.nii', 0.078125, False),
    (atlas_data_dir / 'Keliris_vasc_atlas' / 'T2_3D_AnatomicalTemplate.nii', 0.078125, False),
    (atlas_data_dir / 'Allen_CCF' / 'average_template_50.nii.gz', 0.05, True), # has a saved .alignment.json, therefore True
]

def test_atlases_found():
    missing = [str(path) for path, _, _ in atlases if not path.exists()]
    if missing:
        pytest.skip(f'atlases not found: {missing}')

# %%
def test_read_mat(read_mat_file):
    assert len(read_mat_file) == 1 # napari wants it to be a list with a tuple inside, therefore length 1
    assert len(read_mat_file[0]) == 3 # the tuple inside the list should contain 3 elements

    image, kwargs, napari_layer_type = read_mat_file[0]

    assert len(image.shape) in [3,4] # the recorded data should have 3 spatial dimensions and sometimes a time dimension

    assert len(kwargs['scale']) == image.ndim
    assert kwargs['metadata']['spatial_axes'] == tuple(range(image.ndim - 3, image.ndim)) # spatial axes last
    assert not np.iscomplexobj(image)
    assert kwargs['metadata']['source_path'].endswith('.mat')

    if 'affine' in kwargs:
        assert np.asarray(kwargs['affine']).shape == (image.ndim + 1, image.ndim + 1)

    assert napari_layer_type == 'image'

# %%

@pytest.mark.parametrize('path, voxel_size_mm, aligned_to_vascular', atlases, ids=[path.name for path, _, _ in atlases])
def test_read_atlas(path, voxel_size_mm, aligned_to_vascular):
    if not path.exists():
        pytest.skip(f'atlas not found: {path}')
    layers = read_atlas(path, ask = False)
    assert len(layers) == 1 # napari wants it to be a list with a tuple inside, therefore length 1
    assert len(layers[0]) == 3 # the tuple inside the list should contain 3 elements

    image, kwargs, napari_layer_type = layers[0]

    assert image.ndim == 3 # the atlas has 3 spatial dimensions
    assert kwargs['name'] == path.name.split('.')[0] # without .nii / .nii.gz
    assert kwargs['blending'] == 'additive'
    assert kwargs['metadata']['source_path'] == str(path)
    if aligned_to_vascular:
        assert np.asarray(kwargs['affine']).shape == (4, 4)
    assert napari_layer_type == 'image'



