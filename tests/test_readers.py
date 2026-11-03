"""
Test readers created to enable loading files from napari via File -> Open File(s)
"""
# %% import libraries
import pytest
from pathlib import Path
import numpy as np

# %% Import functions to test

from src.fusiconverter.reader import read_mat, read_atlas

# %% Get paths to data files

session_date_id = '20251211-2'
recorded_data_dir = Path(__file__).parent.parent.parent / Path('data') / session_date_id
paths_mat = sorted([path for path in recorded_data_dir.glob('*/*.mat') if '_BM' not in path.stem and '_DP' not in path.stem and '_HQ' not in path.stem])

def test_mat_files_found():
    assert paths_mat, f'no .mat recordings found in {recorded_data_dir}'

@pytest.fixture(scope='module', params=paths_mat, ids=lambda p: p.name)
def read_mat_file(request):
    return read_mat(str(request.param), ask = False)

# %%

vascular_atlas_dir = Path(__file__).parent.parent.parent / Path('data') / 'Keliris_vasc_atlas'
path_vascular_atlas = vascular_atlas_dir / 'VascularProbabilisticAtlas.nii'

def test_vascular_atlas_found():
    assert path_vascular_atlas, f'no vascular atlas found'

@pytest.fixture
def read_atlas_file():
    return read_atlas(path_vascular_atlas, ask = False)

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

def test_read_atlas(read_atlas_file):
    assert len(read_atlas_file) == 1 # napari wants it to be a list with a tuple inside, therefore length 1
    assert len(read_atlas_file[0]) == 3 # the tuple inside the list should contain 3 elements

    image, kwargs, napari_layer_type = read_atlas_file[0]

    assert kwargs['name'] == 'VascularProbabilisticAtlas'
    assert kwargs['metadata']['source_path'].endswith('.nii') or kwargs['metadata']['source_path'].endswith('.nii.gz')
    assert napari_layer_type == 'image'
