"""
Test the napari readers on small synthetic files: which reader a file gets, the layers they return and when a saved alignment is applied
"""
import pytest
import numpy as np
from qtpy.QtWidgets import QMessageBox

from fusiconverter.reader import napari_get_reader, read_mat, read_atlas
from fusiconverter.viewer_ops import save_transform_matrix


def save_alignment(source_path, transform_matrix):
    """Save 'transform_matrix' as the alignment of 'source_path' to the vascular reference atlas"""
    save_transform_matrix(source_path, 'VascularProbabilisticAtlas.nii', transform_matrix, (0.1, 0.2, 0.3), (0.078125,) * 3)

def answer_question(monkeypatch, answer):
    """Make the 'Apply saved alignment?' dialog answer 'answer' without showing it, and return the list of dialog titles asked"""
    asked = []
    def question(parent, title, text):
        asked.append(title)
        return answer
    monkeypatch.setattr(QMessageBox, 'question', question)
    return asked

VOXEL_SIZE = {'voxelSize': [0.1, 0.2, 0.3]}

# %% napari_get_reader

@pytest.mark.parametrize('path, reader', [
    ('recording.mat', read_mat),
    ('RECORDING.MAT', read_mat),
    ('atlas.nii', read_atlas),
    ('atlas.nii.gz', read_atlas),
    ('recording.h5', None),
])
def test_get_reader(path, reader):
    """.mat files go to read_mat, .nii and .nii.gz files to read_atlas, whatever the case of the extension, and anything else gets no reader"""
    assert napari_get_reader(path) is reader

# %% read_mat

def test_read_mat_3d(make_mat, image):
    """A 3D recording becomes one image layer named after the image variable and the file, with the voxel size as scale, all axes spatial and no affine"""
    path = make_mat(image, metadata=VOXEL_SIZE)
    [(data, kwargs, layer_type)] = read_mat(path, ask=False)
    assert layer_type == 'image'
    assert np.array_equal(data, image)
    assert kwargs['name'] == 'Ihq_recording'
    assert kwargs['scale'] == (0.1, 0.2, 0.3)
    assert kwargs['metadata'] == {'source_path': str(path), 'spatial_axes': (0, 1, 2)}
    assert 'affine' not in kwargs

def test_read_mat_4d_puts_time_first(make_mat):
    """A recording with a trailing time axis gets time moved to the front, with scale 1 for time and the spatial axes last"""
    image = np.random.default_rng(0).random((2, 3, 4, 5)) # (Z, X, Y, T)
    [(data, kwargs, _)] = read_mat(make_mat(image, metadata=VOXEL_SIZE), ask=False)
    assert data.shape == (5, 2, 3, 4)
    assert np.array_equal(data[1], image[..., 1]) # the second element of the first axis of data (after reading in the original matrix) should be the same as the second element of the last axis in the original matrix
    assert kwargs['scale'] == (1.0, 0.1, 0.2, 0.3)
    assert kwargs['metadata']['spatial_axes'] == (1, 2, 3)

def test_read_mat_takes_magnitude_of_complex(make_mat, image):
    """Complex ultrasound data is shown as its magnitude"""
    complex_image = image + 1j * image[::-1]
    [(data, _, _)] = read_mat(make_mat(complex_image, metadata=VOXEL_SIZE), ask=False)
    assert not np.iscomplexobj(data)
    assert np.allclose(data, np.abs(complex_image))

def test_read_mat_applies_alignment_if_user_says_yes(make_mat, image, random_similarity_transform, monkeypatch):
    """With a saved alignment, the user is asked whether to apply it, and answering Yes puts it in the layer affine"""
    asked = answer_question(monkeypatch, QMessageBox.Yes)
    path = make_mat(image, metadata=VOXEL_SIZE)
    known = random_similarity_transform(seed=0)
    save_alignment(path, known)

    [(_, kwargs, _)] = read_mat(path) # ask=True is the default
    assert asked == ['Apply saved alignment?']
    assert np.allclose(kwargs['affine'], known)

def test_read_mat_skips_alignment_if_user_says_no(make_mat, image, random_similarity_transform, monkeypatch):
    """With a saved alignment, answering No leaves the layer unaligned"""
    asked = answer_question(monkeypatch, QMessageBox.No)
    path = make_mat(image, metadata=VOXEL_SIZE)
    save_alignment(path, random_similarity_transform(seed=0))

    [(_, kwargs, _)] = read_mat(path)
    assert asked == ['Apply saved alignment?']
    assert 'affine' not in kwargs

# %% read_atlas

def test_read_atlas(make_nifti):
    """An atlas becomes one image layer named after the file without .nii/.nii.gz, scaled to mm, with additive blending and no affine"""
    path = make_nifti((50, 50, 50))
    [(data, kwargs, layer_type)] = read_atlas(path, ask=False)
    assert layer_type == 'image'
    assert data.shape == (4, 5, 6)
    assert kwargs['name'] == 'atlas'
    assert kwargs['scale'] == pytest.approx((0.05, 0.05, 0.05))
    assert kwargs['blending'] == 'additive'
    assert kwargs['metadata'] == {'source_path': str(path)}
    assert 'affine' not in kwargs

def test_read_atlas_applies_alignment_in_allen_folder(monkeypatch, tmp_path, make_nifti, random_similarity_transform):
    """An atlas in an Allen_CCF folder with a saved alignment is moved onto the vascular reference atlas"""
    asked = answer_question(monkeypatch, QMessageBox.Yes)
    (tmp_path / 'Allen_CCF').mkdir()
    path = make_nifti((50, 50, 50), filename='Allen_CCF/atlas.nii.gz')
    known = random_similarity_transform(seed=0)
    save_alignment(path, known)

    [(_, kwargs, _)] = read_atlas(path, ask=True)

    assert asked == ['Apply saved alignment?']
    assert np.allclose(kwargs['affine'], known)

def test_read_atlas_ignores_alignment_outside_allen_folder(tmp_path, make_nifti, random_similarity_transform):
    """An atlas in any other folder is never moved, even with a saved alignment next to it, since the vascular atlas is the reference frame"""
    (tmp_path / 'Keliris_vasc_atlas').mkdir()
    path = make_nifti((0.078125,) * 3, filename='Keliris_vasc_atlas/atlas.nii.gz')
    save_alignment(path, random_similarity_transform(seed=0))

    [(_, kwargs, _)] = read_atlas(path, ask=False)

    assert 'affine' not in kwargs
