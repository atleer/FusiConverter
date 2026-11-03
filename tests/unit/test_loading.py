"""
Test loading recordings and atlases from file, and putting a loaded recording on the axis order napari needs
"""
import pytest
import numpy as np

from fusiconverter.utils import prepare_image_for_layer, load_mat_image, load_atlas_image


# %% prepare_image_for_layer

def test_prepare_3d():
    """A 3D image is left as it is, with the voxel size as scale and all three axes spatial"""
    image = np.zeros((2, 3, 4))
    prepared, scale, spatial_axes = prepare_image_for_layer(image, (0.1, 0.2, 0.3))
    assert prepared.shape == (2, 3, 4)
    assert scale == (0.1, 0.2, 0.3)
    assert spatial_axes == (0, 1, 2)

def test_prepare_4d_moves_time_to_front():
    """A 4D image has its trailing time axis moved to the front, with scale 1 for time and the spatial axes last"""
    image = np.random.default_rng(0).random((2, 3, 4, 5)) # (Z, X, Y, T)
    prepared, scale, spatial_axes = prepare_image_for_layer(image, (0.1, 0.2, 0.3))
    assert prepared.shape == (5, 2, 3, 4)
    assert np.array_equal(prepared[1], image[..., 1])
    assert scale == (1.0, 0.1, 0.2, 0.3)
    assert spatial_axes == (1, 2, 3)

def test_prepare_without_voxel_size_leaves_image_alone():
    """With no voxel size, the image isn't moved, the scale is all 1s and the last three axes count as spatial"""
    image = np.zeros((2, 3, 4, 5))
    prepared, scale, spatial_axes = prepare_image_for_layer(image, None)
    assert prepared.shape == (2, 3, 4, 5)
    assert scale == (1.0, 1.0, 1.0, 1.0)
    assert spatial_axes == (1, 2, 3)

@pytest.mark.parametrize('shape', [(2, 3), (2, 3, 4, 5, 6)])
def test_prepare_dimension_mismatch_raises(shape):
    """An image that isn't 3D or 3D + time raises a ValueError, since it's unclear which axes are spatial"""
    with pytest.raises(ValueError):
        prepare_image_for_layer(np.zeros(shape), (0.1, 0.2, 0.3))

# %% load_mat_image

@pytest.fixture
def image():
    """A small, reproducible 3D recording"""
    return np.random.default_rng(0).random((2, 3, 4))

@pytest.mark.parametrize('image_name', ['bmode', 'doppler', 'Ihq', 'I'])
def test_load_mat_finds_image_under_each_name(make_mat, image, image_name):
    """The image is found whichever of the variable names used by the different acquisition modes it is stored under"""
    loaded, _, _, loaded_name = load_mat_image(make_mat(image, image_name=image_name))
    assert np.array_equal(loaded, image)
    assert loaded_name == image_name

def test_load_mat_without_known_image_name_raises(make_mat, image):
    """A file with none of the known image variable names ("doppler", "bmode", "Ihq", or "I") raises a ValueError"""
    with pytest.raises(ValueError):
        load_mat_image(make_mat(image, image_name='something_else'))

def test_load_mat_reads_voxel_size_and_origin(make_mat, image):
    """Voxel size and origin are read from the metadata struct as tuples of floats"""
    path = make_mat(image, metadata={'voxelSize': [0.1, 0.2, 0.3], 'origin': [1, 2, 3]})
    _, voxel_size, origin, _ = load_mat_image(path)
    assert voxel_size == (0.1, 0.2, 0.3)
    assert origin == (1.0, 2.0, 3.0)

def test_load_mat_without_metadata(make_mat, image):
    """A file without a metadata struct loads, with no voxel size and no origin"""
    _, voxel_size, origin, _ = load_mat_image(make_mat(image))
    assert voxel_size is None
    assert origin is None

def test_load_mat_ignores_tag(make_mat, image):
    """The 'tag' metadata in tagged/stimulus recordings doesn't get in the way of loading"""
    path = make_mat(image, metadata={'voxelSize': [0.1, 0.2, 0.3], 'tag': 'stimulus'})
    loaded, voxel_size, _, _ = load_mat_image(path)
    assert np.array_equal(loaded, image)
    assert voxel_size == (0.1, 0.2, 0.3)

def test_load_mat_keeps_complex_values(make_mat, image):
    """Complex ultrasound data comes back complex and unchanged (taking the magnitude is left to the reader)"""
    complex_image = image + 1j * image[::-1]
    loaded, _, _, _ = load_mat_image(make_mat(complex_image))
    assert np.iscomplexobj(loaded)
    assert np.array_equal(loaded, complex_image)

def test_load_mat_keeps_trailing_time_axis(make_mat):
    """A recording with a time axis comes back with time still last (moving it is left to prepare_image_for_layer)"""
    image = np.zeros((2, 3, 4, 5)) # (Z, X, Y, T)
    loaded, _, _, _ = load_mat_image(make_mat(image))
    assert loaded.shape == (2, 3, 4, 5)

# %% load_atlas_image

@pytest.mark.parametrize('zoom, voxel_size_mm', [
    (50, 0.05), # Allen CCF, header in um
    (1, 0.001), # smallest value still read as um
    (0.78125, 0.078125), # 78um Keliris vascular atlas, header 10x too big
    (0.2, 0.02), # smallest value still read as 10x too big
    (0.078125, 0.078125), # Keliris VascularProbabilisticAtlas, header already in mm
])
def test_load_atlas_converts_voxel_size_to_mm(make_nifti, zoom, voxel_size_mm):
    """Rodent MRI atlases are sometimes scaled up 10x to make neuroimaging tools built for human brains accept them.
    When the atlas is loaded here, the header voxel size is checked and scaled to be right for a mouse brain if necessary.
    The unit label in the atlas header is wrong in some cases (see loading.py for more details), so the voxel size is assumed based on what is 
    biologically plausible (e.g. that the mouse brain is not as big as the human brain). This function checks that the 
    scaling based on that assumption is carried out correctly."""
    data, scale_mm = load_atlas_image(make_nifti((zoom, zoom, zoom)))
    assert scale_mm == pytest.approx((voxel_size_mm,) * 3)
    assert data.shape == (4, 5, 6)
