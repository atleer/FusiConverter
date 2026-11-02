"""
Test putting a loaded recording on the axis order napari needs
"""
import pytest
import numpy as np

from fusiconverter.utils import prepare_image_for_layer


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
