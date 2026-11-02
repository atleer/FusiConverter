"""
Test the transform maths: fitting landmarks, composing manual adjustments and moving transforms in and out of napari layer affines
"""
import pytest
import numpy as np

from fusiconverter.viewer_ops import (
    fit_similarity_transform,
    transform_residuals,
    compose_transform_matrix,
    put_transform_matrix_in_layer_affine,
    get_transform_matrix_from_layer_affine,
    atlas_index_transform,
    save_transform_matrix,
    load_transform_matrix,
)


def apply_transform(transform_matrix, points):
    """Map (n, 3) points through a 4x4 homogeneous transform"""
    return points @ transform_matrix[:3, :3].T + transform_matrix[:3, 3]

@pytest.fixture
def landmarks():
    """Six random, reproducible landmark positions (mm) to transform and fit"""
    return np.random.default_rng(0).uniform(-3, 3, size=(6, 3))

# %% fit_similarity_transform

def test_fit_recovers_rigid_transform(landmarks, random_similarity_transform):
    """Moving the landmarks by a known rotation + translation and fitting them gives back that same transform"""
    known = random_similarity_transform(seed=1)
    fitted = fit_similarity_transform(landmarks, apply_transform(known, landmarks)) # default allow_scaling=False
    assert np.allclose(fitted, known)

def test_fit_recovers_rotation_and_uniform_scale(landmarks, random_similarity_transform):
    """With allow_scaling=True, a known rotation + uniform scale + translation is recovered too"""
    known = random_similarity_transform(seed=2, scale=1.7)
    fitted = fit_similarity_transform(landmarks, apply_transform(known, landmarks), allow_scaling=True)
    assert np.allclose(fitted, known)

def test_fit_ignores_scaling_by_default(landmarks, random_similarity_transform):
    """Check that landmarks that were scaled still give a pure rotation when scaling isn't allowed (the default)"""
    known = random_similarity_transform(seed=3, scale=1.7)
    fitted = fit_similarity_transform(landmarks, apply_transform(known, landmarks))
    linear = fitted[:3, :3]
    assert np.allclose(linear @ linear.T, np.eye(3))

def test_fit_never_returns_a_mirror(landmarks):
    """The optimal way to match mirrored landmarks would be a reflection, but that would effectively swap the left and right hemispheres, which makes no sense biologically. 
    Therefore, the fit shouldn't be a reflection even if that minimizes the error. This function checks that the fitted transform is not a reflection, but a proper rotation (determinant +1), 
    even if it means that the error is larger."""
    mirrored = landmarks * [-1, 1, 1]
    fitted = fit_similarity_transform(landmarks, mirrored)
    assert np.isclose(np.linalg.det(fitted[:3, :3]), 1.0)

def test_fit_needs_three_landmarks(landmarks):
    """Fewer than 3 landmark pairs raises a ValueError"""
    with pytest.raises(ValueError):
        fit_similarity_transform(landmarks[:2], landmarks[:2])

# %% transform_residuals

def test_residuals_are_zero_for_exact_fit(landmarks, random_similarity_transform):
    """Landmarks that the transform maps exactly onto their targets have zero residuals"""
    known = random_similarity_transform(seed=4)
    residuals = transform_residuals(known, landmarks, apply_transform(known, landmarks))
    assert np.allclose(residuals, 0)

def test_residuals_are_distance_per_landmark(landmarks):
    """Each residual is the distance (mm) between a transformed landmark and its target"""
    fixed = landmarks.copy()
    fixed[0] += [3, 4, 0] # 5 mm off
    residuals = transform_residuals(np.eye(4), landmarks, fixed)
    assert np.allclose(residuals, [5, 0, 0, 0, 0, 0])

# %% compose_transform_matrix

def test_compose_identity():
    """No translation, no rotation and scale 1 give the identity matrix"""
    assert np.allclose(compose_transform_matrix((0, 0, 0), (0, 0, 0), (1, 1, 1)), np.eye(4))

def test_compose_keeps_centre_in_place():
    """Rotating and scaling about a centre leaves that centre where it was"""
    centre = np.array([1.0, 2.0, 3.0])
    transform_matrix = compose_transform_matrix((0, 0, 0), (30, -20, 45), (1.5, 0.8, 1.2), centre)
    assert np.allclose(apply_transform(transform_matrix, centre[None]), centre)

def test_compose_translation_moves_centre_by_translation():
    """With a translation added, the centre moves by exactly that translation"""
    centre = np.array([1.0, 2.0, 3.0])
    translation = np.array([0.5, -1.0, 2.0])
    transform_matrix = compose_transform_matrix(translation, (30, -20, 45), (1.5, 0.8, 1.2), centre)
    assert np.allclose(apply_transform(transform_matrix, centre[None]), centre + translation)

def test_compose_scale_is_per_axis():
    """Without rotation, each scale factor stretches its own axis only"""
    transform_matrix = compose_transform_matrix((0, 0, 0), (0, 0, 0), (2, 3, 4))
    assert np.allclose(transform_matrix[:3, :3], np.diag([2, 3, 4]))

# %% layer affine

def test_layer_affine_round_trip_3d(random_similarity_transform):
    """For a 3D layer the affine is the transform itself, and reading it back gives the same transform"""
    known = random_similarity_transform(seed=5)
    affine = put_transform_matrix_in_layer_affine(known, ndim=3)
    assert np.allclose(affine, known)
    assert np.allclose(get_transform_matrix_from_layer_affine(affine), known)

def test_layer_affine_round_trip_4d_leaves_time_alone(random_similarity_transform):
    """For a 4D layer with a leading time axis, the transform goes on the spatial axes only and reads back unchanged"""
    known = random_similarity_transform(seed=6)
    spatial_axes = (1, 2, 3) # leading time axis
    affine = put_transform_matrix_in_layer_affine(known, ndim=4, spatial_axes=spatial_axes)
    assert affine.shape == (5, 5)
    # the affine has a first row and a first column with 1 and then only 0s. That ensures that the time axis is not transformed when the transform is applied to a matrix with a time dimension. The first two asserts checks that this row and column is correct
    assert np.allclose(affine[0], [1, 0, 0, 0, 0]) # time row untouched
    assert np.allclose(affine[:, 0], [1, 0, 0, 0, 0]) # time column untouched
    assert np.allclose(get_transform_matrix_from_layer_affine(affine, spatial_axes), known)

# %% atlas_index_transform

def test_atlas_index_transform_matches_going_through_mm(random_similarity_transform):
    """Going straight from recording voxel indices to atlas voxel indices gives the same result as indices -> mm -> transform -> atlas indices"""
    known = random_similarity_transform(seed=7)
    voxel_size_mm = np.array([0.1, 0.2, 0.3])
    atlas_voxel_size_mm = np.array([0.05, 0.05, 0.05])
    voxel_indices = np.array([[0, 0, 0], [1, 2, 3], [10, 5, 7]], dtype=float)

    matrix, offset = atlas_index_transform(voxel_size_mm, known, atlas_voxel_size_mm)

    expected = apply_transform(known, voxel_indices * voxel_size_mm) / atlas_voxel_size_mm
    assert np.allclose(voxel_indices @ matrix.T + offset, expected)

# %% save_transform_matrix / load_transform_matrix

def test_save_and_load_landmark_alignment(tmp_path, random_similarity_transform):
    """A landmark alignment saved next to the recording loads back with the same matrix, voxel sizes, residuals per landmark and their RMS"""
    known = random_similarity_transform(seed=8)
    hq_source_path = tmp_path / 'recording.mat'

    json_path = save_transform_matrix(hq_source_path, 'atlas.nii', known, (0.1, 0.2, 0.3), (0.05, 0.05, 0.05),
                                      landmark_names=['bregma', 'lambda'], residuals_mm=[3.0, 4.0])
    alignment = load_transform_matrix(json_path)

    assert json_path == tmp_path / 'recording.mat.alignment.json'
    assert isinstance(alignment['transform_matrix'], np.ndarray)
    assert np.allclose(alignment['transform_matrix'], known)
    assert alignment['hq_source_path'] == str(hq_source_path)
    assert alignment['atlas_source_path'] == 'atlas.nii'
    assert alignment['method'] == 'landmarks'
    assert alignment['hq_voxel_size_in_mm'] == [0.1, 0.2, 0.3]
    assert alignment['atlas_voxel_size_in_mm'] == [0.05, 0.05, 0.05]
    assert alignment['landmark_names'] == ['bregma', 'lambda']
    assert alignment['residuals_mm'] == {'bregma': 3.0, 'lambda': 4.0}
    assert np.isclose(alignment['rms_error'], np.sqrt((3.0**2 + 4.0**2) / 2))
    assert 'created' in alignment
