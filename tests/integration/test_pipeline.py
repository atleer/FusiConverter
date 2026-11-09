"""
Test the whole chain from landmarks to a saved area map, on a small synthetic recording and atlas where the right answer is known
"""
import json
import pytest
import numpy as np

from fusiconverter.reader import read_mat
from fusiconverter.landmarks import load_landmarks, match_landmarks
from fusiconverter.viewer_ops import (
    annotate_volume,
    summarize_areas,
    save_area_map,
)
from fusiconverter.transforms import (
    fit_similarity_transform,
    transform_residuals,
    save_transform_matrix,
    get_transform_matrix_from_layer_affine,
)


RECORDING_VOXEL_SIZE = (0.1, 0.1, 0.1)
ATLAS_VOXEL_SIZE = (0.05, 0.05, 0.05)

# The recording sits in the atlas rotated 90 degrees about axis 0 and shifted: recording mm (a, b, c) -> atlas mm (a + 0.05, 0.3 - c, b).
# With these voxel sizes, recording voxel (i, j, k) then lands exactly on atlas voxel (2i + 1, 6 - 2k, 2j).
KNOWN_TRANSFORM = np.array([
    [1, 0, 0, 0.05],
    [0, 0, -1, 0.3],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
])

# landmarks are saved in voxel indices of the volume they're placed on
RECORDING_LANDMARKS = {'a': [0, 0, 0], 'b': [1, 0, 0], 'c': [0, 2, 0], 'd': [0, 0, 3], 'e': [1, 2, 3]}
ATLAS_LANDMARKS = {'a': [1, 6, 0], 'b': [3, 6, 0], 'c': [1, 6, 4], 'd': [1, 0, 0], 'e': [3, 0, 4]} # (2i + 1, 6 - 2k, 2j)

# every atlas voxel has its own label, so a recording voxel looking up the wrong atlas voxel gets the wrong label
ANNOTATION = np.arange(1, 4 * 7 * 5 + 1).reshape(4, 7, 5)


@pytest.mark.parametrize('image_shape', [(2, 3, 4), (2, 3, 4, 5)], ids=['3d', '4d'])
def test_landmarks_to_area_map(tmp_path, make_mat, image_shape):
    """Landmarks on a recording and an atlas give back the known transform, which is saved, picked up again when the recording is opened,
    and used to label each recording voxel with the atlas voxel it lands on, for both a 3D and a 3D + time recording"""
    recording_path = make_mat(np.ones(image_shape), metadata={'voxelSize': RECORDING_VOXEL_SIZE})
    atlas_path = tmp_path / 'atlas.nii'
    for source_path, landmarks in [(recording_path, RECORDING_LANDMARKS), (atlas_path, ATLAS_LANDMARKS)]:
        with open(f'{source_path}.landmarks.json', 'w') as f:
            json.dump(landmarks, f)

    # fit the landmarks in mm, as AlignmentWidget.align does
    names, recording_points, atlas_points = match_landmarks(load_landmarks(recording_path), load_landmarks(atlas_path))
    moving = recording_points * RECORDING_VOXEL_SIZE
    fixed = atlas_points * ATLAS_VOXEL_SIZE
    transform_matrix = fit_similarity_transform(moving, fixed)
    residuals = transform_residuals(transform_matrix, moving, fixed)
    assert np.allclose(transform_matrix, KNOWN_TRANSFORM)
    assert np.allclose(residuals, 0)
    save_transform_matrix(recording_path, atlas_path, transform_matrix, RECORDING_VOXEL_SIZE, ATLAS_VOXEL_SIZE, names, residuals)

    # opening the recording again puts the saved transform in the layer affine, where RegisterToAreasWidget reads it back from
    [(image, kwargs, _)] = read_mat(recording_path, ask=False)
    spatial_axes = list(kwargs['metadata']['spatial_axes'])
    transform_matrix = get_transform_matrix_from_layer_affine(kwargs['affine'], spatial_axes)
    assert np.allclose(transform_matrix, KNOWN_TRANSFORM)

    spatial_shape = tuple(np.asarray(image.shape)[spatial_axes].tolist())
    voxel_size = np.asarray(kwargs['scale'])[spatial_axes]
    labels, fraction_outside = annotate_volume(spatial_shape, voxel_size, transform_matrix, ANNOTATION, ATLAS_VOXEL_SIZE)
    assert np.array_equal(labels, ANNOTATION[1::2, 6::-2, 0::2].transpose(0, 2, 1)) # labels[i, j, k] == ANNOTATION[2i + 1, 6 - 2k, 2j] the numbers in the ANNOTATION indeces are specific to the synthetic data above
    assert fraction_outside == 0.0

    structures = {int(labels[0, 0, 0]): {'acronym': 'AAA', 'name': 'Area A', 'division': 'Div'}}
    summary = summarize_areas(labels, structures)
    json_path = save_area_map(recording_path, summary, labels, 'annotation.nii.gz', fraction_outside)
    with open(json_path) as f:
        area_map = json.load(f)
    assert area_map['hq_shape'] == [2, 3, 4]
    assert area_map['n_voxels_unlabelled'] == 0
    # each recording voxel landed on a different atlas voxel, so each label is one structure covering one voxel
    assert sorted(structure['id'] for structure in area_map['structures']) == sorted(labels.ravel().tolist())
    by_id = {s['id']: s for s in area_map['structures']}
    assert by_id[int(labels[0, 0, 0])]['acronym'] == 'AAA'            # found in the graph
    assert by_id[int(labels[0, 0, 1])]['acronym'].startswith('unknown_')  # not in the graph
