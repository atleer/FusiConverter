"""
Test labelling a recording's voxels with atlas structures and summarising which structures it covers
"""
import numpy as np

from fusiconverter.viewer_ops import annotate_volume, summarize_areas


# %% annotate_volume

def test_annotate_identity_copies_annotation():
    """A recording on the same grid as the atlas, with no transform, gets exactly the atlas labels"""
    annotation = np.arange(1, 61).reshape(3, 4, 5)
    labels, fraction_outside = annotate_volume(annotation.shape, (0.05, 0.05, 0.05), np.eye(4), annotation, (0.05, 0.05, 0.05))
    assert np.array_equal(labels, annotation)
    assert fraction_outside == 0.0

def test_annotate_coarser_recording_samples_every_other_atlas_voxel():
    """A recording with voxels twice the atlas voxel size picks up every other atlas voxel"""
    annotation = np.arange(1, 6 * 8 * 10 + 1).reshape(6, 8, 10)
    labels, fraction_outside = annotate_volume((3, 4, 5), (0.1, 0.1, 0.1), np.eye(4), annotation, (0.05, 0.05, 0.05))
    assert np.array_equal(labels, annotation[::2, ::2, ::2])
    assert fraction_outside == 0.0

def test_annotate_voxels_outside_atlas_are_zero():
    """Recording voxels that land outside the atlas get label 0 and count towards the fraction outside"""
    annotation = np.arange(1, 41).reshape(2, 4, 5)
    labels, fraction_outside = annotate_volume((4, 4, 5), (0.05, 0.05, 0.05), np.eye(4), annotation, (0.05, 0.05, 0.05))
    assert np.array_equal(labels[:2], annotation)
    assert np.all(labels[2:] == 0)
    assert fraction_outside == 0.5

def test_annotate_translation_shifts_lookup():
    """Translating the recording by one atlas voxel looks up the labels one voxel further along"""
    annotation = np.arange(1, 61).reshape(3, 4, 5)
    transform_matrix = np.eye(4)
    transform_matrix[0, 3] = 0.05 # one atlas voxel along axis 0
    labels, fraction_outside = annotate_volume((2, 4, 5), (0.05, 0.05, 0.05), transform_matrix, annotation, (0.05, 0.05, 0.05))
    assert np.array_equal(labels, annotation[1:])
    assert fraction_outside == 0.0

# %% summarize_areas

def test_summarize_areas():
    """Structures are listed most voxels first with their names, counts, fractions and flat voxel indices, and background (0) is left out"""
    labels = np.array([[[0, 5, 5], [7, 7, 7], [9, 5, 7]]])
    structures = {
        5: {'acronym': 'CA1', 'name': 'Field CA1', 'division': 'HPF'},
        7: {'acronym': 'VISp', 'name': 'Primary visual area', 'division': 'Isocortex'},
    }

    summary = summarize_areas(labels, structures)

    assert [structure['id'] for structure in summary] == [7, 5, 9] # most voxels first, background (0) dropped
    assert [structure['n_voxels'] for structure in summary] == [4, 3, 1]
    assert summary[0]['acronym'] == 'VISp'
    assert summary[0]['division'] == 'Isocortex'
    assert np.isclose(summary[0]['fraction_of_recording'], 4 / 9)
    assert np.array_equal(summary[0]['voxel_indices'], [3, 4, 5, 8])

def test_summarize_areas_unknown_id():
    """A label that isn't in the structure graph is still listed, as 'unknown_<id>'"""
    summary = summarize_areas(np.array([[[9]]]), {})
    assert summary[0]['acronym'] == 'unknown_9'
    assert summary[0]['division'] == ''

def test_summarize_areas_all_background():
    """A recording with only background (0) gives an empty summary"""
    assert summarize_areas(np.zeros((2, 2, 2), dtype=np.uint32), {}) == []
