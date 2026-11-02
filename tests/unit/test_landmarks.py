"""
Test matching landmarks by name between two layers
"""
import numpy as np

from fusiconverter.viewer_ops import match_landmarks


def test_match_landmarks_keeps_shared_names_sorted():
    """Only landmarks named in both layers are kept, sorted by name, with their coordinates in the same order"""
    landmarks_a = {'bregma': [1, 2, 3], 'lambda': [4, 5, 6], 'only_a': [0, 0, 0]}
    landmarks_b = {'lambda': [40, 50, 60], 'bregma': [10, 20, 30], 'only_b': [9, 9, 9]}

    names, coords_a, coords_b = match_landmarks(landmarks_a, landmarks_b)

    assert names == ['bregma', 'lambda']
    assert np.array_equal(coords_a, [[1, 2, 3], [4, 5, 6]])
    assert np.array_equal(coords_b, [[10, 20, 30], [40, 50, 60]])
    assert coords_a.dtype == float

def test_match_landmarks_no_overlap():
    """Layers with no landmark names in common give no matches"""
    names, coords_a, coords_b = match_landmarks({'a': [1, 2, 3]}, {'b': [1, 2, 3]})
    assert names == []
    assert len(coords_a) == 0
    assert len(coords_b) == 0
