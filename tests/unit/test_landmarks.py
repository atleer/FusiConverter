"""
Test saving, loading and matching landmarks
"""
import numpy as np
import napari

from fusiconverter.landmarks import match_landmarks, save_landmarks, load_landmarks


def make_points_layer(source_path, landmarks):
    """A napari Points layer (no viewer needed) holding 'landmarks' ({name: coordinates}), as AddLandmark builds it"""
    return napari.layers.Points(
        np.array(list(landmarks.values()), dtype=float),
        features={'landmark_name': np.array(list(landmarks), dtype=object)},
        metadata={'source_path': str(source_path)},
    )

# %% save_landmarks / load_landmarks

def test_load_landmarks_without_file(tmp_path):
    """A recording with no landmarks file has no landmarks"""
    assert load_landmarks(tmp_path / 'recording.mat') == {}

def test_save_and_load_landmarks(tmp_path):
    """Saved landmarks load back with the same names and coordinates, from <source>.landmarks.json"""
    source_path = tmp_path / 'recording.mat'
    landmarks = {'bregma': [1.0, 2.0, 3.0], 'lambda': [4.0, 5.0, 6.0]}

    save_landmarks(make_points_layer(source_path, landmarks))

    assert (tmp_path / 'recording.mat.landmarks.json').exists()
    assert load_landmarks(source_path) == landmarks

def test_save_landmarks_append(tmp_path):
    """Appending keeps the landmarks already on file, and a landmark with the same name is replaced by the new one"""
    source_path = tmp_path / 'recording.mat'
    save_landmarks(make_points_layer(source_path, {'bregma': [1.0, 2.0, 3.0], 'lambda': [4.0, 5.0, 6.0]}))

    save_landmarks(make_points_layer(source_path, {'lambda': [40.0, 50.0, 60.0], 'midline': [7.0, 8.0, 9.0]}), mode='append')

    assert load_landmarks(source_path) == {'bregma': [1.0, 2.0, 3.0], 'lambda': [40.0, 50.0, 60.0], 'midline': [7.0, 8.0, 9.0]}

def test_save_landmarks_overwrite(tmp_path):
    """Overwriting makes the file match the layer exactly, so landmarks only on file are dropped"""
    source_path = tmp_path / 'recording.mat'
    save_landmarks(make_points_layer(source_path, {'bregma': [1.0, 2.0, 3.0], 'lambda': [4.0, 5.0, 6.0]}))

    save_landmarks(make_points_layer(source_path, {'lambda': [40.0, 50.0, 60.0], 'midline': [7.0, 8.0, 9.0]}), mode='overwrite')

    assert load_landmarks(source_path) == {'lambda': [40.0, 50.0, 60.0], 'midline': [7.0, 8.0, 9.0]}

# %% match_landmarks

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
