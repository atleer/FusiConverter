import napari
import numpy as np
import json
from .paths import registration_file_path


def load_landmarks(source_path):
    """Load landmarks that have already been written to file, if there are any"""
    json_path = registration_file_path(source_path, 'landmarks.json')
    if not json_path.exists():
        return {}
    with open(json_path) as f:
        return json.load(f)

def save_landmarks(points_layer, mode='append'):
    source_path = points_layer.metadata.get('source_path') # source path stems from name of data that the landmarks are put on (e.g. atlas or the recorded image)

    features = points_layer.features # names of landmarks are in features
    # overwrite means the file ends up matching the layer exactly: loaded landmarks are kept, landmarks deleted in napari are dropped
    landmarks = {}
    for landmark_name, coords_landmark in zip(features['landmark_name'], points_layer.data):
        landmarks[landmark_name] = coords_landmark.tolist()

    if mode == 'append':
        # add new landmarks to existing landmarks on disk. If new landmarks have same name as landmarks on disk, new landmarks take precedence and will overwrite landmarks on disk
        merged = load_landmarks(source_path)
        merged.update(landmarks)
        landmarks = merged

    json_path = registration_file_path(source_path, 'landmarks.json')
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, 'w') as f:
        json.dump(landmarks, f, indent=2)

def get_or_create_landmarks_layer(viewer, image_layer) -> napari.layers.Points:
    points_name = f'{image_layer.name}_landmarks'
    if points_name in viewer.layers:
        return viewer.layers[points_name]

    source_path = image_layer.metadata.get('source_path')

    saved_landmarks = load_landmarks(source_path) # this causes new landmarks to be appended to landmarks already set and written to file, if there are any

    points_layer = viewer.add_points(
        data = np.array(list(saved_landmarks.values())),
        name = points_name, # takes its name from the filename - atlas or HQ volume
        ndim=image_layer.ndim,
        scale=image_layer.scale,
        affine=image_layer.affine.affine_matrix,
        features={
            'landmark_name': np.array(list(saved_landmarks.keys()), dtype=object),
            'from_disk': np.ones(len(saved_landmarks), dtype=bool),
            },
        text='landmark_name',
        metadata={'source_path': source_path},
    )
    return points_layer

def match_landmarks(landmarks_a: dict, landmarks_b: dict):
    """Gets the landmarks in common between layers a and b and their respective coordinates"""
    names = sorted(set(landmarks_a) & set(landmarks_b))
    coords_a = np.array([landmarks_a[name] for name in names], dtype=float)
    coords_b = np.array([landmarks_b[name] for name in names], dtype=float)
    return names, coords_a, coords_b
