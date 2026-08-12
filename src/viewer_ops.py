import napari
import numpy as np
from pathlib import Path
import json

def load_landmarks(source_path):
    """Load landmarks that have already been written to file, if there are any"""
    json_path = Path(f'{source_path}.landmarks.json')
    if not json_path.exists():
        return {}
    with open(json_path) as f:
        return json.load(f)

def save_landmarks(points_layer, mode='append'):
    source_path = points_layer.metadata.get('source_path') # source path stems from name of data that the landmarks are put on (e.g. atlas or the recorded image)

    features = points_layer.features # names of landmarks and a list of booleans ('from_disk') indicating whether the landmark was added in this session or loaded from existing file are in features
    landmarks = {}
    for landmark_name, coords_landmark, from_disk in zip(features['landmark_name'], points_layer.data, features['from_disk']):
        # overwrite means only the landmarks added in this session are kept
        if mode == 'overwrite' and from_disk: # from_disk is just a boolean value
            continue
        landmarks[landmark_name] = coords_landmark.tolist() # added if the landmark doesn't already exist on file or the landmark already exists but the user has chosen to append

    if mode == 'append':
        # add new landmarks to existing landmarks on disk. If new landmarks have same name as landmarks on disk, new landmarks take precedence and will overwrite landmarks on disk
        merged = load_landmarks(source_path)
        merged.update(landmarks)
        landmarks = merged

    with open(Path(f'{source_path}.landmarks.json'), 'w') as f:
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
        features={
            'landmark_name': np.array(list(saved_landmarks.keys()), dtype=object),
            'from_disk': np.ones(len(saved_landmarks), dtype=bool),
            },
        text='landmark_name',
        metadata={'source_path': source_path},
    )
    return points_layer

# # Function to crop image in spatial dimensions (no temporal)
# def crop_image(viewer, z_range, x_range, y_range):
#     layer = viewer.layers.selection.active
#     if layer is None or not isinstance(layer, napari.layers.Image):
#         print("No image layer selected.")
#         return
    
#     # Get the shape of the image
#     shape = layer.data.shape
#     z_shape, x_shape, y_shape = shape[0], shape[1], shape[2]

#     # Set default ranges if not provided
#     if not z_range:
#         z_range = f'0:{z_shape}'
#     if not x_range:
#         x_range = f'0:{x_shape}'
#     if not y_range:
#         y_range = f'0:{y_shape}'
        
#     # Parse the input ranges
#     z_min, z_max = map(int, z_range.split(':'))
#     x_min, x_max = map(int, x_range.split(':'))
#     y_min, y_max = map(int, y_range.split(':'))

#     # Crop the image
#     cropped_image = layer.data[z_min:z_max, x_min:x_max, y_min:y_max]

#     # Add the cropped image as a new layer
#     viewer.add_image(cropped_image, name=f'Cropped_{layer.name}')

# def apply_log_normalization(viewer):
#     layer = viewer.layers.selection.active
#     if layer is None or not isinstance(layer, napari.layers.Image):
#         print("No image layer selected.")
#         return
#     # Map the image data to [0, 1] range
#     data_min = np.min(layer.data)
#     data_max = np.max(layer.data)
#     scaled_data = (layer.data - data_min) / (data_max - data_min)

#     # Apply logarithmic normalization
#     normalized_data = np.log1p(scaled_data)
    
#     # Add the normalized image as a new layer
#     viewer.add_image(normalized_data, name=f'LogNorm_{layer.name}')
