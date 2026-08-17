import sys
from pathlib import Path


import h5py
import napari
import numpy as np
from tqdm import tqdm
import os
from pathlib import Path

root_dir = Path(__file__).parent.parent
os.chdir(root_dir)
sys.path.insert(0, str(root_dir))

from src.utils import *
from src.viewer_ops import *
from src.widgets import AddLandmark, RegistrationWidget#, CropWidget


h5_filepaths = select_files_from_gui("Navigate to experiment folder and select fUSI H5 Files to View")
if not h5_filepaths:
    print("No Files Selected.  Quitting...")
    sys.exit()

# Load images
images = {}
image_sources = {}
for h5_filepath in tqdm(h5_filepaths, desc='Loading Images...'):
    with h5py.File(h5_filepath) as data:
        filename = Path(h5_filepath).stem
        image_type = data.attrs['imageType']
        layer_name = f'{image_type}_{filename}'
        images[layer_name] = (np.abs(data['image']), get_voxel_size_mm(data))
        image_sources[layer_name] = h5_filepath

# Optionally load the Allen CCF atlas as an additional layer
atlas_path = prompt_load_atlas()
if atlas_path:
    print(f"Loading atlas from {atlas_path}...")
    images['Atlas'] = load_atlas_image(atlas_path)
    image_sources['Atlas'] = atlas_path

# View in Napari
print("Launching Napari Image Viewer...")
viewer = napari.Viewer()
for name, (image, scale) in images.items():
    if scale is None:
        print(f"No voxel size found for '{name}', displaying at raw voxel scale.")
        scale = (1,) * image.ndim
    elif image.ndim > len(scale): # only needed if the image has more dimensions than the scale (e.g. a time dimension).
        # Non-spatial trailing axes (e.g. time) get a scale of 1.
        scale = tuple(scale) + (1,) * (image.ndim - len(scale))
    layer = viewer.add_image(name=name, data=image, scale=scale, metadata={'source_path': image_sources[name]})

    registration_path = Path(f"{image_sources[name]}.registration.json")
    if registration_path.exists():
        registration = load_registration(registration_path)
        layer.affine = to_layer_affine(registration['transform_matrix'], layer.ndim)
        print(f"Applied saved registration to '{name}' "
              f"(RMS {registration['rms_error']:.3f} mm)")

        points_layer = get_or_create_landmarks_layer(viewer, layer)   # loads the saved .landmarks.json
        points_layer.affine = to_layer_affine(registration['transform_matrix'], points_layer.ndim)


# crop_widget = CropWidget(viewer)
# viewer.window.add_dock_widget(crop_widget, area='right')

add_landmark_widget = AddLandmark(viewer)
viewer.window.add_dock_widget(add_landmark_widget, area='right')

registration_widget = RegistrationWidget(viewer)
viewer.window.add_dock_widget(registration_widget)

napari.run()
