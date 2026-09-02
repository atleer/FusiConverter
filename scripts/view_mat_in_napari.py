# %%
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

from src.fusiconverter.utils import *
from src.fusiconverter.viewer_ops import *
from src.fusiconverter.widgets import AddLandmark, AlignmentWidget, AlignToAtlasWidget, RegisterToAreasWidget#, CropWidget

# %%

mat_filepaths = select_files_from_gui("Navigate to experiment folder and select fUSI .mat Files to View")

images = {}
image_sources = {}
for mat_filepath in tqdm(mat_filepaths, desc='Loading Images...'):
    filename = Path(mat_filepath).stem
    image, voxel_size, origin, image_name = load_mat_image(mat_filepath)
    layer_name = f'{image_name}_{filename}'
    images[layer_name] = (np.abs(image), voxel_size)
    image_sources[layer_name] = mat_filepath



# Optionally load the Allen CCF atlas as an additional layer
atlas_path = prompt_load_atlas()
if atlas_path:
    print(f"Loading atlas from {atlas_path}...")
    images['Atlas'] = load_atlas_image(atlas_path)
    image_sources['Atlas'] = atlas_path

# View in Napari
print("Launching Napari Image Viewer...")
viewer = napari.Viewer()
for name, (image, voxel_size) in images.items():
    if voxel_size is None:
        print(f"No voxel size found for '{name}', displaying at raw voxel scale.")
    # moves a time axis to the front so the spatial axes line up with the 3D atlas
    try:
        image, scale, spatial_axes = prepare_image_for_layer(image, voxel_size)
    except ValueError as error:
        # one odd file shouldn't stop the rest of the session from opening
        print(f"'{name}': {error} Displaying at raw voxel scale.")
        image, scale, spatial_axes = prepare_image_for_layer(image, None)
    layer = viewer.add_image(name=name, data=image, scale=scale,
                             metadata={'source_path': image_sources[name],
                                       'spatial_axes': spatial_axes})

    # apply existing transformation matrix from file to layer
    alignment_path = Path(f"{image_sources[name]}.alignment.json")
    if alignment_path.exists():
        alignment = load_transform_matrix(alignment_path)
        layer.affine = put_transform_matrix_in_layer_affine(alignment['transform_matrix'], layer.ndim, spatial_axes)
        print(f"Applied saved transformation matrix for alignment to '{name}' "
              f"(RMS {alignment['rms_error']:.3f} mm)")

        points_layer = get_or_create_landmarks_layer(viewer, layer)   # loads the saved .landmarks.json
        points_layer.affine = put_transform_matrix_in_layer_affine(alignment['transform_matrix'], points_layer.ndim)


add_landmark_widget = AddLandmark(viewer)
viewer.window.add_dock_widget(add_landmark_widget, area='right')

alignment_widget = AlignmentWidget(viewer)
viewer.window.add_dock_widget(alignment_widget)

align_to_atlas_widget = AlignToAtlasWidget(viewer)
viewer.window.add_dock_widget(align_to_atlas_widget, area='right')

register_to_atlas_areas_widget = RegisterToAreasWidget(viewer)
viewer.window.add_dock_widget(register_to_atlas_areas_widget, area='right')

napari.run()

# %%
