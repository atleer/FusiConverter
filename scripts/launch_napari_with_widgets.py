# %%
from pathlib import Path


import napari
import os

root_dir = Path(__file__).parent.parent
os.chdir(root_dir)

from fusiconverter.reader import read_atlas
from fusiconverter.widgets import AddLandmark, AlignmentWidget, ManualAlignmentWidget, RegisterToAreasWidget#, CropWidget

# %%

# View in Napari
print("Launching Napari Image Viewer...")
viewer = napari.Viewer()

add_landmark_widget = AddLandmark(viewer)
viewer.window.add_dock_widget(add_landmark_widget, area='right')

alignment_widget = AlignmentWidget(viewer)
viewer.window.add_dock_widget(alignment_widget)

manual_alignment_widget = ManualAlignmentWidget(viewer)
viewer.window.add_dock_widget(manual_alignment_widget, area='right')

register_to_atlas_areas_widget = RegisterToAreasWidget(viewer)
viewer.window.add_dock_widget(register_to_atlas_areas_widget, area='right')

# Load the atlases on startup, applying the Allen atlas's saved alignment without asking.
# The data folder sits next to FusiConverter.
data_dir = root_dir.parent / 'data'
atlas_paths = [
    data_dir / 'Keliris_vasc_atlas' / 'VascularProbabilisticAtlas.nii',  # reference frame, so load it first
    data_dir / 'Allen_CCF' / 'average_template_50.nii.gz',
]
for atlas_path in atlas_paths:
    if atlas_path.exists():
        for data, kwargs, _ in read_atlas(atlas_path, ask=True):
            viewer.add_image(data, **kwargs)
    else:
        print(f'Atlas not found, skipping: {atlas_path}  (open it via File -> Open File(s))')

napari.run()

# %%
