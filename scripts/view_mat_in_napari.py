# %%
import sys
from pathlib import Path


import napari
import os

root_dir = Path(__file__).parent.parent
os.chdir(root_dir)
sys.path.insert(0, str(root_dir))

from src.fusiconverter.utils import *
from src.fusiconverter.viewer_ops import *
from src.fusiconverter.widgets import AddLandmark, AlignmentWidget, ManualAlignmentWidget, RegisterToAreasWidget#, CropWidget

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

napari.run()

# %%
