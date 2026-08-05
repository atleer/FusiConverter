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
from src.widgets import CropWidget


h5_filepaths = select_files_from_gui("Select FuSI H5 Files to View")
if not h5_filepaths:
    print("No Files Selected.  Quitting...")
    sys.exit()

# Load images
images = {}
for h5_filepath in tqdm(h5_filepaths, desc='Loading Images...'):
    with h5py.File(h5_filepath) as data:
        filename = Path(h5_filepath).stem
        image_type = data.attrs['imageType']
        images[f'{image_type}_{filename}' ] = np.abs(data['image'])

# View in Napari
print("Launching Napari Image Viewer...")
viewer = napari.Viewer()
for name, image in images.items():
    viewer.add_image(name=name, data=image)


crop_widget = CropWidget(viewer)
viewer.window.add_dock_widget(crop_widget, area='right')


napari.run()
