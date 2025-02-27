import sys
from pathlib import Path

import h5py
import napari
import numpy as np
from tqdm import tqdm

from utils import select_files_from_gui, crop_image
from qtpy.QtWidgets import QVBoxLayout, QWidget, QLabel, QLineEdit, QPushButton


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

## Crop Widget
class CropWidget(QWidget):
    def __init__(self, viewer):
        super().__init__()
        self.viewer = viewer
        self.setLayout(QVBoxLayout())

        self.layout().addWidget(QLabel('Z Range (e.g., 0:10):'))
        self.z_range_input = QLineEdit()
        self.layout().addWidget(self.z_range_input)

        self.layout().addWidget(QLabel('X Range (e.g., 0:100):'))
        self.x_range_input = QLineEdit()
        self.layout().addWidget(self.x_range_input)

        self.layout().addWidget(QLabel('Y Range (e.g., 0:100):'))
        self.y_range_input = QLineEdit()
        self.layout().addWidget(self.y_range_input)

        crop_button = QPushButton('Crop Image')
        crop_button.clicked.connect(self.crop)
        self.layout().addWidget(crop_button)

    def crop(self):
        z_range = self.z_range_input.text()
        x_range = self.x_range_input.text()
        y_range = self.y_range_input.text()
        crop_image(self.viewer, z_range, x_range, y_range)


crop_widget = CropWidget(viewer)
viewer.window.add_dock_widget(crop_widget, area='right')


napari.run()
