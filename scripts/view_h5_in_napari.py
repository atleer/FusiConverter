import sys
from pathlib import Path

import h5py
import napari
import numpy as np

from utils import select_files_from_gui


h5_filepaths = select_files_from_gui("Select FuSI H5 Files to View")
if not h5_filepaths:
    print("No Files Selected.  Quitting...")
    sys.exit()



# Load images
images = {}
for h5_filepath in h5_filepaths:
    with h5py.File(h5_filepath) as data:
        filename = Path(h5_filepath).stem
        image = np.abs(data['image'])
        image_type = data.attrs['imageType']
        name = f'{image_type}_{filename}'
        images[name] = image


# View in Napari
print("Launching Napari Image Viewer...")
viewer = napari.Viewer()
for name, image in images.items():
    viewer.add_image(name=name, data=image)


napari.run()
