import numpy as np
import tkinter as tk
from tkinter import filedialog
import napari


def select_files_from_gui(title=None, defaultextension='.mat') -> tuple[str, ...]:
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    file_path = filedialog.askopenfilenames(title=title, defaultextension=defaultextension)  # Open file dialog
    if not file_path:
        return ()
    return file_path


def select_save_dir_from_gui(title="Save Directory") -> str | None:
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    file_path = filedialog.askdirectory(title=title)  # Open folder dialog
    if not file_path:
        return None
    return file_path

# Function to crop image in spatial dimensions (no temporal)
def crop_image(viewer, z_range, x_range, y_range):
    layer = viewer.layers.selection.active
    if layer is None or not isinstance(layer, napari.layers.Image):
        print("No image layer selected.")
        return
    
    # Get the shape of the image
    z_shape, x_shape, y_shape, _ = layer.data.shape

    # Set default ranges if not provided
    if not z_range:
        z_range = f'0:{z_shape}'
    if not x_range:
        x_range = f'0:{x_shape}'
    if not y_range:
        y_range = f'0:{y_shape}'
        
    # Parse the input ranges
    z_min, z_max = map(int, z_range.split(':'))
    x_min, x_max = map(int, x_range.split(':'))
    y_min, y_max = map(int, y_range.split(':'))

    # Crop the image
    cropped_image = layer.data[z_min:z_max, x_min:x_max, y_min:y_max, :]

    # Add the cropped image as a new layer
    viewer.add_image(cropped_image, name=f'Cropped_{layer.name}')

def apply_log_normalization(viewer):
    layer = viewer.layers.selection.active
    if layer is None or not isinstance(layer, napari.layers.Image):
        print("No image layer selected.")
        return
    # Map the image data to [0, 1] range
    data_min = np.min(layer.data)
    data_max = np.max(layer.data)
    scaled_data = (layer.data - data_min) / (data_max - data_min)

    # Apply logarithmic normalization
    normalized_data = np.log1p(scaled_data)
    
    # Add the normalized image as a new layer
    viewer.add_image(normalized_data, name=f'LogNorm_{layer.name}')