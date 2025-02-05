from pathlib import Path
import tkinter as tk
from tkinter import filedialog
from warnings import warn
from scipy import io
import h5py


def select_file_from_gui(title=None, defaultextension='.mat'):
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    file_path = filedialog.askopenfilename(title=title, defaultextension=defaultextension)  # Open file dialog
    return file_path


def select_saveas_file_from_gui(default_name=''):
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    file_path = filedialog.asksaveasfilename(title="Save As", initialfile=default_name, defaultextension='.h5')  # Open file dialog
    return file_path

# Load File of Interest
input_file_path = select_file_from_gui("Select .MAT Ultrasound Recording File to Load")
print(f'Loaded {input_file_path}.')

# Load the Matlab data
data = io.loadmat(input_file_path)

# Assign the image
output_file_path = select_saveas_file_from_gui(default_name=Path(input_file_path).stem)
print(f'Saving to {output_file_path}...')


# Get Image Data, so it's put in a consistent variable name.
for image_name in ['bmode', 'doppler', 'I']:
    if image_name in data.keys():
        print(f'Found {image_name} image.')
        break
else:
    raise ValueError("Could not find image data in file.")

# Load up Metadata
mdata = {}
for name in data['metadata'].dtype.names:
    print(name)
    mdata[name] = data['metadata'][name].item().flatten()

# Fix Spelling
if 'origen' in mdata:
    mdata['origin'] = mdata.pop('origen')


# Exclude Unsuported values:
if 'tag' in mdata:
    mdata.pop('tag')
    warn('tags not yet supported, ask Nick if you need them.')

# Extract Datasets
datasets = {}
datasets['image'] = data[image_name]
for name in ['time', 't0']:
    if name in mdata:
        datasets[name] = mdata.pop(name)




with h5py.File(output_file_path, 'w') as f:
    for name, dataset in datasets.items():
        f.create_dataset(name=name, data=dataset)

    for name, value in mdata.items():
        if value.size == 1:
            value = value.item()
        f.attrs[name] = value



print(f'Success!  Save file to {output_file_path}')


