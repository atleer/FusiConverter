import tkinter as tk
from tkinter import filedialog
from scipy import io
import h5py


def select_file_from_gui(title=None, defaultextension='.mat'):
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    file_path = filedialog.askopenfilename(title=title, defaultextension=defaultextension)  # Open file dialog
    return file_path


def select_saveas_file_from_gui():
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    file_path = filedialog.asksaveasfilename(title="Save As", defaultextension='.h5')  # Open file dialog
    return file_path

# Load File of Interest
input_file_path = select_file_from_gui("Select .MAT Ultrasound Recording File to Load")
print(f'Loaded {input_file_path}.')

# Load the Matlab data
data = io.loadmat(input_file_path)

# Assign the image
output_file_path = select_saveas_file_from_gui()
print(f'Saving to {output_file_path}...')


for image_name in ['bmode', 'doppler', 'I']:
    if image_name in data.keys():
        print(f'Found {image_name} image.')
        break
else:
    raise ValueError("Could not find image data in file.")
    # import pdb
    # pdb.set_trace()

# Load up Metadata
mdata = {}
for name in data['metadata'].dtype.names:
    print(name)
    mdata[name] = data['metadata'][name].item().flatten()

# Fix Spelling
if 'origen' in mdata:
    mdata['origin'] = mdata.pop('origen')


with h5py.File(output_file_path, 'w') as f:
    f.create_dataset(name='image', data=data[image_name])
    f.attrs['acquisition_type'] = image_name

    for name, value in mdata.items():
        if value.size == 1:
            value = value.item()
        f.attrs[name] = value


    


