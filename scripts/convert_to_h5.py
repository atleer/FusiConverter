# %%
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog
from warnings import warn

import h5py
from scipy import io
from tqdm import tqdm

from utils import select_files_from_gui, select_save_dir_from_gui
# %%



def main_pipeline(filepath_in, filepath_out):
    # Load the Matlab data
    # %%
    # filepath_in = r"C:\Users\delgr\Downloads\Rat_coronal_Stim_10-20.mat"
    data = io.loadmat(filepath_in)
    data

    # %%

    # Get Image Data, so it's put in a consistent variable name.
    for image_name in ['bmode', 'doppler', 'I']:
        if image_name in data.keys():
            break
    else:
        raise ValueError("Could not find image data in file.")

    # Load up Metadata
    mdata = {}
    if 'metadata' in data:
        for name in data['metadata'].dtype.names:
            mdata[name] = data['metadata'][name].item().flatten()
            

    # Fix Spelling
    if 'origen' in mdata:
        mdata['origin'] = mdata.pop('origen')


    # Exclude Unsuported values:
    if 'tag' in mdata:
        mdata.pop('tag')
        warn('tags not yet supported.')

    # Extract Datasets
    datasets = {}
    datasets['image'] = data[image_name]
    for name in ['time', 't0', 'time0']:
        if name in mdata:
            datasets[name] = mdata.pop(name)


    with h5py.File(filepath_out, 'w') as f:
        for name, dataset in datasets.items():
            f.create_dataset(name=name, data=dataset)

        for name, value in mdata.items():
            if value.size == 1:
                value = value.item()
            f.attrs[name] = value

        if not mdata and image_name == 'I':
            f.attrs['imageType'] = 'I'
            



if __name__ == '__main__':

    # Load File of Interest
    input_file_paths = select_files_from_gui("Select .MAT Ultrasound Recording File to Load")
    if not input_file_paths:
        print("No file selected.  Exiting...")
        sys.exit()


    output_file_dir = select_save_dir_from_gui()
    output_file_paths = [Path(output_file_dir) / Path(inpath).with_suffix('.h5').name for inpath in input_file_paths]

    for input_file, output_file, in tqdm(zip(input_file_paths, output_file_paths)):
        main_pipeline(filepath_in=input_file, filepath_out=output_file)




