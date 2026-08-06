import sys
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm
import os

root_dir = Path(__file__).parent.parent
os.chdir(root_dir)
sys.path.insert(0, str(root_dir))

from src.utils import (
    select_files_from_gui,
    select_save_dir_from_gui,
    load_registration,
    load_mat_image,
    load_atlas_image,
    resample_volume_to_atlas,
)


registration_paths = select_files_from_gui(
    "Select a .registration.json File (produced by the Registration Widget in view_h5_in_napari.py)",
    defaultextension='.json',
)
if not registration_paths:
    print("No registration file selected. Quitting...")
    sys.exit()
registration = load_registration(registration_paths[0])

atlas_image, atlas_voxel_size_mm = load_atlas_image(registration['atlas_source'])

mat_filepaths = select_files_from_gui("Select Non-HQ .MAT Files to Register (e.g. T_*.mat)")
if not mat_filepaths:
    print("No files selected. Quitting...")
    sys.exit()

output_dir = select_save_dir_from_gui()
if not output_dir:
    print("No output directory selected. Quitting...")
    sys.exit()

for mat_filepath in mat_filepaths:
    image, mdata = load_mat_image(mat_filepath)
    source_voxel_size_mm = mdata['voxelSize']

    n_timepoints = image.shape[-1]
    registered_timepoints = [
        resample_volume_to_atlas(
            image[..., t],
            source_voxel_size_mm=source_voxel_size_mm,
            matrix_mm=registration['matrix_mm'],
            atlas_shape=atlas_image.shape,
            atlas_voxel_size_mm=atlas_voxel_size_mm,
        )
        for t in tqdm(range(n_timepoints), desc=f'Registering {Path(mat_filepath).name}...')
    ]
    registered_image = np.stack(registered_timepoints, axis=-1).astype(np.float32)

    output_path = Path(output_dir) / f'{Path(mat_filepath).stem}_registered.h5'
    with h5py.File(output_path, 'w') as f:
        f.create_dataset(name='image', data=registered_image)
        for name in ['time', 't0']:
            if name in mdata:
                f.create_dataset(name=name, data=mdata[name])
        f.create_dataset(name='voxelSize', data=np.array(atlas_voxel_size_mm))

        f.attrs['imageType'] = 'registered'
        f.attrs['registration_model'] = registration['model']
        f.attrs['registration_source'] = str(registration_paths[0])
        f.attrs['hq_source'] = registration['hq_source']
        f.attrs['atlas_source'] = registration['atlas_source']

    print(f"Wrote {output_path}")

print("Done.")
