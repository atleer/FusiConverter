import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
import nibabel as nib


def select_files_from_gui(title=None, defaultextension='.mat') -> tuple[str, ...]:
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    root.attributes('-topmost', True)   # force child dialogs to the front
    file_path = filedialog.askopenfilenames(title=title, defaultextension=defaultextension)  # Open file dialog
    if not file_path:
        return ()
    return file_path


def select_save_dir_from_gui(title="Save Directory") -> str | None:
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    root.attributes('-topmost', True)   # force child dialogs to the front
    file_path = filedialog.askdirectory(title=title)  # Open folder dialog
    if not file_path:
        return None
    return file_path

def prompt_load_atlas() -> str | None:
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    root.attributes('-topmost', True)   # force child dialogs to the front
    load = messagebox.askyesno(
        "Load Atlas",
        "Would you like to load the Allen CCF atlas as an additional layer on top of the recorded image?\n\n"
        "If so, navigate to the folder Allen_CCF after clicking \"yes\" and select the atlas to load. Recommended: 50um resolution (e.g. 'average_template_50.nii.gz') in Allen_CCF."
    )
    if not load:
        root.destroy()
        return None
    atlas_path = filedialog.askopenfilename(
        title="Select Allen CCF Atlas File (recommended: 50um, e.g. average_template_50.nii.gz)",
        filetypes=[("NIfTI files", "*.nii.gz *.nii"), ("All files", "*.*")],
    )
    root.destroy()
    return atlas_path or None


def load_atlas_image(atlas_path) -> tuple[np.ndarray, tuple[float, float, float]]:
    img = nib.load(atlas_path)
    # Allen CCF .nii.gz files declare their units as "mm" but the zoom values
    # are actually written in um (e.g. 50.0 for the 50um atlas), so divide by
    # 1000 to get a real mm voxel size comparable to the h5 files' voxelSize.
    scale_mm = tuple(z / 1000 for z in img.header.get_zooms()[:3])
    return np.asarray(img.get_fdata()), scale_mm


def get_voxel_size_mm(h5_file) -> tuple[float, ...] | None:
    if 'voxelSize' not in h5_file:
        return None
    return tuple(h5_file['voxelSize'][()])