import numpy as np
import tkinter as tk
from tkinter import filedialog
import nibabel as nib
from scipy import io
import h5py



def load_atlas_image(atlas_path) -> tuple[np.ndarray, tuple[float, float, float]]:
    img = nib.load(atlas_path)
    zooms = img.header.get_zooms()[:3]
    # the headers all say "mm", but not all of them mean it, so tell them apart by size
    if zooms[0] >= 1:
        # Allen CCF: written in um (e.g. 50.0 for the 50um atlas)
        to_mm = 1 / 1000
    elif zooms[0] >= 0.2:
        # rodent MRI blown up 10x so human-brain tools accept it (e.g. 0.78125 for the 78um Keliris vascular atlas)
        to_mm = 1 / 10
    else:
        # already real mm (e.g. 0.078125 in the Keliris VascularProbabilisticAtlas)
        to_mm = 1
    scale_mm = tuple(z * to_mm for z in zooms)
    return np.asarray(img.get_fdata()), scale_mm

def prepare_image_for_layer(image, voxel_size):
    """Put a loaded acquisition on the axis order napari needs, and say which axes are spatial.

    napari lines layers up by their trailing (last) axes, so a time axis has to go first for the
    spatial axes to line up with the 3D atlas.

    Returns (image, scale, spatial_axes), where spatial_axes are the layer axes holding (Z, X, Y).
    """
    if voxel_size is None:
        # nothing to go on, so fall back on the same "spatial axes are the last three" convention
        return image, (1.0,) * image.ndim, tuple(range(max(0, image.ndim - 3), image.ndim))

    n_spatial = len(voxel_size)
    if image.ndim == n_spatial + 1:
        # there are 4 dimensions, i.e. a time dimension, and the time dimension is moved from the last to the first axis
        image = np.moveaxis(image, -1, 0)
        scale = (1.0,) + tuple(voxel_size)
    elif image.ndim == n_spatial:
        scale = tuple(voxel_size)
    else:
        raise ValueError(
            f"Image has {image.ndim} dimensions but voxel size has {n_spatial}; "
            "don't know which axes are spatial."
        )
    return image, scale, tuple(range(image.ndim - n_spatial, image.ndim))

def load_mat_image(mat_path):
    """Read image, voxel size (mm) and origin (mm) straight from a .mat acquisition file."""

    if '2026' not in str(mat_path):
        data = io.loadmat(mat_path)       
        for image_name in ['bmode', 'doppler', 'Ihq', 'I']:
            if image_name in data:
                break
        else:
            raise ValueError(f"Could not find image data in {mat_path}.")

        metadata = {}
        if 'metadata' in data:
            for name in data['metadata'].dtype.names:
                metadata[name] = np.asarray(data['metadata'][name].item()).flatten()

        if 'origen' in metadata:
            metadata['origin'] = metadata.pop('origen')

        voxel_size = tuple(float(v) for v in metadata['voxelSize']) if 'voxelSize' in metadata else None
        origin = tuple(float(v) for v in metadata['origin']) if 'origin' in metadata else None # TODO: Can I just drop origin? What is it needed for?
        return np.asarray(data[image_name]), voxel_size, origin, image_name
    else:
        with h5py.File(mat_path, 'r') as file:
            image_name = 'I'
            image = file[image_name][()]
            voxel_size = tuple(float(v) for v in file['metadata/voxelSize'][()].flatten())
        return image, voxel_size, None, image_name
        
def prompt_load_annotation() -> str | None:
    """Ask the user for the Allen CCF annotation volume file that says which structure each atlas voxel is in."""
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    root.attributes('-topmost', True)   # force child dialogs to the front
    annotation_path = filedialog.askopenfilename(
        title="Select Allen CCF Annotation File (pick the resolution you aligned to, e.g. annotation_50.nii.gz)",
        filetypes=[("NIfTI files", "*.nii.gz *.nii"), ("All files", "*.*")],
    )
    root.destroy()
    return annotation_path or None
