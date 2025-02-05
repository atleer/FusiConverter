# %%
import os
from pathlib import Path
from scipy import io
import h5py

os.chdir(Path(__file__).parent.parent)

# %%
data = io.loadmat(r'data\raw\Bmode__BM_113336.mat')
data

# %%


# %%
list(data.keys())

# %%
data['bmode']

# %%
mdata = data['metadata']
mdata 
# %%
imageDim = mdata['imageDim'][0][0][0][0].item()

# %%

# %%
f = h5py.File("output.h5", "w")
f.create_dataset('bmode', data=data['bmode'])
f.attrs['imageDim'] = mdata['imageDim'][0][0][0][0].item()

f.close()

# %%

import napari


# %%
data['bmode'].real
# %%

viewer = napari.view_image(data['bmode'].real)

# %%
napari.view_image(data=data['bmode'].imag)