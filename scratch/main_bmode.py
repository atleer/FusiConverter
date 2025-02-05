# %%
import os
from pathlib import Path
from warnings import warn
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
name = mdata.dtype.names[0]
value = mdata[name].item().flatten()
if value.size == 1:
    value = value.item()


warn('Tags not yet supported.')
# name = mdata.dtype.names[1]
# mdata[name].item().flatten()

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