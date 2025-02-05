# %%
import os
from pathlib import Path
from scipy import io
import h5py

os.chdir(Path(__file__).parent.parent)

# %%
data = io.loadmat(r'data\raw\Tag1_longRec__FUS_172709.mat')
data

# %%
data.keys()

# %%
data['metadata']['tag'].item().item()[0]

# %%
data['I']

# %%
mdata = data['metadata']
mdata


# %%
mdata.dtype.names


time = mdata.pop('time')

# %%
mdata['time'].item().flatten()

# %%
mdata['t0'].item().flatten()

# %%
'time' in mdata.dtype.names