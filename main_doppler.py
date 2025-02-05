# %%
import os
from pathlib import Path
from scipy import io
import h5py

os.chdir(Path(__file__).parent.parent)

# %%
data = io.loadmat(r'data\raw\Doppler__DP_113520.mat')
data

# %%
data['doppler']

# %%
data['metadata']