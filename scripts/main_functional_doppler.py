# %%
import os
from pathlib import Path
from scipy import io
import h5py

os.chdir(Path(__file__).parent.parent)

# %%
data = io.loadmat(r'data\raw\Test_longRec__FUS_113940.mat')
data

# %%
data['I']

# %%
data['metadata']

