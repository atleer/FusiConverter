import pytest
import numpy as np
import pandas as pd
import nibabel as nib
from pathlib import Path
from scipy import io
from scipy.spatial.transform import Rotation

# the data folder sits next to the FusiConverter folder
DATA_DIR = Path(__file__).resolve().parents[2] / 'data'


def pytest_collection_modifyitems(items):
    """Skip tests marked 'data' when ../data isn't there (e.g. a fresh clone), rather than failing them"""
    if DATA_DIR.exists():
        return
    skip_data = pytest.mark.skip(reason=f'data folder not found at {DATA_DIR}')
    for item in items:
        if item.get_closest_marker('data'):
            item.add_marker(skip_data)


@pytest.fixture
def random_similarity_transform():
    """Builds a known 4x4 transform (random rotation, uniform 'scale', random translation), the same for the same 'seed'"""
    def make(seed, scale=1.0):
        rng = np.random.default_rng(seed)
        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] = scale * Rotation.random(rng=rng).as_matrix()
        transform_matrix[:3, 3] = rng.uniform(-5, 5, size=3)
        return transform_matrix
    return make

@pytest.fixture
def make_mat(tmp_path):
    """Writes a synthetic .mat recording into tmp_path, laid out like the FUSi exports.

    'image' can be complex or have a trailing time axis, 'image_name' is the variable it's stored under
    ('bmode', 'doppler', 'Ihq' or 'I'), and 'metadata' (e.g. {'voxelSize': ..., 'origen': ...}) becomes
    the MATLAB 'metadata' struct. With metadata=None the file has no metadata at all.
    """
    def make(image, image_name='Ihq', metadata=None, filename='recording.mat'):
        contents = {image_name: image}
        if metadata is not None:
            contents['metadata'] = {name: np.asarray(value) for name, value in metadata.items()} # savemat writes a dict as a struct
        path = tmp_path / filename
        io.savemat(path, contents)
        return path
    return make

@pytest.fixture
def make_nifti(tmp_path):
    """Writes a small synthetic NIfTI atlas into tmp_path whose header voxel size is 'zooms'"""
    def make(zooms, shape=(4, 5, 6), filename='atlas.nii.gz'):
        data = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
        path = tmp_path / filename
        img = nib.Nifti1Image(data, np.diag([*zooms, 1]))
        img.header.set_xyzt_units('mm') # the real atlases say "mm" whatever unit they're actually in, and it's definitely wrong in some cases. E.g. the Allen Atlas would be 13 m x 8 m x 11 m if the mm label in the header was correct.
        nib.save(img, path)
        return path
    return make


@pytest.fixture
def make_structure_csvs(tmp_path):
    """Writes a tiny Allen structure graph CSV into tmp_path, plus (unless with_coarse=False) the coarse-structures CSV next to it.

    Only the columns load_structure_graph reads are included. Isocortex and HPF are the coarse divisions.
    """
    def make(with_coarse=True):
        graph = pd.DataFrame({
            'acronym': ['root', 'grey', 'Isocortex', 'VISp', 'HPF', 'CA1'],
            'id': [997, 8, 315, 385, 1089, 382],
            'name': ['root', 'Basic cell groups and regions', 'Isocortex', 'Primary visual area', 'Hippocampal formation', 'Field CA1'],
            'structure_id_path': ['[997]', '[997, 8]', '[997, 8, 315]', '[997, 8, 315, 385]', '[997, 8, 1089]', '[997, 8, 1089, 382]'],
            'rgb_triplet': ['[255, 255, 255]', '[191, 218, 227]', '[112, 255, 113]', '[8, 133, 140]', '[126, 208, 75]', '[126, 208, 75]'],
        })
        graph_path = tmp_path / 'allen_mouse_connectivity_structure_graph.csv'
        graph.to_csv(graph_path, index=False)
        if with_coarse:
            coarse = graph[graph['acronym'].isin(['Isocortex', 'HPF'])]
            coarse.to_csv(tmp_path / 'allen_mouse_connectivity_coarse_structures.csv', index=False)
        return graph_path
    return make

@pytest.fixture
def image():
    """A small, reproducible 3D recording"""
    return np.random.default_rng(0).random((2, 3, 4))
