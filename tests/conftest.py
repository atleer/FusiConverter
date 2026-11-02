import pytest
import numpy as np
from pathlib import Path
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
