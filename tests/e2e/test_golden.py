"""
Rerun the landmark fit and the area registration on one real session and compare with the golden outputs that make_golden.py recorded
"""
import json
import pytest
import numpy as np

from make_golden import GOLDEN_PATH, input_paths, run_pipeline

pytestmark = pytest.mark.data


@pytest.fixture(scope='module')
def golden():
    """The recorded golden outputs"""
    with open(GOLDEN_PATH) as f:
        return json.load(f)

@pytest.fixture(scope='module')
def outputs(data_dir, tmp_path_factory):
    """The pipeline's outputs today, run once for all tests here, with the area map written to a temporary folder"""
    missing = [str(path) for path in input_paths(data_dir) if not path.exists()]
    if missing:
        pytest.skip(f'missing input files: {missing}')
    return run_pipeline(data_dir, tmp_path_factory.mktemp('e2e'))


def test_landmark_fit_matches_golden(outputs, golden):
    """The landmark fit gives the same landmarks, transform matrix and residuals"""
    assert outputs['landmark_fit']['landmark_names'] == golden['landmark_fit']['landmark_names']
    assert np.allclose(outputs['landmark_fit']['transform_matrix'], golden['landmark_fit']['transform_matrix'])
    assert np.allclose(outputs['landmark_fit']['residuals_mm'], golden['landmark_fit']['residuals_mm'])

def test_labels_match_golden(outputs, golden):
    """Every recording voxel gets the same structure label"""
    assert outputs['labels_sha256'] == golden['labels_sha256']

def test_area_map_matches_golden(outputs, golden):
    """The saved area map lists the same structures, voxel counts and fraction outside the atlas"""
    assert outputs['area_map'] == golden['area_map']
