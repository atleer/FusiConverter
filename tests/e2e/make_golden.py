"""
Record the golden outputs that tests/e2e/test_golden.py compares against, by running the landmark fit and the area registration on one real session.
"Golden" means what the code produces today, not output checked to be correct. Run it once by hand from the repo root,

    uv run python tests/e2e/make_golden.py

and rerun it only when a change in the outputs is intended
"""
import hashlib
import json
import tempfile
from pathlib import Path
import numpy as np

from fusiconverter.reader import read_mat, read_atlas
from fusiconverter.utils import load_atlas_image, load_structure_graph
from fusiconverter.viewer_ops import (
    load_landmarks,
    match_landmarks,
    fit_similarity_transform,
    transform_residuals,
    get_transform_matrix_from_layer_affine,
    annotate_volume,
    summarize_areas,
    save_area_map,
)

GOLDEN_PATH = Path(__file__).parent / 'golden' / 'golden.json'

# relative to the data folder
HQ_PATH = Path('20251211-0', '20251211-0_GEY-P659', '03_OBac+_GEY-P659_fUS_251211_140600_3d_HQ.mat')
VASCULAR_ATLAS_PATH = Path('Keliris_vasc_atlas', 'VascularProbabilisticAtlas.nii')
ALLEN_TEMPLATE_PATH = Path('Allen_CCF', 'average_template_50.nii.gz')
ANNOTATION_PATH = Path('Allen_CCF', 'annotation', 'annotation_50.nii.gz')
STRUCTURE_GRAPH_PATH = ANNOTATION_PATH.parent / 'allen_mouse_connectivity_structure_graph.csv'


def input_paths(data_dir):
    """Every real file the pipeline reads, including the saved landmarks and alignments next to the recording and atlases"""
    return [
        data_dir / HQ_PATH,
        data_dir / f'{HQ_PATH}.landmarks.json',
        data_dir / f'{HQ_PATH}.alignment.json',
        data_dir / VASCULAR_ATLAS_PATH,
        data_dir / f'{VASCULAR_ATLAS_PATH}.landmarks.json',
        data_dir / ALLEN_TEMPLATE_PATH,
        data_dir / f'{ALLEN_TEMPLATE_PATH}.alignment.json',
        data_dir / ANNOTATION_PATH,
        data_dir / STRUCTURE_GRAPH_PATH,
    ]

def run_pipeline(data_dir, out_dir):
    """Run the landmark fit (as AlignmentWidget.align does) and the area registration with the saved alignments
    (as RegisterToAreasWidget.register does) on the real session, and return the outputs to compare.
    The area map is written into 'out_dir', never next to the real data"""
    hq_path = data_dir / HQ_PATH
    # ask=False applies the saved alignments without asking
    [(hq_image, hq_kwargs, _)] = read_mat(hq_path, ask=False)
    [(_, vascular_kwargs, _)] = read_atlas(data_dir / VASCULAR_ATLAS_PATH, ask=False)
    [(_, allen_kwargs, _)] = read_atlas(data_dir / ALLEN_TEMPLATE_PATH, ask=False)

    # landmark fit: HQ landmarks -> vascular atlas landmarks, both in mm
    names, hq_points, atlas_points = match_landmarks(load_landmarks(hq_path), load_landmarks(data_dir / VASCULAR_ATLAS_PATH))
    moving = hq_points * np.array(hq_kwargs['scale'][:3])
    fixed = atlas_points * np.array(vascular_kwargs['scale'][:3])
    fit_matrix = fit_similarity_transform(moving, fixed)
    residuals = transform_residuals(fit_matrix, moving, fixed)

    # area registration: recording mm -> vascular atlas mm -> Allen mm, since the annotation is in Allen space
    spatial_axes = list(hq_kwargs['metadata']['spatial_axes'])
    transform_matrix = get_transform_matrix_from_layer_affine(hq_kwargs['affine'], spatial_axes)
    allen_transform_matrix = get_transform_matrix_from_layer_affine(allen_kwargs['affine'])
    transform_matrix = np.linalg.inv(allen_transform_matrix) @ transform_matrix

    annotation, annotation_voxel_size = load_atlas_image(data_dir / ANNOTATION_PATH)
    structures = load_structure_graph(data_dir / STRUCTURE_GRAPH_PATH)
    voxel_size = np.asarray(hq_kwargs['scale'])[spatial_axes]
    spatial_shape = tuple(np.asarray(hq_image.shape)[spatial_axes].tolist())
    labels, fraction_outside = annotate_volume(spatial_shape, voxel_size, transform_matrix, annotation, annotation_voxel_size)
    summary = summarize_areas(labels, structures)

    json_path = save_area_map(Path(out_dir) / hq_path.name, summary, labels, data_dir / ANNOTATION_PATH, fraction_outside)
    with open(json_path) as f:
        area_map = json.load(f)
    for key in ('source_path', 'annotation_source_path', 'voxel_indices_path', 'created'): # these depend on where and when it was run
        del area_map[key]

    return {
        'landmark_fit': {
            'landmark_names': list(names),
            'transform_matrix': fit_matrix.tolist(),
            'residuals_mm': np.asarray(residuals).tolist(),
        },
        'labels_sha256': hashlib.sha256(labels.tobytes()).hexdigest(),
        'area_map': area_map,
    }


if __name__ == '__main__':
    with tempfile.TemporaryDirectory() as out_dir:
        outputs = run_pipeline(Path(__file__).resolve().parents[3] / 'data', out_dir)
    GOLDEN_PATH.parent.mkdir(exist_ok=True)
    with open(GOLDEN_PATH, 'w') as f:
        json.dump(outputs, f, indent=2)
    print(f'Wrote {GOLDEN_PATH}')
