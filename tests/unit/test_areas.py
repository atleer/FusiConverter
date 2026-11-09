"""
Test labelling a recording's voxels with atlas structures, summarising which structures it covers and saving the result
"""
import json
import numpy as np

from fusiconverter.areas import annotate_volume, summarize_areas, save_area_map, load_structure_graph


# %% annotate_volume

def test_annotate_identity_copies_annotation():
    """A recording on the same grid as the atlas, with no transform, gets exactly the atlas labels"""
    annotation = np.arange(1, 61).reshape(3, 4, 5)
    labels, fraction_outside = annotate_volume(annotation.shape, (0.05, 0.05, 0.05), np.eye(4), annotation, (0.05, 0.05, 0.05))
    assert np.array_equal(labels, annotation)
    assert fraction_outside == 0.0

def test_annotate_coarser_recording_samples_every_other_atlas_voxel():
    """A recording with voxels twice the atlas voxel size picks up every other atlas voxel"""
    annotation = np.arange(1, 6 * 8 * 10 + 1).reshape(6, 8, 10)
    labels, fraction_outside = annotate_volume((3, 4, 5), (0.1, 0.1, 0.1), np.eye(4), annotation, (0.05, 0.05, 0.05))
    assert np.array_equal(labels, annotation[::2, ::2, ::2])
    assert fraction_outside == 0.0

def test_annotate_voxels_outside_atlas_are_zero():
    """Recording voxels that land outside the atlas get label 0 and count towards the fraction outside"""
    annotation = np.arange(1, 41).reshape(2, 4, 5)
    labels, fraction_outside = annotate_volume((4, 4, 5), (0.05, 0.05, 0.05), np.eye(4), annotation, (0.05, 0.05, 0.05))
    assert np.array_equal(labels[:2], annotation)
    assert np.all(labels[2:] == 0)
    assert fraction_outside == 0.5

def test_annotate_translation_shifts_lookup():
    """Translating the recording by one atlas voxel looks up the labels one voxel further along"""
    annotation = np.arange(1, 61).reshape(3, 4, 5)
    transform_matrix = np.eye(4)
    transform_matrix[0, 3] = 0.05 # one atlas voxel along axis 0
    labels, fraction_outside = annotate_volume((2, 4, 5), (0.05, 0.05, 0.05), transform_matrix, annotation, (0.05, 0.05, 0.05))
    assert np.array_equal(labels, annotation[1:])
    assert fraction_outside == 0.0

# %% summarize_areas

STRUCTURES = {
    5: {'acronym': 'CA1', 'name': 'Field CA1', 'division': 'HPF'},
    7: {'acronym': 'VISp', 'name': 'Primary visual area', 'division': 'Isocortex'},
}
LABELS = np.array([[[0, 5, 5], [7, 7, 7], [9, 5, 7]]])

def test_summarize_areas():
    """Structures are listed most voxels first with their names, counts, fractions and flat voxel indices, and background (0) is left out"""
    summary = summarize_areas(LABELS, STRUCTURES)

    assert [structure['id'] for structure in summary] == [7, 5, 9] # most voxels first, background (0) dropped
    assert [structure['n_voxels'] for structure in summary] == [4, 3, 1]
    assert summary[0]['acronym'] == 'VISp'
    assert summary[0]['division'] == 'Isocortex'
    assert np.isclose(summary[0]['fraction_of_recording'], 4 / 9)
    assert np.array_equal(summary[0]['voxel_indices'], [3, 4, 5, 8])

def test_summarize_areas_unknown_id():
    """A label that isn't in the structure graph is still listed, as 'unknown_<id>'"""
    summary = summarize_areas(np.array([[[9]]]), {})
    assert summary[0]['acronym'] == 'unknown_9'
    assert summary[0]['division'] == ''

def test_summarize_areas_all_background():
    """A recording with only background (0) gives an empty summary"""
    assert summarize_areas(np.zeros((2, 2, 2), dtype=np.uint32), {}) == []

# %% load_structure_graph

def test_load_structure_graph(make_structure_csvs):
    """Each structure id maps to its acronym, name, colour and the coarse division it sits in"""
    structures = load_structure_graph(make_structure_csvs())
    assert structures[385] == {'acronym': 'VISp', 'name': 'Primary visual area', 'division': 'Isocortex', 'rgb': (8, 133, 140)}
    assert structures[382]['division'] == 'HPF'
    assert structures[315]['division'] == 'Isocortex' # a coarse division is its own division
    assert structures[997]['division'] == '' # structures above the coarse divisions have none

def test_load_structure_graph_without_coarse_csv(make_structure_csvs):
    """Without the coarse-structures CSV next to the structure graph, the structures still load but have no division"""
    structures = load_structure_graph(make_structure_csvs(with_coarse=False))
    assert structures[385]['acronym'] == 'VISp'
    assert all(structure['division'] == '' for structure in structures.values())

# %% save_area_map

def test_save_area_map(tmp_path):
    """The JSON holds the summary without voxel indices, the NPZ holds the voxel indices per structure id, and the caller's summary is left as it was"""
    source_path = tmp_path / 'recording.mat'
    summary = summarize_areas(LABELS, STRUCTURES)

    json_path = save_area_map(source_path, summary, LABELS, 'annotation_50.nii.gz', 1 / 9) # the one 0 voxel in LABELS lies outside the atlas

    assert json_path == tmp_path / 'recording.mat.areas.json'
    with open(json_path) as f:
        area_map = json.load(f)
    assert area_map['source_path'] == str(source_path)
    assert area_map['annotation_source_path'] == 'annotation_50.nii.gz'
    assert area_map['n_voxels_total'] == 9
    assert area_map['n_voxels_unlabelled'] == 1
    assert area_map['fraction_outside_atlas'] == 1 / 9
    assert area_map['hq_shape'] == [1, 3, 3]
    assert area_map['voxel_indices_path'] == str(tmp_path / 'recording.mat.areas.npz')
    assert [structure['id'] for structure in area_map['structures']] == [7, 5, 9]
    assert all('voxel_indices' not in structure for structure in area_map['structures'])
    assert 'created' in area_map

    voxel_indices = np.load(tmp_path / 'recording.mat.areas.npz')
    assert np.array_equal(voxel_indices['labels_shape'], [1, 3, 3])
    for structure in summary:
        assert np.array_equal(voxel_indices[str(structure['id'])], structure['voxel_indices'])

    assert all('voxel_indices' in structure for structure in summary) # caller's dicts untouched

def test_save_area_map_all_background(tmp_path):
    """A recording with no structures gets a JSON with no voxel-indices path, and no NPZ is written"""
    source_path = tmp_path / 'recording.mat'
    labels = np.zeros((2, 2, 2), dtype=np.uint32)

    json_path = save_area_map(source_path, [], labels, 'annotation_50.nii.gz', 1.0)

    with open(json_path) as f:
        area_map = json.load(f)
    assert area_map['voxel_indices_path'] is None
    assert area_map['structures'] == []
    assert not (tmp_path / 'recording.mat.areas.npz').exists()
