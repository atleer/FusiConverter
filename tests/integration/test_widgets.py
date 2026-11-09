"""
Test the four dock widgets in a real (hidden) napari viewer, on small synthetic recordings and atlases.
Dialogs are replaced by fixed answers, so nothing waits for a click
"""
import json
import pytest
import numpy as np

import fusiconverter.widgets
from fusiconverter.widgets import AddLandmark, AlignmentWidget, ManualAlignmentWidget, RegisterToAreasWidget
from fusiconverter.reader import read_mat, read_atlas
from fusiconverter.viewer_ops import get_transform_matrix_from_layer_affine

pytestmark = pytest.mark.gui

RECORDING_VOXEL_SIZE = (0.1, 0.2, 0.3)
ATLAS_VOXEL_SIZE = (0.078125,) * 3

# landmarks are in voxel indices of the volume they're placed on
RECORDING_LANDMARKS = {'a': [0, 0, 0], 'b': [1, 0, 0], 'c': [0, 2, 0], 'd': [0, 0, 3]}


def open_recording(viewer, make_mat, image):
    """Open a synthetic recording the way File -> Open does (without the alignment prompt) and return its layer"""
    [(data, kwargs, _)] = read_mat(make_mat(image, metadata={'voxelSize': RECORDING_VOXEL_SIZE}), ask=False) # make_mat (from conftest) creates the synthetic matrix, read_mat is an imported reader function
    return viewer.add_image(data, **kwargs)

def open_atlas(viewer, make_nifti):
    """Open a synthetic vascular reference atlas and return its layer"""
    [(data, kwargs, _)] = read_atlas(make_nifti(ATLAS_VOXEL_SIZE, filename='VascularProbabilisticAtlas.nii'), ask=False)
    return viewer.add_image(data, **kwargs)

def dock(viewer, widget_class):
    """Create a widget and dock it in the viewer, as the launch script does"""
    widget = widget_class(viewer)
    viewer.window.add_dock_widget(widget)
    return widget

def write_landmarks(source_path, landmarks):
    """Write a landmarks file next to 'source_path', as AddLandmark.save would"""
    with open(f'{source_path}.landmarks.json', 'w') as f:
        json.dump(landmarks, f)

def read_json(path):
    """Contents of a JSON file"""
    with open(path) as f:
        return json.load(f)

def atlas_landmarks(transform_matrix):
    """Where RECORDING_LANDMARKS land in atlas voxel indices under 'transform_matrix' (recording mm -> atlas mm)"""
    landmarks = {}
    for name, voxel in RECORDING_LANDMARKS.items():
        mm = np.append(np.multiply(voxel, RECORDING_VOXEL_SIZE), 1)
        landmarks[name] = ((transform_matrix @ mm)[:3] / ATLAS_VOXEL_SIZE).tolist()
    return landmarks

def place_landmarks(viewer, widget, landmarks, time_index=None):
    """Place 'landmarks' with the AddLandmark widget, one click each, on the layer picked in the widget"""
    for name, voxel in landmarks.items():
        widget.name_input.setText(name)
        widget.add_landmark()
        coords = voxel if time_index is None else [time_index, *voxel]
        viewer.layers[f'{widget.layer_landmarks_added.currentText()}_landmarks'].add(coords)

@pytest.fixture
def recording_and_atlas(make_napari_viewer, make_mat, make_nifti, image):
    """A viewer with a 3D recording and the vascular atlas open"""
    viewer = make_napari_viewer()
    return viewer, open_recording(viewer, make_mat, image), open_atlas(viewer, make_nifti)

def landmark_alignment(viewer, recording, atlas, transform_matrix):
    """An AlignmentWidget that has just fitted 'transform_matrix' from landmark files on the recording and atlas"""
    write_landmarks(recording.metadata['source_path'], RECORDING_LANDMARKS)
    write_landmarks(atlas.metadata['source_path'], atlas_landmarks(transform_matrix))
    widget = dock(viewer, AlignmentWidget)
    widget.moving_layer.setCurrentText(recording.name)
    widget.fixed_layer.setCurrentText(atlas.name)
    widget.align()
    return widget

# %% plugin wiring

def test_viewer_opens_mat_with_plugin(make_napari_viewer, make_mat, image):
    """napari.yaml registers the reader, so napari can open a .mat recording through the fusiconverter plugin"""
    viewer = make_napari_viewer()
    [layer] = viewer.open(make_mat(image, metadata={'voxelSize': RECORDING_VOXEL_SIZE}), plugin='fusiconverter')
    assert layer.name == 'Ihq_recording'
    assert np.allclose(layer.scale, RECORDING_VOXEL_SIZE)

# %% AddLandmark

def test_add_landmark_creates_points_layer(make_napari_viewer, make_mat, image):
    """Adding a landmark creates a points layer for the picked image, in add mode, that names the next point after the landmark"""
    viewer = make_napari_viewer()
    recording = open_recording(viewer, make_mat, image)
    widget = dock(viewer, AddLandmark)
    widget.name_input.setText('a')
    widget.add_landmark()

    points_layer = viewer.layers[f'{recording.name}_landmarks']
    assert points_layer.mode == 'add'
    points_layer.add([0, 1, 2])
    assert list(points_layer.features['landmark_name']) == ['a']
    assert np.allclose(points_layer.data, np.array([[0., 1., 2.]]))

def test_save_landmarks_new_file(make_napari_viewer, make_mat, image):
    """With no landmarks file yet, saving writes the placed landmarks without asking"""
    viewer = make_napari_viewer()
    recording = open_recording(viewer, make_mat, image)
    widget = dock(viewer, AddLandmark)
    place_landmarks(viewer, widget, {'a': [0, 1, 2]})
    widget.save()
    assert read_json(f"{recording.metadata['source_path']}.landmarks.json") == {'a': [0, 1, 2]}

@pytest.mark.parametrize('mode, expected', [
    ('append', {'old': [1, 1, 1], 'a': [0, 1, 2]}),
    ('overwrite', {'a': [0, 1, 2]}),
    (None, {'old': [1, 1, 1]}),
], ids=['append', 'overwrite', 'cancel'])
def test_save_landmarks_existing_file(make_napari_viewer, make_mat, image, monkeypatch, mode, expected):
    """With a landmarks file already there, append adds to it, overwrite replaces it and cancel leaves it as it was"""
    viewer = make_napari_viewer()
    recording = open_recording(viewer, make_mat, image)
    source_path = recording.metadata['source_path']
    widget = dock(viewer, AddLandmark)
    place_landmarks(viewer, widget, {'a': [0, 1, 2]}) # placed before the file exists, so the layer holds only 'a'
    write_landmarks(source_path, {'old': [1, 1, 1]})
    monkeypatch.setattr(widget, '_ask_save_mode', lambda json_path: mode) # the _ask_save_mode is a function in AddLandmark, setattr sets the widgets attribute the way the _ask_save_mode function would have using the mode argument
    widget.save()
    assert read_json(f'{source_path}.landmarks.json') == expected

def test_load_landmarks(make_napari_viewer, make_mat, image):
    """Loading shows the saved landmarks as a points layer on the picked image"""
    viewer = make_napari_viewer()
    recording = open_recording(viewer, make_mat, image)
    write_landmarks(recording.metadata['source_path'], RECORDING_LANDMARKS)
    widget = dock(viewer, AddLandmark)
    widget.load()

    points_layer = viewer.layers[f'{recording.name}_landmarks']
    assert list(points_layer.features['landmark_name']) == list(RECORDING_LANDMARKS)
    assert np.allclose(points_layer.data, list(RECORDING_LANDMARKS.values()))

# %% AlignmentWidget

def test_align_recovers_known_transform(recording_and_atlas, random_similarity_transform):
    """Landmarks on the recording and atlas give back the transform they were made with, which is put in the recording's affine"""
    viewer, recording, atlas = recording_and_atlas
    known = random_similarity_transform(seed=0)
    landmark_alignment(viewer, recording, atlas, known)
    assert np.allclose(recording.affine.affine_matrix, known)

def test_align_needs_three_landmarks(recording_and_atlas):
    """With fewer than 3 landmark names on both layers, the user is told that a minimum of 3 matching landmarks the recording isn't moved"""
    viewer, recording, atlas = recording_and_atlas
    write_landmarks(recording.metadata['source_path'], {'a': [0, 0, 0], 'b': [1, 0, 0]})
    write_landmarks(atlas.metadata['source_path'], {'a': [0, 0, 0], 'b': [1, 0, 0]})
    widget = dock(viewer, AlignmentWidget)
    widget.moving_layer.setCurrentText(recording.name)
    widget.fixed_layer.setCurrentText(atlas.name)
    widget.align()
    assert widget.status_label.text() == 'Need >=3 matching landmark names on both layers, found 2.'
    assert np.allclose(recording.affine.affine_matrix, np.eye(4))

def test_save_alignment_new_file(recording_and_atlas, random_similarity_transform):
    """With no alignment file yet, saving writes the fitted transform and the landmarks it came from without asking"""
    viewer, recording, atlas = recording_and_atlas
    known = random_similarity_transform(seed=0)
    widget = landmark_alignment(viewer, recording, atlas, known)
    widget.save()

    alignment = read_json(f"{recording.metadata['source_path']}.alignment.json")
    assert np.allclose(alignment['transform_matrix'], known)
    assert alignment['method'] == 'landmarks'
    assert alignment['landmark_names'] == sorted(RECORDING_LANDMARKS)
    assert alignment['atlas_source_path'] == atlas.metadata['source_path']

@pytest.mark.parametrize('mode, overwritten', [('overwrite', True), (None, False)], ids=['overwrite', 'cancel'])
def test_save_alignment_existing_file(recording_and_atlas, random_similarity_transform, monkeypatch, mode, overwritten):
    """With an alignment file already there, overwrite replaces it and cancel leaves it as it was"""
    viewer, recording, atlas = recording_and_atlas
    json_path = f"{recording.metadata['source_path']}.alignment.json"
    with open(json_path, 'w+') as f:
        f.write('old')
    widget = landmark_alignment(viewer, recording, atlas, random_similarity_transform(seed=0))
    monkeypatch.setattr(widget, '_ask_save_mode', lambda json_path: mode)  # the _ask_save_mode is a function in AlignmentWidget, setattr sets the widgets attribute the way the _ask_save_mode function would have using the mode argument
    widget.save()
    with open(json_path) as f:
        assert (f.read() != 'old') == overwritten # the f file no longer contains 'old' because it has been overwritten

@pytest.mark.xfail(strict=True, reason="align takes the first three entries of scale and of each landmark, which on a 3D + time layer include time")
def test_align_4d_recording(make_napari_viewer, make_mat, make_nifti, random_similarity_transform):
    """Landmarks placed on a 3D + time recording give back the known transform, put in the spatial part of its affine"""
    viewer = make_napari_viewer()
    recording = open_recording(viewer, make_mat, np.ones((2, 3, 4, 5)))
    atlas = open_atlas(viewer, make_nifti)
    known = random_similarity_transform(seed=0)
    write_landmarks(atlas.metadata['source_path'], atlas_landmarks(known))

    add_landmark = dock(viewer, AddLandmark)
    add_landmark.layer_landmarks_added.setCurrentText(recording.name)
    place_landmarks(viewer, add_landmark, RECORDING_LANDMARKS, time_index=0)
    widget = dock(viewer, AlignmentWidget)
    widget.moving_layer.setCurrentText(recording.name)
    widget.fixed_layer.setCurrentText(atlas.name)
    widget.align()

    assert np.allclose(get_transform_matrix_from_layer_affine(recording.affine.affine_matrix, spatial_axes=[1, 2, 3]), known)

# %% ManualAlignmentWidget

def manual_alignment(viewer, recording):
    """A ManualAlignmentWidget with 'recording' picked"""
    widget = dock(viewer, ManualAlignmentWidget)
    widget.layer_choice.setCurrentText(recording.name)
    return widget

def translate_z(widget, mm):
    """Set the Z translate spinbox to 'mm'"""
    boxes, _ = widget.controls['Translate (mm)']
    boxes[0].setValue(mm)

def test_manual_translate_and_reset(recording_and_atlas):
    """The Z translate spinbox shifts the recording by that many mm along Z, and reset puts it back"""
    viewer, recording, _ = recording_and_atlas
    widget = manual_alignment(viewer, recording)
    translate_z(widget, 1.5)
    expected = np.eye(4)
    expected[0, 3] = 1.5
    assert np.allclose(recording.affine.affine_matrix, expected)

    widget.reset()
    assert np.allclose(recording.affine.affine_matrix, np.eye(4))

def test_manual_save(recording_and_atlas):
    """Saving writes the adjusted transform, marked as a manual alignment"""
    viewer, recording, _ = recording_and_atlas
    widget = manual_alignment(viewer, recording)
    translate_z(widget, 1.5)
    widget.save()

    alignment = read_json(f"{recording.metadata['source_path']}.alignment.json")
    assert alignment['method'] == 'manual'
    assert np.allclose(alignment['transform_matrix'], recording.affine.affine_matrix)

@pytest.mark.xfail(strict=True, reason="save looks for a layer named 'Atlas', which no atlas layer is called")
def test_manual_save_records_atlas(recording_and_atlas):
    """Saving records the open vascular atlas as the atlas the recording was aligned to"""
    viewer, recording, atlas = recording_and_atlas
    widget = manual_alignment(viewer, recording)
    translate_z(widget, 1.5)
    widget.save()

    alignment = read_json(f"{recording.metadata['source_path']}.alignment.json")
    assert alignment['atlas_source_path'] == atlas.metadata['source_path']
    assert np.allclose(alignment['atlas_voxel_size_in_mm'], ATLAS_VOXEL_SIZE)

@pytest.mark.xfail(strict=True, reason="save after a landmark alignment writes method 'manual' and drops the landmark fit")
def test_manual_save_after_landmark_alignment_keeps_landmark_fit(recording_and_atlas, random_similarity_transform, monkeypatch):
    """Adjusting a landmark alignment by hand and saving keeps the landmark fit as it was fitted, plus the manual adjustment on top of it"""
    viewer, recording, atlas = recording_and_atlas
    known = random_similarity_transform(seed=0)
    landmark_alignment(viewer, recording, atlas, known).save()
    monkeypatch.setattr(fusiconverter.widgets, '_ask_overwrite_alignment', lambda parent, json_path: 'overwrite')
    widget = manual_alignment(viewer, recording)
    translate_z(widget, 1.5)
    widget.save()

    alignment = read_json(f"{recording.metadata['source_path']}.alignment.json")
    assert alignment['method'] == 'landmarks+manual'
    landmark_fit = alignment['landmark_fit']
    assert np.allclose(landmark_fit['transform_matrix'], known)
    assert landmark_fit['landmark_names'] == sorted(RECORDING_LANDMARKS)
    assert set(landmark_fit['residuals_mm']) == set(RECORDING_LANDMARKS)
    assert landmark_fit['rms_error'] == pytest.approx(0, abs=1e-9)
    assert np.allclose(np.array(alignment['manual_adjustment']) @ known, alignment['transform_matrix'])

# %% RegisterToAreasWidget

SHIFT = np.array([
    [1, 0, 0, 1],
    [0, 1, 0, 2],
    [0, 0, 1, 3],
    [0, 0, 0, 1],
], dtype=float)

@pytest.fixture
def registration(make_napari_viewer, make_mat, image, tmp_path, monkeypatch):
    """A viewer with a recording, a register widget, and a synthetic annotation (one label per voxel, same grid as the recording)
    and structure graph in place of the real Allen files"""
    viewer = make_napari_viewer()
    recording = open_recording(viewer, make_mat, image)
    annotation = np.arange(1, image.size + 1).reshape(image.shape)
    structures = {1: {'acronym': 'AAA', 'name': 'Area A', 'division': 'Div', 'rgb': (255, 0, 0)}}
    monkeypatch.setattr(fusiconverter.widgets, 'load_atlas_image', lambda path: (annotation, RECORDING_VOXEL_SIZE))
    monkeypatch.setattr(fusiconverter.widgets, 'load_structure_graph', lambda path: structures)
    monkeypatch.setattr(fusiconverter.widgets, 'prompt_load_annotation', lambda: str(tmp_path / 'annotation.nii.gz')) # only asked if ../data has no annotation
    widget = dock(viewer, RegisterToAreasWidget)
    widget.image_layer.setCurrentText(recording.name)
    return viewer, recording, widget, annotation

def add_allen_atlas(viewer, tmp_path, affine):
    """Add a layer that stands in for the Allen atlas: its source is in an Allen_CCF folder and 'affine' moves it onto the vascular atlas"""
    viewer.add_image(np.zeros((2, 2, 2)), name='average_template_50', affine=affine,
                     metadata={'source_path': str(tmp_path / 'Allen_CCF' / 'average_template_50.nii.gz')})

def test_register_adds_labels_and_saves(registration, tmp_path):
    """An aligned recording, with an aligned Allen atlas open, gets a Labels layer with the structure each voxel sits in, and its area map is saved"""
    viewer, recording, widget, annotation = registration
    recording.affine = SHIFT
    add_allen_atlas(viewer, tmp_path, SHIFT) # moved the same way as the recording, so recording voxels sit on the same annotation voxels
    widget.register()

    labels_layer = viewer.layers[f'{recording.name}_areas']
    assert np.array_equal(labels_layer.metadata['structure_ids'][labels_layer.data], annotation)
    area_map = read_json(f"{recording.metadata['source_path']}.areas.json")
    assert len(area_map['structures']) == annotation.size

def test_register_needs_aligned_recording(registration, tmp_path):
    """A recording that hasn't been aligned is not registered, and the user is told to align it first"""
    viewer, recording, widget, _ = registration
    add_allen_atlas(viewer, tmp_path, SHIFT)
    widget.register()
    assert 'has not been aligned to the atlas yet' in widget.status_label.text()
    assert f'{recording.name}_areas' not in viewer.layers

def test_register_needs_aligned_allen_atlas(registration):
    """Without an aligned Allen atlas open, the recording is not registered, and the user is told to open one"""
    viewer, recording, widget, _ = registration
    recording.affine = SHIFT
    widget.register()
    assert widget.status_label.text().startswith('Open the Allen atlas')
    assert f'{recording.name}_areas' not in viewer.layers
