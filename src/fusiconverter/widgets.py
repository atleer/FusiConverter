import numpy as np
import napari
from qtpy.QtWidgets import QVBoxLayout, QWidget, QLabel, QLineEdit, QPushButton, QComboBox, QMessageBox, QCheckBox
from src.fusiconverter.viewer_ops import save_landmarks, load_landmarks, get_or_create_landmarks_layer, match_landmarks, fit_similarity_transform, transform_residuals, to_layer_affine, save_alignment_matrix, load_alignment_matrix, annotate_volume # apply_log_normalization, crop_image
from src.fusiconverter.utils import (
    select_files_from_gui,
    select_save_dir_from_gui,
    load_atlas_image,
    load_image,
    prompt_load_annotation,
    load_structure_graph,
)
import sys
from pathlib import Path

class AddLandmark(QWidget):
    def __init__(self, viewer):
        super().__init__()
        self.viewer = viewer
        self.setLayout(QVBoxLayout())

        self.layout().addWidget(QLabel('Target Image Layer:'))
        self.layer_landmarks_added = QComboBox() # adds a blank dropdown menu
        self.layout().addWidget(self.layer_landmarks_added)
        self._refresh_layer_choices()
        self.viewer.layers.events.inserted.connect(self._refresh_layer_choices) # refresh widget when a new layer is added to napari so that it shows up in dropdown menu
        self.viewer.layers.events.removed.connect(self._refresh_layer_choices) # refresh widget when a new layer is removed from napari so that it shows up in dropdown menu

        self.layout().addWidget(QLabel('Landmark Name:')) # Adds text "Landmark Name" above open field
        self.name_input = QLineEdit() # Adds open field to enter name for landmark
        self.layout().addWidget(self.name_input)

        add_button = QPushButton('Add Landmark')
        add_button.clicked.connect(self.add_landmark)
        self.layout().addWidget(add_button)

        load_button = QPushButton('Load Existing Landmarks')
        load_button.clicked.connect(self.load)
        self.layout().addWidget(load_button)

        save_button = QPushButton('Save Landmarks')
        save_button.clicked.connect(self.save)
        self.layout().addWidget(save_button)


    def _refresh_layer_choices(self):
        """Used to refresh widget when a new layer is added or removed in napari so that it shows up in dropdown menu where you pick a layer to add the landmark to"""
        current = self.layer_landmarks_added.currentText()
        image_layer_names = [layer.name for layer in self.viewer.layers if isinstance(layer, napari.layers.Image)]
        self.layer_landmarks_added.blockSignals(True)
        self.layer_landmarks_added.clear()
        self.layer_landmarks_added.addItems(image_layer_names)
        if current in image_layer_names:
            self.layer_landmarks_added.setCurrentText(current)
        self.layer_landmarks_added.blockSignals(False)

    def add_landmark(self):
        layer_name = self.layer_landmarks_added.currentText() # text string containing name of currently selected layer
        image_layer = self.viewer.layers[layer_name] # the currently selected layer as an object (e.g. the atlas or the recorded image)

        landmark_name = self.name_input.text().strip() # get name of landmark
        if not landmark_name:
            print("Enter a landmark name first")
            return

        # add the landmark
        points_layer = get_or_create_landmarks_layer(self.viewer, image_layer)
        # clear the selection so you can write back onto selected points
        points_layer.selected_data = set()
        points_layer.feature_defaults = {
            'landmark_name': np.array([landmark_name]),
            'from_disk': np.array([False]),
        } # stamps the landmark name the user entered onto the *next* point added
        self.viewer.layers.selection.active = points_layer # makes the points layer the activated layer in napari
        points_layer.mode = 'add'

    def save(self):
        """Write the landmarks to file"""
        layer_name = self.layer_landmarks_added.currentText() # text string containing name of currently selected layer
        image_layer = self.viewer.layers[layer_name]  # the currently selected layer (e.g. the atlas or the recorded image)
        if image_layer is None:
            return

        points_name = f'{image_layer.name}_landmarks' # base of file name
        if points_name not in self.viewer.layers:
            print(f'No landmarks layer for {image_layer.name} to save. Add a landmark first.')
            return

        # check with user if they want to overwrite or append
        points_layer = self.viewer.layers[points_name]
        json_path = Path(f"{points_layer.metadata.get('source_path')}.landmarks.json")
        mode = 'overwrite'
        if json_path.exists():
            mode = self._ask_save_mode(json_path)
            if mode is None: # user cancelled
                return

        # write new landmarks to disk
        save_landmarks(self.viewer.layers[points_name], mode=mode)
        print(f'Saved landmarks to {json_path} ({mode})')

    def load(self):
        """Load existing landmarks"""

        layer_name = self.layer_landmarks_added.currentText() # text string containing name of currently selected layer
        image_layer = self.viewer.layers[layer_name] # the currently selected layer as an object (e.g. the atlas or the recorded image)

        # add the landmark
        points_layer = get_or_create_landmarks_layer(self.viewer, image_layer)


    def _ask_save_mode(self, json_path):
        """Ask whether to append new landmarks to existing landmarks file or overwrite it"""

        box = QMessageBox(self)
        box.setWindowTitle('Save Landmarks')
        box.setIcon(QMessageBox.Question)
        box.setText(f'{json_path.name} already exists')
        box.setInformativeText(
            'Append keeps landmarks that are already in the file. \n' \
            'Overwrite keeps only the landmarks you added in this session;\n' \
            'landmarks loaded from file are discarded.'
        )
        append_button = box.addButton('Append', QMessageBox.AcceptRole)
        overwrite_button = box.addButton('Overwrite', QMessageBox.DestructiveRole)
        box.addButton('Cancel', QMessageBox.RejectRole)
        box.exec_()

        clicked = box.clickedButton()
        if clicked is append_button:
            return 'append'
        if clicked is overwrite_button:
            return 'overwrite'
        return None

class AlignmentWidget(QWidget):
    def __init__(self, viewer):
        super().__init__()
        self.viewer = viewer
        self.setLayout(QVBoxLayout())

        self.layout().addWidget(QLabel('Layer to be moved (HQ file)'))
        self.moving_layer = QComboBox()
        self.layout().addWidget(self.moving_layer)

        self.layout().addWidget(QLabel("Fixed layer (Atlas)"))
        self.fixed_layer = QComboBox()
        self.layout().addWidget(self.fixed_layer)

        self._refresh_layer_choices()
        self.viewer.layers.events.inserted.connect(self._refresh_layer_choices)
        self.viewer.layers.events.removed.connect(self._refresh_layer_choices)

        align_button = QPushButton('Align to Atlas')
        align_button.clicked.connect(self.align)
        self.layout().addWidget(align_button)

        save_button = QPushButton('Save Alignment Matrix')
        save_button.clicked.connect(self.save)
        self.layout().addWidget(save_button)

        self.status_label = QLabel('')
        self.status_label.setWordWrap(True)
        self.layout().addWidget(self.status_label)

        self._last_alignment = None

    def _refresh_layer_choices(self):
        """Used to refresh widget when a new layer is added or removed in napari so that it shows up in dropdown menu where you pick a layer to add the landmark to"""
        image_layer_names = [layer.name for layer in self.viewer.layers if isinstance(layer, napari.layers.Image)]
        for hq_and_atlas in (self.moving_layer, self.fixed_layer):
            current = hq_and_atlas.currentText()
            hq_and_atlas.blockSignals(True)
            hq_and_atlas.clear()
            hq_and_atlas.addItems(image_layer_names)
            if current in image_layer_names:
                hq_and_atlas.setCurrentText(current)
            hq_and_atlas.blockSignals(False)

    def align(self):
        moving_layer_name = self.moving_layer.currentText()
        fixed_layer_name = self.fixed_layer.currentText()
        if not moving_layer_name or not fixed_layer_name or moving_layer_name == fixed_layer_name:
            self.status_label.setText("Pick two different layers (HQ and Atlas).")
            return
        moving_layer = self.viewer.layers[moving_layer_name]
        fixed_layer = self.viewer.layers[fixed_layer_name]

        source_path_moving = moving_layer.metadata.get('source_path')
        source_path_fixed = fixed_layer.metadata.get('source_path')

        moving_points_name = f'{moving_layer_name}_landmarks'
        if moving_points_name in self.viewer.layers:
            moving_points_layer = self.viewer.layers[moving_points_name]
            moving_landmarks = dict(zip(moving_points_layer.properties['landmark_name'], moving_points_layer.data))
        else:
            # if no active points layers containing landmarks
            moving_landmarks = load_landmarks(source_path_moving)

        fixed_points_name = f'{fixed_layer_name}_landmarks'
        if fixed_points_name in self.viewer.layers:
            fixed_points_layer = self.viewer.layers[fixed_points_name]
            fixed_landmarks = dict(zip(fixed_points_layer.properties['landmark_name'], fixed_points_layer.data))
        else:
            # if no active points layers containing landmarks
            fixed_landmarks = load_landmarks(source_path_fixed)

        landmark_names, moving_pts, fixed_pts = match_landmarks(moving_landmarks, fixed_landmarks)
        
        if len(landmark_names) < 3:
            self.status_label.setText(
                f"Need >=3 matching landmark names on both layers, found {len(landmark_names)}."
            )
            return
        
        # ensure that volume (moving) to be aligned and atlas (fixed) are on the same scale
        moving = moving_pts * np.array(moving_layer.scale[:3])
        fixed  = fixed_pts  * np.array(fixed_layer.scale[:3])

        transform_matrix = fit_similarity_transform(moving, fixed)
        residuals = transform_residuals(transform_matrix, moving, fixed)

        print("Transform matrix computed")

        moving_layer.affine = transform_matrix
        self._last_alignment = {
            'hq_source_path': source_path_moving,
            'atlas_source_path': source_path_fixed,
            'transform_matrix': transform_matrix,
            'hq_voxel_size_in_mm':moving_layer.scale[:3],
            "atlas_voxel_size_in_mm":fixed_layer.scale[:3],
            'landmark_names':landmark_names,
            'residuals_mm': residuals,
            #'moving_layer': moving_layer,
            #'fixed_layer': fixed_layer,
        }

        points_name = f'{moving_layer_name}_landmarks'
        if points_name in self.viewer.layers:
            points_layer = self.viewer.layers[points_name]
            points_layer.affine = to_layer_affine(transform_matrix, points_layer.ndim)

    def save(self):
        """Write the landmarks to file"""
        if self._last_alignment is None:
            self.status_label.setText("Run 'Align to Atlas' before saving.")
            return

        json_path = Path(f"{self._last_alignment['hq_source_path']}.alignment.json")
        if not json_path.exists() :
            save_alignment_matrix(**self._last_alignment)
            print(f'Saved transformation matrix for alignment to {json_path}')
            return
        
        # file with transformation matrix for alignment already exists, check with user if they want to overwrite
        mode = self._ask_save_mode(json_path)
        if mode is None:  
            # user cancelled
            return

        if mode == 'overwrite':
            save_alignment_matrix(**self._last_alignment)
            print(f'Saved transformation matrix for alignment to {json_path}')

            
    def _ask_save_mode(self, json_path):
        """If file with transformation matrix for alignment already exists, ask whether to overwrite it"""
        box = QMessageBox(self)
        box.setWindowTitle('Save Transformation Matrix for Alignment')
        box.setIcon(QMessageBox.Question)
        box.setText(f'A file with a transformation matrix for alignment already exists for this session: {json_path.name}. Do you want to overwrite it?')
        overwrite_button = box.addButton('Overwrite', QMessageBox.DestructiveRole)
        box.addButton('Cancel', QMessageBox.RejectRole)
        box.exec_()

        clicked = box.clickedButton()
        if clicked is overwrite_button:
            return 'overwrite'
        return None

# apply transform to new data widget
class AlignToAtlasWidget(QWidget):
    """Align a new acquisition (e.g. ...T_0.mat) to atlas using transformation matrix fitted to session's HQ volume"""

    def __init__(self, viewer):
        super().__init__()
        self.viewer = viewer
        self.setLayout(QVBoxLayout())

        self.layout().addWidget(QLabel('Align New Image'))
        align_button = QPushButton('Align New Image to Atlas')
        align_button.clicked.connect(self.align)
        self.layout().addWidget(align_button)

        # TODO: what are the next three lines for exactly?
        self.status_label = QLabel('')
        self.status_label.setWordWrap(True)
        self.layout().addWidget(self.status_label)

    def align(self):
        image_paths = select_files_from_gui("Select Non-HQ .MAT Files to Align (e.g. T_*.mat)", defaultextension='.mat')

        alignment_paths = select_files_from_gui(
            "Select an .alignment.json File",
            defaultextension='.json',
        )
        if not alignment_paths:
            self._report(f"No alignment file selected. Alignment skipped.")
            print("No alignment file selected. Quitting...")
            return
        
        alignment = load_alignment_matrix(alignment_paths[0])

        if not image_paths:
            print("No files selected. Quitting...")
            sys.exit()

        for image_path in image_paths:
            image_path = Path(image_path)

            image, voxel_size, origin, image_name = load_image(image_path)

            if voxel_size is None:
                self._report(f"No voxel size in {image_path.name}, cannot align it automatically to atlas.")

            # napari lines layers up by their trailing (last) axes, so time needs to go first for the spatial axes to be aligned with the atlas axes
            if image.ndim == len(voxel_size) + 1:
                image = np.moveaxis(image, -1, 0)
                scale = (1.0,) + tuple(voxel_size)
                spatial_axes = tuple(range(1, image.ndim))
            elif image.ndim == len(voxel_size):
                scale = tuple(voxel_size)
                spatial_axes = (0, 1, 2)
            else:
                print(f"{image_path.name}: image has {image.ndim} dimensions but voxel size has "
                f"{len(voxel_size)}; don't know which axes are spatial.")

            layer = self.viewer.add_image(
                data=image,
                name=image_path.stem,
                scale=scale,
                metadata={'source_path': str(image_path)}
            )
            layer.affine = to_layer_affine(alignment['transform_matrix'], ndim=image.ndim, spatial_axes=spatial_axes)

    def _find_alignment_file(self, image_path: Path):
        """Find the *.alignment.json sitting next to the selected file. One match is used
        as is; zero or several means the user picks."""
        candidates = sorted(image_path.parent.glob('*.alignment.json'))
        if len(candidates) == 1:
            return candidates[0]

        if not candidates:
            self._report(f"No .alignment.json found next to {image_path.name}. Select one.")
        else:
            self._report(
                f"{len(candidates)} .alignment.json files next to {image_path.name}. Select one."
            )

        selected = select_files_from_gui(
            f"Select the .alignment.json File to Use for {image_path.name}",
            defaultextension='.json',
        )
        if not selected:
            self._report(f"No alignment file selected; skipped {image_path.name}.")
            return None
        return selected[0]

    def _report(self, message: str):
        self.status_label.setText(message)
        print(message)

class RegisterToAreasWidget(QWidget):
    """Label every voxel of an aligned recording with the Allen brain structure it sits in.

    Adds a Labels layer on top of the recording; hovering a voxel with that layer selected shows
    the structure acronym, full name and major division in napari's status bar.
    """

    def __init__(self, viewer):
        super().__init__()
        self.viewer = viewer
        self.setLayout(QVBoxLayout())

        self.layout().addWidget(QLabel('Aligned Layer to Register'))
        self.image_layer = QComboBox()
        self.layout().addWidget(self.image_layer)

        self._refresh_layer_choices()
        self.viewer.layers.events.inserted.connect(self._refresh_layer_choices)
        self.viewer.layers.events.removed.connect(self._refresh_layer_choices)

        # napari cannot slice a rotated layer in 2D, so by default the recording is put back on its
        # own grid and the atlas is brought to it instead
        self.recording_space = QCheckBox('View in recording space (needed to be enable view in 2D)')
        self.recording_space.setChecked(True)
        self.layout().addWidget(self.recording_space)

        register_button = QPushButton('Register to Atlas Areas')
        register_button.clicked.connect(self.register)
        self.layout().addWidget(register_button)

        self.status_label = QLabel('')
        self.status_label.setWordWrap(True)
        self.layout().addWidget(self.status_label)

    def _refresh_layer_choices(self):
        """Used to refresh widget when a new layer is added or removed in napari so that it shows up in the dropdown menu"""
        current = self.image_layer.currentText()
        image_layer_names = [layer.name for layer in self.viewer.layers if isinstance(layer, napari.layers.Image)]
        self.image_layer.blockSignals(True)
        self.image_layer.clear()
        self.image_layer.addItems(image_layer_names)
        if current in image_layer_names:
            self.image_layer.setCurrentText(current)
        self.image_layer.blockSignals(False)

    def register(self):
        layer_name = self.image_layer.currentText()
        if not layer_name:
            self._report("Pick a layer to register.")
            return
        image_layer = self.viewer.layers[layer_name]

        # the Labels layer is 3D, and napari lines layers up by their trailing axes, so the spatial
        # axes have to be the last three for the labels to sit on top of the recording
        spatial_axes = tuple(image_layer.metadata.get('spatial_axes', range(image_layer.ndim - 3, image_layer.ndim)))

        transform_matrix = self._alignment_of(image_layer, spatial_axes)
        if transform_matrix is None:
            self._report(
                f"'{layer_name}' has not been aligned to the atlas yet. Run 'Align to Atlas' "
                "or load an existing alignment matrix from file."
            )
            return

        annotation_path = prompt_load_annotation()
        if not annotation_path:
            self._report("No annotation file selected. Registration skipped.")
            return

        # get atlas image and atlas voxel size
        annotation, annotation_voxel_size = load_atlas_image(annotation_path) # annotation and annotation_voxel_size are the same as atlas_image and atlas_voxel_size
        # get names of structures in atlas
        structures = load_structure_graph(Path(annotation_path).parent)

        voxel_size = np.asarray(image_layer.scale)[list(spatial_axes)]
        spatial_shape = tuple(np.asarray(image_layer.data.shape)[list(spatial_axes)].tolist())

        labels, fraction_outside = annotate_volume(spatial_shape, voxel_size, transform_matrix, annotation, annotation_voxel_size)

    def recording_space():
        raise NotImplementedError