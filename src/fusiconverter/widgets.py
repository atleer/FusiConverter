import numpy as np
import napari
from qtpy.QtWidgets import QVBoxLayout, QWidget, QLabel, QLineEdit, QPushButton, QComboBox, QMessageBox, QCheckBox
from src.fusiconverter.viewer_ops import (
    save_landmarks, 
    load_landmarks, 
    get_or_create_landmarks_layer, 
    match_landmarks, fit_similarity_transform, 
    transform_residuals, 
    put_transform_matrix_in_layer_affine, 
    get_transform_matrix_from_layer_affine, 
    save_transform_matrix, 
    load_transform_matrix, 
    annotate_volume, 
    resample_atlas_to_recording,
    summarize_areas,
    save_area_map
)
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
from napari.utils.colormaps import DirectLabelColormap

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
        moving_layer.metadata['alignment_matrix'] = transform_matrix
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
            points_layer.affine = put_transform_matrix_in_layer_affine(transform_matrix, points_layer.ndim)

    def save(self):
        """Write the landmarks to file"""
        if self._last_alignment is None:
            self.status_label.setText("Run 'Align to Atlas' before saving.")
            return

        json_path = Path(f"{self._last_alignment['hq_source_path']}.alignment.json")
        if not json_path.exists() :
            save_transform_matrix(**self._last_alignment)
            print(f'Saved transformation matrix for alignment to {json_path}')
            return
        
        # file with transformation matrix for alignment already exists, check with user if they want to overwrite
        mode = self._ask_save_mode(json_path)
        if mode is None:  
            # user cancelled
            return

        if mode == 'overwrite':
            save_transform_matrix(**self._last_alignment)
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

        # sets the status to empty so that it can be populated by _report later
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
        
        alignment = load_transform_matrix(alignment_paths[0])

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
            layer.metadata['transform_matrix'] = alignment['transform_matrix']
            layer.affine = put_transform_matrix_in_layer_affine(alignment['transform_matrix'], ndim=image.ndim, spatial_axes=spatial_axes)

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

        register_button = QPushButton('Register to Atlas Areas')
        register_button.clicked.connect(self.register)
        self.layout().addWidget(register_button)

        # TODO: implement _save function so that only argument is self
        # save_button = QPushButton('Save Registered Areas')
        # save_button.clicked.connect(self._save)
        # self.layout().addWidget(save_button)

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

        transform_matrix = self._get_transform_matrix(image_layer, spatial_axes)
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
        structures = load_structure_graph(Path(annotation_path).parent / 'allen_mouse_connectivity_structure_graph.csv')

        voxel_size = np.asarray(image_layer.scale)[list(spatial_axes)]
        spatial_shape = tuple(np.asarray(image_layer.data.shape)[list(spatial_axes)].tolist())

        labels, fraction_outside = annotate_volume(spatial_shape, voxel_size, transform_matrix, annotation, annotation_voxel_size)

        summary = summarize_areas(labels, structures)

        if not summary:
            self._report(
                f"No voxel of '{layer_name}' landed on a labelled atlas structure "
                f"({fraction_outside:.0%} of the volume fell outside the atlas). Check the alignment."
            )
            return

        self._to_recording_space(image_layer, transform_matrix, spatial_shape, voxel_size,
                                    annotation.shape, annotation_voxel_size)

        self._add_labels_layer(image_layer, labels, structures, annotation_path)
        self._save(image_layer, summary, labels, annotation_path, fraction_outside) # TODO: remove this when _save button with ask to overwrite has been properly implemented
        self._report_summary(layer_name, summary, labels, fraction_outside)

    def _get_transform_matrix(self, image_layer, spatial_axes):
        """Get the matrix used to align the recording's to the atlas, or return None if it has not been aligned.

        Normally, the transform matrix is the layer's affine, but once the layer has been put back into recording space
        the affine is identity again, so the matrix is kept in the layer metadata as well.
        """
        layer_affine = np.asarray(image_layer.affine.affine_matrix)
        if not np.allclose(layer_affine, np.eye(image_layer.ndim + 1)):
            return get_transform_matrix_from_layer_affine(layer_affine, spatial_axes)
        transform_matrix = image_layer.metadata.get('transform_matrix')
        return None if transform_matrix is None else np.asarray(transform_matrix)

    def _to_recording_space(self, image_layer, transform_matrix, spatial_shape, voxel_size, annotation_shape, annotation_voxel_size):
        """Put the recording back on its own grid and bring the atlas.
        
        A rotated layer cannot be sliced in 2D by napari, so the recording needs to keep its own axis 
        and the atlas template is instead resampled onto the recording's axes. 
        """

        image_layer.metadata['transform_matrix'] = transform_matrix
        image_layer.affine = np.eye(image_layer.ndim + 1)

        landmarks_name = f'{image_layer.name}_landmarks'
        if landmarks_name in self.viewer.layers:
            landmarks_layer = self.viewer.layers[landmarks_name]
            landmarks_layer.affine = np.eye(landmarks_layer.ndim + 1)

        # any unaligned 3D image layer on the annotation's grid is an atlas
        for layer in list(self.viewer.layers):
            if layer is image_layer or not isinstance(layer, napari.layers.Image):
                continue
            if layer.ndim != 3 or tuple(layer.data.shape) != tuple(annotation_shape):
                continue
            if not np.allclose(np.asarray(layer.affine.affine_matrix), np.eye(4)):
                continue


            resampled_name = f'{layer.name}_in_{image_layer.name}'
            if resampled_name in self.viewer.layers:
                # start from scratch
                self.viewer.layers.remove(resampled_name)

            resampled = resample_atlas_to_recording(np.asarray(layer.data), np.asarray(layer.scale), transform_matrix, spatial_shape, voxel_size, order = 1)

            self.viewer.add_image(resampled, name = resampled_name, scale = voxel_size, blending='additive', metadata={'source_path': layer.metadata.get('source_path')})
            layer.visible = False # the original atlas no longer lines up with the recorded brain, so make it invisible

    def _add_labels_layer(self, image_layer, labels, structures, annotation_path):
        "Put the structure ids on screen in colours on top of the recording"

        # next two lines: [[0, 0 , 985], [315, 315, 985], [672, 0, 985]] -> [[0, 0 , 3], [1, 1, 3], [2, 0, 3]]
        # only the indeces of different labels within compact are kept. Needed for napari's DirectLabelColormap. Labels can be reconstructed with unique_ids[compact] 
        unique_ids = np.unique(labels)
        if unique_ids[0] != 0:
            unique_ids = np.concatenate([[0], unique_ids])
        compact = np.searchsorted(unique_ids, labels).astype(np.uint16)

        spatial_axes = list(range(image_layer.ndim - 3, image_layer.ndim))
        layer_affine = np.asarray(image_layer.affine.affine_matrix)

        labels_affine = np.eye(4) # note: next two lines are only needed if self.recording_space.isChecked() isn't commented out
        labels_affine[:3, :3] = layer_affine[np.ix_(spatial_axes, spatial_axes)]
        labels_affine[:3, 3] = layer_affine[spatial_axes, image_layer.ndim]
    
        # napari matches a features row to a label through the 'index' column, positionally - so
        # these have to be plain lists in the same order, not a filtered dataframe
        colors = {None: (0.0, 0.0, 0.0, 0.0)} # anything unlabelled stays transparent
        features = {'index': [], 'acronym': [], 'name': [], 'division': [], 'structure_id': []}
        for compact_id, structure_id in enumerate(unique_ids.tolist()):
            if structure_id == 0: # background
                continue
            structure = structures.get(structure_id, {})
            colors[compact_id] = tuple(channel / 255 for channel in structure.get('rgb', (255, 255, 255))) + (1.0,)
            features['index'].append(compact_id)
            features['acronym'].append(structure.get('acronym', f'unknown_{structure_id}'))
            features['name'].append(structure.get('name', 'not in the structure graph'))
            features['division'].append(structure.get('division', ''))
            features['structure_id'].append(structure_id)

        labels_layer = self.viewer.add_labels(
            compact,
            name=f'{image_layer.name}_areas',
            scale=np.asarray(image_layer.scale)[-3:],
            affine=labels_affine,
            opacity=0.5,
            colormap=DirectLabelColormap(color_dict=colors),
            features=features,
            metadata={
                'source_path': image_layer.metadata.get('source_path'),
                'annotation_source_path': str(annotation_path),
                'structure_ids': unique_ids, # unique_ids[label value] is the Allen structure id
            },
        )
        # select it so hovering reports the structure names in napari's status bar
        self.viewer.layers.selection.active = labels_layer
        return labels_layer

    def _save(self, image_layer, summary, labels, annotation_path, fraction_outside):
        """Write the registration to atlas areas to file"""
        source_path = image_layer.metadata.get('source_path')
        if source_path is None:
            print(f"No source path for '{image_layer.name}', not writing an area map.")
            return

        json_path = Path(f'{source_path}.areas.json')
        if json_path.exists() and not self._ask_overwrite(json_path):
            return

        save_area_map(source_path, summary, labels, annotation_path, fraction_outside)
        print(f'Saved area map to {json_path}')

    
    def _report_summary(self, layer_name, summary, labels, fraction_outside):
        top = ' · '.join(f"{structure['acronym']} ({structure['fraction_of_recording']:.0%})"
                         for structure in summary[:6])
        unlabelled = np.count_nonzero(labels == 0) / labels.size
        self._report(
            f"{layer_name}: {len(summary)} structures. Most covered: {top}. "
            f"Unlabelled {unlabelled:.0%}, outside atlas {fraction_outside:.0%}."
        )
        print(f'\nStructures covered by {layer_name} ({len(summary)} in total, all of them in the .areas.json):')
        for structure in summary[:25]:
            print(f"  {structure['acronym']:<12} {structure['n_voxels']:>8} voxels "
                  f"({structure['fraction_of_recording']:6.2%})  {structure['division']:<10} {structure['name']}")
        if len(summary) > 25:
            print(f'  ... and {len(summary) - 25} smaller structures')
            
    def _ask_overwrite(self, json_path):
        box = QMessageBox(self)
        box.setWindowTitle('Save Area Map')
        box.setIcon(QMessageBox.Question)
        box.setText(f'An area map already exists for this recording: {json_path.name}. Do you want to overwrite it?')
        overwrite_button = box.addButton('Overwrite', QMessageBox.DestructiveRole)
        box.addButton('Cancel', QMessageBox.RejectRole)
        box.exec_()
        return box.clickedButton() is overwrite_button

    def _report(self, message: str):
            self.status_label.setText(message)
            print(message)