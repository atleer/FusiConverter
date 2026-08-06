import napari
import numpy as np
from qtpy.QtWidgets import QVBoxLayout, QWidget, QLabel, QLineEdit, QPushButton, QComboBox

from src.utils import (
    crop_image,
    apply_log_normalization,
    get_or_create_landmarks_layer,
    load_landmarks,
    match_named_points,
    fit_similarity_transform,
    transform_residuals_mm,
    save_registration,
    resample_volume_to_atlas,
)

## Crop Widget
class CropWidget(QWidget):
    def __init__(self, viewer):
        super().__init__()
        self.viewer = viewer
        self.setLayout(QVBoxLayout())

        self.layout().addWidget(QLabel('Z Range (e.g., 0:10):'))
        self.z_range_input = QLineEdit()
        self.layout().addWidget(self.z_range_input)

        self.layout().addWidget(QLabel('X Range (e.g., 0:100):'))
        self.x_range_input = QLineEdit()
        self.layout().addWidget(self.x_range_input)

        self.layout().addWidget(QLabel('Y Range (e.g., 0:100):'))
        self.y_range_input = QLineEdit()
        self.layout().addWidget(self.y_range_input)

        crop_button = QPushButton('Crop Image & Log-Norm')
        crop_button.clicked.connect(self.crop)
        self.layout().addWidget(crop_button)

    def crop(self):
        z_range = self.z_range_input.text()
        x_range = self.x_range_input.text()
        y_range = self.y_range_input.text()
        crop_image(self.viewer, z_range, x_range, y_range)
        apply_log_normalization(self.viewer)

## Landmark Widget
class LandmarkWidget(QWidget):
    def __init__(self, viewer):
        super().__init__()
        self.viewer = viewer
        self.setLayout(QVBoxLayout())

        self.layout().addWidget(QLabel('Target Image Layer:'))
        self.layer_combo = QComboBox()
        self.layout().addWidget(self.layer_combo)
        self._refresh_layer_choices()
        self.viewer.layers.events.inserted.connect(self._refresh_layer_choices)
        self.viewer.layers.events.removed.connect(self._refresh_layer_choices)

        self.layout().addWidget(QLabel('Landmark Name:'))
        self.name_input = QLineEdit()
        self.layout().addWidget(self.name_input)

        add_button = QPushButton('Add Landmark')
        add_button.clicked.connect(self.add_landmark)
        self.layout().addWidget(add_button)

    def _refresh_layer_choices(self, event=None):
        current = self.layer_combo.currentText()
        image_layer_names = [layer.name for layer in self.viewer.layers if isinstance(layer, napari.layers.Image)]
        self.layer_combo.blockSignals(True)
        self.layer_combo.clear()
        self.layer_combo.addItems(image_layer_names)
        if current in image_layer_names:
            self.layer_combo.setCurrentText(current)
        self.layer_combo.blockSignals(False)

    def add_landmark(self):
        layer_name = self.layer_combo.currentText()
        if not layer_name or layer_name not in self.viewer.layers:
            print("No image layer selected.")
            return
        image_layer = self.viewer.layers[layer_name]

        name = self.name_input.text().strip()
        if not name:
            print("Enter a landmark name first.")
            return

        points_layer = get_or_create_landmarks_layer(self.viewer, image_layer)
        points_layer.current_properties = {'name': np.array([name])}
        self.viewer.layers.selection.active = points_layer
        points_layer.mode = 'add'


## Registration Widget
class RegistrationWidget(QWidget):
    def __init__(self, viewer):
        super().__init__()
        self.viewer = viewer
        self.setLayout(QVBoxLayout())

        self.layout().addWidget(QLabel('Moving (HQ) Layer:'))
        self.moving_combo = QComboBox()
        self.layout().addWidget(self.moving_combo)

        self.layout().addWidget(QLabel('Fixed (Atlas) Layer:'))
        self.fixed_combo = QComboBox()
        self.layout().addWidget(self.fixed_combo)

        self._refresh_layer_choices()
        self.viewer.layers.events.inserted.connect(self._refresh_layer_choices)
        self.viewer.layers.events.removed.connect(self._refresh_layer_choices)

        compute_button = QPushButton('Compute && Preview Registration')
        compute_button.clicked.connect(self.compute_registration)
        self.layout().addWidget(compute_button)

        preview_button = QPushButton('Preview Registered HQ (resampled, for 2D)')
        preview_button.clicked.connect(self.preview_resampled)
        self.layout().addWidget(preview_button)

        self.status_label = QLabel('')
        self.status_label.setWordWrap(True)
        self.layout().addWidget(self.status_label)

        self._last_registration = None

    def _refresh_layer_choices(self, event=None):
        image_layer_names = [layer.name for layer in self.viewer.layers if isinstance(layer, napari.layers.Image)]
        for combo in (self.moving_combo, self.fixed_combo):
            current = combo.currentText()
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(image_layer_names)
            if current in image_layer_names:
                combo.setCurrentText(current)
            combo.blockSignals(False)

    def compute_registration(self):
        moving_name = self.moving_combo.currentText()
        fixed_name = self.fixed_combo.currentText()
        if not moving_name or not fixed_name or moving_name == fixed_name:
            self.status_label.setText("Pick two different layers (HQ and Atlas).")
            return
        moving_layer = self.viewer.layers[moving_name]
        fixed_layer = self.viewer.layers[fixed_name]

        moving_source = moving_layer.metadata.get('source_path')
        fixed_source = fixed_layer.metadata.get('source_path')
        if moving_source is None or fixed_source is None:
            self.status_label.setText("Both layers need a known source file.")
            return

        moving_landmarks = load_landmarks(moving_source)
        fixed_landmarks = load_landmarks(fixed_source)
        names, moving_pts, fixed_pts = match_named_points(moving_landmarks, fixed_landmarks)
        if len(names) < 3:
            self.status_label.setText(
                f"Need >=3 matching landmark names on both layers, found {len(names)}."
            )
            return

        moving_mm = moving_pts * np.array(moving_layer.scale[:3])
        fixed_mm = fixed_pts * np.array(fixed_layer.scale[:3])

        matrix_mm = fit_similarity_transform(moving_mm, fixed_mm)
        residuals_mm = transform_residuals_mm(matrix_mm, moving_mm, fixed_mm)

        moving_layer.affine = matrix_mm
        self._last_registration = {
            'moving_layer': moving_layer,
            'fixed_layer': fixed_layer,
            'matrix_mm': matrix_mm,
        }

        json_path = save_registration(
            hq_source_path=moving_source,
            atlas_source_path=fixed_source,
            matrix_mm=matrix_mm,
            hq_voxel_size_mm=moving_layer.scale[:3],
            atlas_voxel_size_mm=fixed_layer.scale[:3],
            landmark_names=names,
            residuals_mm=residuals_mm,
        )

        rms = float(np.sqrt(np.mean(np.square(residuals_mm))))
        max_err = float(np.max(residuals_mm))
        self.status_label.setText(
            f"Registered using {len(names)} landmarks. RMS error: {rms:.3f} mm, "
            f"max: {max_err:.3f} mm.\nSaved to {json_path}"
        )

    def preview_resampled(self):
        if self._last_registration is None:
            self.status_label.setText("Compute a registration first.")
            return

        moving_layer = self._last_registration['moving_layer']
        fixed_layer = self._last_registration['fixed_layer']
        matrix_mm = self._last_registration['matrix_mm']

        resampled = resample_volume_to_atlas(
            moving_layer.data,
            source_voxel_size_mm=moving_layer.scale[:3],
            matrix_mm=matrix_mm,
            atlas_shape=fixed_layer.data.shape,
            atlas_voxel_size_mm=fixed_layer.scale[:3],
        )
        self.viewer.add_image(
            resampled,
            name=f'{moving_layer.name}_registered_preview',
            scale=fixed_layer.scale,
        )
        self.status_label.setText(
            f"Added '{moving_layer.name}_registered_preview' (axis-aligned in atlas space, "
            "safe to scrub in 2D)."
        )