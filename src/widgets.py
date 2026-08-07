from qtpy.QtWidgets import QVBoxLayout, QWidget, QLabel, QLineEdit, QPushButton, QComboBox
import napari
import time

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

class AddLandmark(QWidget):
    def __init__(self, viewer):
        super().__init__()
        self.viewer = viewer
        self.setLayout(QVBoxLayout())

        self.layout().addWidget(QLabel('Target Image Layer:'))
        self.layer_combo = QComboBox()
        self.layout().addWidget(self.layer_combo) # adds a blank dropdown menu
        self._refresh_layer_choices()
        self.viewer.layers.events.inserted.connect(self._refresh_layer_choices) # refresh when a new layer is added to napari so that it shows up in dropdown menu
        self.viewer.layers.events.removed.connect(self._refresh_layer_choices) # refresh when a new layer is removed from napari so that it shows up in dropdown menu

        self.layout().addWidget(QLabel('Landmark Name:'))
        self.name_input = QLineEdit()
        self.layout().addWidget(self.name_input)

        add_button = QPushButton('Add Landmark')
        add_button.clicked.connect(self.add_landmark)


    def _refresh_layer_choices(self):
        current = self.layer_combo.currentText()
        image_layer_names = [layer.name for layer in self.viewer.layers if isinstance(layer, napari.layers.Image)]
        self.layer_combo.blockSignals(True)
        self.layer_combo.clear()
        self.layer_combo.addItems(image_layer_names)
        if current in image_layer_names:
            self.layer_combo.setCurrentText(current)
        self.layer_combo.blockSignals(False)

    def add_landmark(self):
        return NotImplementedError
              
