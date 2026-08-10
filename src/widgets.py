from qtpy.QtWidgets import QVBoxLayout, QWidget, QLabel, QLineEdit, QPushButton
from src.viewer_ops import *

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