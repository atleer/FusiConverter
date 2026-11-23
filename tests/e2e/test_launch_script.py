"""
Run the launch script as the .bat does, with napari.run and the alignment prompt patched out, and check what it sets up
"""
import functools
import runpy
from pathlib import Path
import numpy as np
import pytest
import napari
from qtpy.QtWidgets import QMessageBox

from fusiconverter.widgets import AddLandmark, AlignmentWidget, ManualAlignmentWidget, RegisterToAreasWidget

pytestmark = [pytest.mark.data, pytest.mark.gui]

LAUNCH_SCRIPT = Path(__file__).resolve().parents[2] / 'scripts' / 'launch_napari_with_widgets.py'


@pytest.fixture
def launched(monkeypatch, tmp_path):
    """The launch script's globals after it has run, with the viewer hidden, napari.run returning straight away and 'Yes' to the alignment prompt"""
    monkeypatch.chdir(tmp_path) # so the script's os.chdir is undone afterwards
    monkeypatch.setattr(napari, 'Viewer', functools.partial(napari.Viewer, show=False))
    monkeypatch.setattr(napari, 'run', lambda: None)
    monkeypatch.setattr(QMessageBox, 'question', lambda *args, **kwargs: QMessageBox.Yes)
    script_globals = runpy.run_path(str(LAUNCH_SCRIPT))
    yield script_globals
    script_globals['viewer'].close()


def test_launch_docks_the_four_widgets(launched):
    """The viewer has exactly the four widgets docked"""
    docked = [dock_widget.widget() for dock_widget in launched['viewer'].window._dock_widgets.values()]
    assert sorted(type(widget).__name__ for widget in docked) == sorted(
        cls.__name__ for cls in (AddLandmark, AlignmentWidget, ManualAlignmentWidget, RegisterToAreasWidget)
    )

def test_launch_opens_the_atlases(launched):
    """Both atlases are opened, the vascular reference atlas unmoved and the Allen atlas moved by its saved alignment"""
    layers = launched['viewer'].layers
    assert 'VascularProbabilisticAtlas' in layers
    assert 'average_template_50' in layers
    assert np.allclose(layers['VascularProbabilisticAtlas'].affine.affine_matrix, np.eye(4))
    assert not np.allclose(layers['average_template_50'].affine.affine_matrix, np.eye(4))
