import pytest
from pathlib import Path

# the data folder sits next to the FusiConverter folder
DATA_DIR = Path(__file__).resolve().parents[2] / 'data'


def pytest_collection_modifyitems(items):
    """Skip tests marked 'data' when ../data isn't there (e.g. a fresh clone), rather than failing them"""
    if DATA_DIR.exists():
        return
    skip_data = pytest.mark.skip(reason=f'data folder not found at {DATA_DIR}')
    for item in items:
        if item.get_closest_marker('data'):
            item.add_marker(skip_data)
