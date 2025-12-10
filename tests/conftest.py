from pathlib import Path

import pytest
from fixtures import *  # noqa: F401, F403


def pytest_collection_modifyitems(items):
    for item in items:
        item_path = Path(item.nodeid.split("::")[0])
        if "integration" in item_path.parts:
            item.add_marker(pytest.mark.integration)
