import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dashboard_data import load_bundle  # noqa: E402


@pytest.fixture(scope="session")
def bundle():
    return load_bundle(ROOT / "data")
