"""Test config: make the `skywatch` package importable even without an editable
install, and expose the sample-response fixture path."""

from __future__ import annotations

import sys
from pathlib import Path

# backend/ on sys.path so `import skywatch` works when running pytest in-place.
BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import pytest

DATA_DIR = Path(__file__).resolve().parent / "data"


@pytest.fixture
def sample_states_path() -> Path:
    return DATA_DIR / "sample_states.json"
