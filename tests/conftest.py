"""Pytest configuration ensuring src/ and repository root are importable across pytest versions."""

from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _REPO_ROOT / "src"

for _path in (str(_SRC_DIR), str(_REPO_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from tests.helpers import (  # noqa: E402, F401
    D,
    FakeBoxes,
    FakeCapture,
    FakeModel,
    FakePrediction,
    FakeTensor,
    FakeTracker,
)
