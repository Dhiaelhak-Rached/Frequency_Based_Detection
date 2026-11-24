"""Isolated PX4 SITL + Gazebo simulation package for acoustic guidance validation."""

from pathlib import Path
import sys

_SRC = str(Path(__file__).resolve().parent.parent / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
