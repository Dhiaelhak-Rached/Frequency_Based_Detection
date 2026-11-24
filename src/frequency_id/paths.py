"""Canonical project-root, configuration, model, report, and external-tool path resolution."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
from typing import Union


def get_project_root() -> Path:
    """Return the canonical project root directory.

    Resolution order:
    1. ``FREQUENCY_ID_ROOT`` environment variable (if set).
    2. Nearest ancestor of this file containing ``pyproject.toml``.
    3. Current working directory fallback.
    """
    env_root = os.environ.get("FREQUENCY_ID_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()

    here = Path(__file__).resolve()
    for candidate in (here.parent, *here.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate

    return Path.cwd().resolve()


PROJECT_ROOT: Path = get_project_root()
MODEL_DIR: Path = PROJECT_ROOT / "model"
CONFIG_DIR: Path = PROJECT_ROOT / "config"
REPORTS_DIR: Path = PROJECT_ROOT / "sim" / "reports"
TRACKER_LIB_DIR: Path = PROJECT_ROOT / "visual_tracking" / "tracker_lib"


def resolve_project_path(value: Union[str, Path]) -> Path:
    """Resolve a path relative to the project root (or return it unchanged if absolute)."""
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return get_project_root() / path


def get_px4_dir() -> Path:
    """Return the local PX4-Autopilot checkout path for SITL runs."""
    env_px4 = os.environ.get("PX4_DIR")
    if env_px4:
        return Path(env_px4).expanduser()
    return Path.home() / "PX4-Autopilot"


def get_default_rig_config_path() -> Path:
    """Return ``<PROJECT_ROOT>/rig_config.json``, bootstrapping from the example template if absent."""
    root = get_project_root()
    config_path = root / "rig_config.json"
    if not config_path.exists():
        example_path = root / "config" / "rig_config.example.json"
        if example_path.is_file():
            shutil.copy2(example_path, config_path)
    return config_path
