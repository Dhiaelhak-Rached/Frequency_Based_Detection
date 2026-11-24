"""Canonical project-root, configuration, model, report, and external-tool path resolution."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
from typing import Optional, Union


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


def _get_git_common_worktree_root(root: Path) -> Optional[Path]:
    """If ``root`` is a linked git worktree, return the primary checkout root."""
    dot_git = root / ".git"
    if not dot_git.is_file():
        return None
    try:
        content = dot_git.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not content.startswith("gitdir:"):
        return None
    raw_gitdir = content.split(":", 1)[1].strip()
    gitdir = Path(raw_gitdir)
    if not gitdir.is_absolute():
        gitdir = (root / gitdir).resolve()
    else:
        gitdir = gitdir.resolve()
    # Linked worktrees point to <common_git_dir>/worktrees/<name>.
    if gitdir.parent.name != "worktrees":
        return None
    common_git_dir = gitdir.parent.parent
    if common_git_dir.name != ".git":
        return None
    main_root = common_git_dir.parent
    return main_root if main_root.is_dir() and main_root != root else None


def resolve_project_path(value: Union[str, Path]) -> Path:
    """Resolve a path relative to the project root (or return it unchanged if absolute).

    When running inside a linked git worktree, untracked large assets (such as
    downloaded model checkpoints) that are absent from the worktree fall back to
    the same relative path in the primary checkout if present there.
    """
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    root = get_project_root()
    candidate = root / path
    if candidate.exists():
        return candidate
    main_root = _get_git_common_worktree_root(root)
    if main_root is not None:
        shared_candidate = main_root / path
        if shared_candidate.exists():
            return shared_candidate
    return candidate


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
