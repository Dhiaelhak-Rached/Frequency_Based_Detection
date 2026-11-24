#!/usr/bin/env python3
"""Root compatibility shim delegating to ``frequency_id.ops.jetson_ctl:main``."""

from pathlib import Path
import sys

_SRC = str(Path(__file__).resolve().parent / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from frequency_id.ops.jetson_ctl import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
