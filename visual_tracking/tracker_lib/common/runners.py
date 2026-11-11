"""
runners.py - loads a tracker by name: trackers/<name>/tracker.py -> make(cfg) -> a BaseTracker (init / update).
(The full benchmark project has more here; this integration only vendors DyHiT.)
"""
import importlib.util
import os
import sys

from common.bench_core import ROOT

TRACKERS = ["dyhit"]

_MODULES = {}


def load_tracker(name, cfg):
    """Import trackers/<name>/tracker.py (once) and call its make(cfg) to get a tracker."""
    if name not in _MODULES:
        path = os.path.join(ROOT, "trackers", name, "tracker.py")
        if not os.path.exists(path):
            sys.exit(f"Unknown tracker '{name}' (no {path}); available: {', '.join(TRACKERS)}")
        spec = importlib.util.spec_from_file_location(f"tracker_{name}", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _MODULES[name] = mod
    return _MODULES[name].make(cfg)
