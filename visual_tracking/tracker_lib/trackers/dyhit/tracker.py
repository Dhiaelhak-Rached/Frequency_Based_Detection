"""dyhit: DyHiT (from the HiT repo, MIT), dynamic HiT that routes easy frames through a lighter path
(111 FPS Jetson AGX, LaSOT 62.4). PyTorch, GPU or CPU. Setup: trackers/dyhit/README.md."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from common.hit_family import make_cached  # noqa: E402


def make(cfg):
    return make_cached(cfg, "dyhit", dynamic=True)
