"""
torch_tracker.py - shared code for the PyTorch trackers that come from research repos cloned into external/
(hit_tiny, hit_small, hit_base, dyhit, mixformerv2_s, litetrack_b6). Same approach as trackers/sutrack/tracker.py:

  * the repo's own test-time tracker class (lib/test/tracker/*.py) is used unchanged, through
    initialize(rgb_image, {"init_bbox": [x, y, w, h]}) and track(rgb_image) -> {"target_bbox": [x, y, w, h], ...};
  * frames are converted BGR -> RGB, and the current CUDA stream is synchronized after each GPU update so the
    timing includes this tracker's GPU work without waiting for the separate detector stream;
  * "device" in config.json: "auto" (GPU if available), "cuda" or "cpu". These repos hard-code .cuda(), so CPU mode
    turns .cuda() into a no-op for the whole process (one tracker per process, so nothing else is affected).
"""
import os
import platform
import sys

from common.bench_core import BaseTracker, cv2, rootpath

VENV_HINT = os.path.join(".venv-sutrack", "Scripts" if os.name == "nt" else "bin", "python")  # as in the README


def import_torch():
    try:
        import torch
    except ImportError:
        sys.exit("This tracker needs PyTorch, which is only in its own environment. Run it with\n"
                 f"  {VENV_HINT} " + " ".join(sys.argv) + "\n"
                 "(in VS Code: Python: Select Interpreter -> .venv-sutrack)")
    _training_import_shims()
    return torch


def _training_import_shims():
    """The repos' test classes import things tracking never uses: torch._six (removed in PyTorch 2.0) and jpeg4py
    (a JPEG reader for training data that needs a native library) through their training package lib.train, and
    visdom (a live debug dashboard, only used with debug on; its package no longer builds on Python 3.11) in
    LiteTrack. Stand-ins are installed only when the real module is missing, so the import succeeds."""
    import collections.abc
    import types
    try:
        import torch._six  # noqa: F401
    except ImportError:
        six = types.ModuleType("torch._six")
        six.string_classes = (str, bytes)
        six.int_classes = int
        six.container_abcs = collections.abc
        six.inf = float("inf")
        sys.modules["torch._six"] = six
    try:
        import jpeg4py  # noqa: F401
    except ImportError:
        sys.modules["jpeg4py"] = types.ModuleType("jpeg4py")
    try:
        import visdom  # noqa: F401
    except ImportError:
        vis = types.ModuleType("visdom")
        vis.server = types.ModuleType("visdom.server")
        sys.modules["visdom"], sys.modules["visdom.server"] = vis, vis.server


def add_repo(cfg, key, url):
    """Put the cloned repo (config.json key '<key>') on sys.path; its code is imported as the package 'lib'."""
    repo = rootpath(cfg.get(key))
    if not repo or not os.path.isdir(os.path.join(repo, "lib")):
        sys.exit(f'"{key}" in config.json must point to a clone of {url} (git clone {url} {cfg.get(key)})')
    sys.path.insert(0, repo)
    return repo


def checkpoint_path(cfg, key, page):
    ckpt = rootpath(cfg.get(key))
    if not ckpt or not os.path.exists(ckpt):
        sys.exit(f'"{key}" in config.json must point to the checkpoint ({ckpt} not found); see {page}')
    return ckpt


def setup_device(torch, wanted):
    """Return (device, description): device is 'cuda' or 'cpu'; the description goes into every result summary."""
    wanted = (wanted or "auto").lower()
    if wanted == "auto":
        wanted = "cuda" if torch.cuda.is_available() else "cpu"
    if wanted == "cuda":
        if not torch.cuda.is_available():
            sys.exit('device "cuda" asked in config.json, but PyTorch sees no CUDA GPU; set it to "cpu" or "auto"')
        return "cuda", f"cuda ({torch.cuda.get_device_name(0)})"
    if wanted != "cpu":
        sys.exit(f'device must be "auto", "cuda" or "cpu" in config.json (got "{wanted}")')
    # The repos call .cuda() on the network, the preprocessing and a few buffers: keep everything on the CPU.
    torch.Tensor.cuda = lambda self, *a, **k: self
    torch.nn.Module.cuda = lambda self, *a, **k: self
    cpu = platform.processor() or platform.machine()
    return "cpu", f"cpu ({cpu}, {torch.get_num_threads()} threads)"


def allow_full_checkpoints(torch):
    """PyTorch >= 2.6 loads with weights_only=True by default, which rejects the extra objects stored in these
    research checkpoints. The files come from the tracker authors, so allow them (same as sutrack)."""
    orig = torch.load

    def _load(*a, **k):
        k.setdefault("weights_only", False)
        return orig(*a, **k)
    torch.load = _load


def make_params(cfg_node, checkpoint, **extra):
    """The repos' TrackerParams, filled from their config (the repos' own parameter files need a local.py)."""
    from lib.test.utils import TrackerParams
    p = TrackerParams()
    p.cfg = cfg_node
    p.template_factor = cfg_node.TEST.TEMPLATE_FACTOR
    p.template_size = cfg_node.TEST.TEMPLATE_SIZE
    p.search_factor = cfg_node.TEST.SEARCH_FACTOR
    p.search_size = cfg_node.TEST.SEARCH_SIZE
    p.checkpoint = checkpoint
    p.save_all_boxes = False
    p.debug = 0
    for k, v in extra.items():
        setattr(p, k, v)
    return p


class RepoTracker(BaseTracker):
    """Wraps a repo tracker object that has initialize(image, info) and track(image) -> dict (RGB images)."""

    def __init__(self, name, repo_tracker, torch, device, device_desc, score_fn=None):
        self.name = name
        self.t = repo_tracker
        self.torch = torch
        self.cuda = device == "cuda"
        self.device = device_desc
        self.score_fn = score_fn

    @staticmethod
    def _rgb(frame):
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if cv2 is not None else frame[:, :, ::-1].copy()

    def init(self, frame, box):
        self.t.initialize(self._rgb(frame), {"init_bbox": [float(v) for v in box]})
        if self.cuda:
            self.torch.cuda.current_stream().synchronize()

    def update(self, frame):
        out = self.t.track(self._rgb(frame))
        if self.cuda:
            self.torch.cuda.current_stream().synchronize()  # include this tracker's GPU work only
        score = None
        if self.score_fn is not None:
            try:
                score = float(self.score_fn(out))
            except Exception:
                score = None
        return True, tuple(float(v) for v in out["target_bbox"]), score
