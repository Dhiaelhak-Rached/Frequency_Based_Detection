"""
hit_family.py - builds HiT (Tiny / Small / Base) and DyHiT from the HiT repo (github.com/kangben258/HiT, MIT),
using the repo's own test-time classes lib/test/tracker/HiT.py and DyHiT.py. Used by trackers/hit_*/ and
trackers/dyhit/. Config keys per tracker in config.json: <name>_repo, <name>_config, <name>_ckpt, <name>_device.
"""
import copy
import os

from common.torch_tracker import (RepoTracker, add_repo, allow_full_checkpoints, checkpoint_path, import_torch,
                                  make_params, setup_device)

URL = "https://github.com/kangben258/HiT"


def build(cfg, name, dynamic=False):
    """name: config.json prefix (hit_tiny, hit_small, hit_base, dyhit). dynamic=True builds DyHiT."""
    torch = import_torch()
    repo = add_repo(cfg, f"{name}_repo", URL)
    ckpt = checkpoint_path(cfg, f"{name}_ckpt", f"trackers/{name}/README.md")
    device, device_desc = setup_device(torch, cfg.get(f"{name}_device"))
    allow_full_checkpoints(torch)
    # Building the backbone normally downloads LeViT ImageNet weights (pretrained=is_main_process()). The tracker
    # checkpoint replaces every weight right after, so skip that download.
    import lib.models.HiT.backbone as hit_backbone
    hit_backbone.is_main_process = lambda: False
    yaml_name = cfg.get(f"{name}_config")
    if dynamic:
        # build_dyhit() also loads the training-stage weights from cfg.TRAIN.WEIGHT (empty at test time):
        # skip it, the DyHiT checkpoint holds the whole network.
        import lib.models.HiT.hit as hit_model
        hit_model.load_pretrained = lambda model, _cfg: model
        from lib.config.DyHiT.config import cfg as repo_cfg, update_config_from_file
        from lib.test.tracker.DyHiT import DyHiT as TrackerClass
        yaml = os.path.join(repo, "experiments", "DyHiT", yaml_name + ".yaml")
    else:
        from lib.config.HiT.config import cfg as repo_cfg, update_config_from_file
        from lib.test.tracker.HiT import HiT as TrackerClass
        yaml = os.path.join(repo, "experiments", "HiT", yaml_name + ".yaml")
    update_config_from_file(yaml)
    c = copy.deepcopy(repo_cfg)
    extra = dict(yaml_name=yaml_name)
    if dynamic:
        extra.update(threshold=c.TEST.THRESHOLD, interval=c.TEST.INTERVAL)
    params = make_params(c, ckpt, **extra)
    t = TrackerClass(params, "default")
    if not dynamic:
        # The DyHiT commit (cd7b38e) gave forward_backbone() two DyHiT-only arguments but didn't update the HiT test
        # class, which still calls forward_backbone(images). HiT's LeViT backbone ignores them (forward(images,
        # **kwargs)), so passing placeholders restores the original HiT behaviour.
        fb = t.network.forward_backbone
        t.network.forward_backbone = lambda images_list, *a, **k: fb(images_list, None, None)
    # HiT has no confidence output. DyHiT's score only chooses its fast or full route, so it isn't reported either.
    return RepoTracker(name, t, torch, device, device_desc, score_fn=None)


_CACHE = {}


def make_cached(cfg, name, dynamic=False):
    """Build once per process (loading the model is slow), re-initialise per sequence."""
    key = (name, cfg.get(f"{name}_config"), cfg.get(f"{name}_ckpt"), cfg.get(f"{name}_device"))
    if key not in _CACHE:
        _CACHE[key] = build(cfg, name, dynamic)
    return _CACHE[key]
