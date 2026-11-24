"""
logic.py - the decisions of the detector + tracker loop, without ROS and without any model (pure Python, testable on
any PC). Ported from pipeline.py of the test_detector_tracker_car test bench, so both behave the same.

Two states:
  SEARCH  no target: every detection message is searched for one. The first time, the target is chosen by
          `initial_target` (best / largest / center). After a loss the tracker keeps running unseen. For
          trust_tracker_s seconds after the target was last confirmed, only a detection near the tracker's box
          re-acquires it (the detector had missed the target, the tracker probably had not): other cars are ignored.
          After that, the detection nearest to the last confirmed position, within a radius that grows with the time
          since then (reacquire_gate x target size + search_growth px per second, capped by search_radius).
  TRACK   the tracker follows the target on every image; each detection message CHECKS the tracker's box:
            confirm     a detection overlaps the box (IoU >= verify_iou) and it is the target
            correct     no confirming detection, but one is near the target's last confirmed position (or near the
                        tracker's box while that box is itself near it): restart the tracker on it
            switch      the box sits on a detection, but far from the target's last position, or another detection
                        is much nearer to it: the tracker jumped to another object. Corrected back if possible.
            unverified  no detection near: keep the tracker's box; lost when the target has not been confirmed
                        or corrected for lost_after_s seconds (or, without stamps, after lost_after such checks
                        in a row) -> SEARCH

Boxes are (x, y, w, h) in pixels, top-left origin.
"""
import math
from collections import namedtuple

Detection = namedtuple("Detection", "box score label")
# event: what happened; init_box: restart (or start) the tracker on this box, else None
Decision = namedtuple("Decision", "event init_box")


def iou(a, b):
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw = max(0.0, min(ax2, bx2) - max(a[0], b[0]))
    ih = max(0.0, min(ay2, by2) - max(a[1], b[1]))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def dist(a, b):
    """Distance between box centres, in pixels."""
    return math.hypot(a[0] + a[2] / 2 - b[0] - b[2] / 2, a[1] + a[3] / 2 - b[1] - b[3] / 2)


class TargetLogic:
    COUNTERS = ("checks", "confirmed", "corrected", "switches", "unverified", "lost", "reacquired", "searched")

    def __init__(self, verify_iou=0.3, reacquire_gate=1.5, lost_after=3, search_radius=0.0, initial_target="best",
                 image_size=(640, 512), lost_after_s=None, search_growth=200.0, trust_tracker_s=2.0):
        if initial_target not in ("best", "largest", "center"):
            raise ValueError("initial_target must be best, largest or center")
        self.verify_iou = float(verify_iou)
        self.gate = float(reacquire_gate)
        self.lost_after = int(lost_after)
        self.lost_after_s = None if lost_after_s is None or lost_after_s <= 0 else float(lost_after_s)
        self.last_ok = None  # stamp of the last init / confirm / correct / reacquire
        self.search_growth = float(search_growth)
        self.trust_tracker_s = float(trust_tracker_s)
        self.lost_at = None  # stamp of the loss
        self.search_radius = float(search_radius)
        self.initial_target = initial_target
        self.image_size = image_size
        self.mode = "search"
        self.last_conf = None  # the target's last confirmed box (None until the first target is chosen)
        self.misses = 0
        self.counts = {k: 0 for k in self.COUNTERS}

    # ------------------------------------------------------------------ SEARCH
    def search(self, dets, stamp=None, box=None):
        """No target: try to (re)acquire one among `dets` (detections of the image taken at `stamp`; `box` = where the
        still-running tracker is on that image, or None). Returns a Decision ('init', 'reacquire' or 'search')."""
        self.counts["searched"] += 1
        if not dets:
            return Decision("search", None)
        if self.last_conf is None:  # very first target
            d = self._choose(dets)
            event = "init"
        else:
            limit = self.search_limit(stamp)
            size = max(self.last_conf[2], self.last_conf[3])
            recent = (stamp is None or self.last_ok is None or stamp - self.last_ok <= self.trust_tracker_s)
            if box is not None and recent and (limit is None or dist(box, self.last_conf) <= limit):
                # the still-running tracker is plausibly on the target: only a detection near its box counts
                cand = [(dist(d.box, box), d) for d in dets]
                cand = [(dd, d) for dd, d in cand if dd <= self.gate * size or iou(box, d.box) >= self.verify_iou]
            else:
                cand = [(dist(d.box, self.last_conf), d) for d in dets]
                if limit is not None:
                    cand = [(dd, d) for dd, d in cand if dd <= limit]
            if not cand:
                return Decision("search", None)
            d = min(cand, key=lambda t: t[0])[1]
            event = "reacquire"
            self.counts["reacquired"] += 1
        self.mode, self.last_conf, self.misses, self.last_ok = "track", tuple(d.box), 0, stamp
        return Decision(event, tuple(d.box))

    def search_limit(self, stamp):
        """Largest distance (px) from the last known position at which a detection is taken as the target again,
        or None for no limit."""
        limits = []
        since = self.last_ok if self.last_ok is not None else self.lost_at  # the target can move before the loss too
        if self.search_growth > 0 and stamp is not None and since is not None:
            size = max(self.last_conf[2], self.last_conf[3])
            limits.append(self.gate * size + self.search_growth * max(0.0, stamp - since))
        if self.search_radius > 0:
            limits.append(self.search_radius)
        return min(limits) if limits else None

    def _choose(self, dets):
        if self.initial_target == "largest":
            return max(dets, key=lambda d: d.box[2] * d.box[3])
        if self.initial_target == "center":
            w, h = self.image_size
            mid = (w / 2, h / 2, 0, 0)
            return min(dets, key=lambda d: dist(d.box, mid))
        return max(dets, key=lambda d: d.score)

    # ------------------------------------------------------------------ TRACK
    def check(self, box, dets, stamp=None):
        """Check the tracker's `box` (None if the tracker failed) against the detections of the SAME image (taken at
        `stamp`, seconds). Returns a Decision: 'confirm', 'correct' (init_box set), 'switch', 'unverified' or 'lost'."""
        if self.mode != "track":
            raise RuntimeError("check() is only for TRACK mode; use search()")
        self.counts["checks"] += 1
        last = self.last_conf
        size = max(last[2], last[3])
        best = max(dets, key=lambda d: iou(box, d.box), default=None) if box else None
        best_iou = iou(box, best.box) if best else 0.0
        # identity: the confirming detection must not be far from the target's last confirmed position, and no other
        # detection may be much nearer to it (otherwise the tracker jumped to another object)
        nearest = min(dets, key=lambda d: dist(d.box, last), default=None)
        switched = (best is not None and best_iou >= self.verify_iou
                    and (dist(best.box, last) > self.gate * size
                         or (nearest is not best and dist(nearest.box, last) < 0.5 * dist(best.box, last))))
        if switched:
            self.counts["switches"] += 1
        if best is not None and best_iou >= self.verify_iou and not switched:
            self.counts["confirmed"] += 1
            self.last_conf, self.misses, self.last_ok = tuple(best.box), 0, stamp
            return Decision("confirm", None)
        # near the last confirmed position, or near the tracker's box while that box is itself still near it
        box_ok = box is not None and not switched and dist(box, last) <= self.gate * size
        refs = [last] + ([box] if box_ok else [])
        near = [(min(dist(d.box, r) for r in refs), d) for d in dets]
        near = [(dd, d) for dd, d in near if dd <= self.gate * size]
        if near:
            d = min(near, key=lambda t: t[0])[1]
            self.counts["corrected"] += 1
            self.last_conf, self.misses, self.last_ok = tuple(d.box), 0, stamp
            return Decision("correct", tuple(d.box))
        self.misses += 1
        self.counts["unverified"] += 1
        if self.lost_after_s is not None and stamp is not None and self.last_ok is not None:
            lost = stamp - self.last_ok >= self.lost_after_s
        else:
            lost = self.misses >= self.lost_after
        if lost:
            self.counts["lost"] += 1
            self.mode, self.lost_at = "search", stamp
            return Decision("lost", None)
        return Decision("switch" if switched else "unverified", None)
