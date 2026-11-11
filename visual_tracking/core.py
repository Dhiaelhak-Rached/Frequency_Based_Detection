"""
core.py - the tracking loop without ROS: images go in (on_image), detections go in (on_detections), the target's box
comes out. tracker_node.py only converts ROS messages to and from these calls, so this file can be tested on any PC.

Images and detections arrive separately and the detections come a little later (the detector needs time). Every
image is kept for a moment in a short buffer with the tracker's box on it. When the detections for an image arrive
(same timestamp), they are compared with the tracker's box ON THAT IMAGE, not on the newest one. If the tracker must
be restarted, it is restarted on that image and then run again over the newer buffered images to catch up
(at most max_catch_up of them; if it is further behind, the next new image simply continues from the restart).
Detections that arrive before their image (the node was busy) wait in `pending` until the image comes.

The detector runs every `detector_every` images: on_image says which images to send to it (FrameResult.send):
every N-th image while tracking, every image while searching (to find the target again quickly), and at once when
the tracker's score drops (check_on_low_score). Detections of images that were not sent are ignored, so the result
is the same whether the detector reads the node's images or still reads every camera image. detector_every = 0:
the detector is not driven by the node and every detection message is used.

Low latency (DALBD): the node asks pre_image() BEFORE running the tracker and forwards the image to the detector at
once, so the detector no longer waits for the tracker (~20 ms less YOLO latency). Only the low-score check
(check_on_low_score, SUTrack) still has to wait for the tracker's score; on_image() reports it as a late send.
"""
import os
import sys
import time
from collections import deque, namedtuple

FrameResult = namedtuple("FrameResult", "stamp box score mode trk_ms weak send")


def load_tracker(lib, name, device=None, checkpoint=None):
    """Import the trackers' code from tracker_lib/ (common/, trackers/, external/, config.json) and build `name`."""
    lib = os.path.abspath(os.path.expanduser(lib))
    if not os.path.isfile(os.path.join(lib, "common", "runners.py")):
        raise RuntimeError(f"tracker_lib '{lib}' does not contain common/runners.py: set the tracker_lib parameter "
                           "to the drone_target_tracker/tracker_lib folder of the workspace source")
    if lib not in sys.path:
        sys.path.insert(0, lib)
    from common.bench_core import load_config  # noqa: E402 (from tracker_lib)
    from common.runners import TRACKERS, load_tracker as make  # noqa: E402
    if name not in TRACKERS:
        raise RuntimeError(f"unknown tracker '{name}' (available: {', '.join(TRACKERS)})")
    cfg = load_config()
    if device is not None:
        cfg[f"{name}_device"] = device
    if checkpoint is not None:
        cfg[f"{name}_ckpt"] = checkpoint
    return make(name, cfg), cfg.get("lost_score", {}).get(name)


class _Entry:
    __slots__ = ("stamp", "frame", "box", "score")

    def __init__(self, stamp, frame, box, score):
        self.stamp, self.frame, self.box, self.score = stamp, frame, box, score


class TrackerCore:
    """tracker: object with init(frame, box) and update(frame) -> (ok, box, score); frames are BGR uint8 arrays.
    logic: a TargetLogic. Stamps are in seconds (float)."""

    def __init__(self, tracker, logic, lost_score=None, max_catch_up=3, stamp_tolerance=0.02, buffer_size=60,
                 detector_every=5, check_on_low_score=True):
        self.tracker = tracker
        self.logic = logic
        self.lost_score = lost_score
        self.max_catch_up = int(max_catch_up)
        self.stamp_tolerance = float(stamp_tolerance)
        self.buf = deque(maxlen=int(buffer_size))
        self.pending = deque(maxlen=10)  # (stamp, dets) whose image has not arrived yet
        self.ready = False  # the tracker has been initialised (it keeps running while searching, unseen)
        self.last_decision = None  # (stamp, Decision) of the latest detection message used
        self.stats = dict(images=0, tracked=0, detection_msgs=0, stale_detections=0, restarts=0, caught_up=0,
                          catch_up_skipped=0)
        self.detector_every = int(detector_every)
        self.check_on_low_score = bool(check_on_low_score)
        self.since_sent = 0
        self.sent = deque(maxlen=300)  # stamps of the images sent to the detector
        self.stats.update(sent_to_detector=0, ignored_detections=0)
        self.trk_ms = deque(maxlen=2000)
        self.restart_ms = deque(maxlen=500)

    # ------------------------------------------------------------------ images
    def pre_image(self, stamp):
        """Decide, BEFORE the tracker runs on this image, whether it goes to the detector (every image while searching,
        every detector_every-th while tracking). True: forward it now, then call on_image(frame, stamp, sent=True)."""
        if self.detector_every <= 0:
            return False
        self.since_sent += 1
        send = self.logic.mode != "track" or self.since_sent >= self.detector_every
        if send:
            self._mark_sent(stamp)
        return send

    def _mark_sent(self, stamp):
        self.since_sent = 0
        self.sent.append(stamp)
        self.stats["sent_to_detector"] += 1

    def on_image(self, frame, stamp, sent=None):
        """Run the tracker on a new image (if it has a target). Returns a FrameResult for this image.
        sent: what pre_image() returned for this image (None = decide here, after the tracker, as before)."""
        self.stats["images"] += 1
        box = score = trk_ms = None
        if self.ready:
            t0 = time.perf_counter()
            ok, b, score = self.tracker.update(frame)
            trk_ms = (time.perf_counter() - t0) * 1000
            self.trk_ms.append(trk_ms)
            self.stats["tracked"] += 1
            box = tuple(float(v) for v in b) if ok else None
        entry = _Entry(stamp, frame, box, score)
        self.buf.append(entry)
        for p in list(self.pending):  # detections that were waiting for this image
            if abs(p[0] - stamp) <= self.stamp_tolerance:
                self.pending.remove(p)
                self._apply(entry, p[1], p[0])
            elif p[0] < stamp - self.stamp_tolerance:  # their image never came (dropped)
                self.pending.remove(p)
                self.stats["stale_detections"] += 1
        tracking = self.logic.mode == "track"
        box = entry.box if tracking else None  # while searching the tracker's box is not an output
        weak = tracking and (box is None or (entry.score is not None and self.lost_score is not None
                                             and entry.score < self.lost_score))
        send = False
        if self.detector_every > 0 and sent is None:  # decided after the tracker (original behaviour)
            self.since_sent += 1
            send = (not tracking or self.since_sent >= self.detector_every or (weak and self.check_on_low_score))
            if send:
                self._mark_sent(stamp)
        elif self.detector_every > 0:  # already decided by pre_image(); only a low score can still ask the detector
            send = bool(sent)
            if not send and weak and self.check_on_low_score:
                self._mark_sent(stamp)
                send = True
        return FrameResult(stamp, box, entry.score, self.logic.mode, trk_ms, weak, send)

    # ------------------------------------------------------------------ detections
    def on_detections(self, dets, stamp):
        """Detections (list of logic.Detection) computed on the image with this stamp. Returns the logic's Decision,
        or None if that image is too old (no longer buffered) or has not arrived yet (then it is applied later)."""
        self.stats["detection_msgs"] += 1
        if self.detector_every > 0 and not any(abs(s - stamp) <= self.stamp_tolerance for s in self.sent):
            self.stats["ignored_detections"] += 1  # an image the node did not ask about
            return None
        entry = self._find(stamp)
        if entry is None:
            if not self.buf or stamp > self.buf[-1].stamp:
                self.pending.append((stamp, dets))
            else:
                self.stats["stale_detections"] += 1
            return None
        return self._apply(entry, dets, stamp)

    def _apply(self, entry, dets, stamp):
        if self.logic.mode == "search":
            dec = self.logic.search(dets, stamp, entry.box if self.ready else None)
        else:
            dec = self.logic.check(entry.box, dets, stamp)
        if dec.init_box is not None:
            self._restart(entry, dec.init_box)
        self.last_decision = (stamp, dec)
        return dec

    def _find(self, stamp):
        best, best_dt = None, None
        for e in reversed(self.buf):
            dt = abs(e.stamp - stamp)
            if best_dt is None or dt < best_dt:
                best, best_dt = e, dt
            if e.stamp < stamp - self.stamp_tolerance:
                break
        return best if best_dt is not None and best_dt <= self.stamp_tolerance else None

    def _restart(self, entry, box):
        t0 = time.perf_counter()
        self.tracker.init(entry.frame, box)
        entry.box, entry.score = tuple(box), None
        self.ready = True
        self.stats["restarts"] += 1
        later = [e for e in self.buf if e.stamp > entry.stamp]
        if len(later) <= self.max_catch_up:
            for e in later:
                ok, b, s = self.tracker.update(e.frame)
                e.box, e.score = (tuple(float(v) for v in b) if ok else None), s
                self.stats["caught_up"] += 1
        else:
            self.stats["catch_up_skipped"] += 1
            for e in later:  # their boxes came from the old track: no longer valid
                e.box, e.score = None, None
            self.buf[-1].box = None  # the next image continues from the restart position without a stale output
        self.restart_ms.append((time.perf_counter() - t0) * 1000)

    # ------------------------------------------------------------------ outputs
    def summary(self):
        ms = sorted(self.trk_ms)
        out = dict(self.stats)
        out.update(self.logic.counts)
        out["mode"] = self.logic.mode
        if ms:
            out["trk_ms_mean"] = round(sum(ms) / len(ms), 2)
            out["trk_ms_p95"] = round(ms[min(len(ms) - 1, int(0.95 * len(ms)))], 2)
        if self.restart_ms:
            out["restart_ms_mean"] = round(sum(self.restart_ms) / len(self.restart_ms), 2)
        return out
