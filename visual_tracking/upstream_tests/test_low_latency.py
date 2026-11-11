"""The low-latency order of tracker_node: pre_image() decides and the image is forwarded BEFORE the tracker runs.
The images sent and the decisions must be exactly the same as with the original order (decide after the tracker)."""
from visual_tracking.core import TrackerCore
from visual_tracking.logic import TargetLogic
from test_logic import D, FakeTracker


def run(new_order):
    core = TrackerCore(FakeTracker(), TargetLogic(lost_after_s=0.5), detector_every=5, stamp_tolerance=0.01)
    sent = []
    for i in range(40):
        t = i / 30
        if new_order:
            s = core.pre_image(t)
            res = core.on_image(None, t, sent=s)
            sent.append(res.send)
        else:
            sent.append(core.on_image(None, t).send)
        if i == 2:
            core.on_detections([D(10, 10)], t)               # target found on image 2
        elif sent[-1] and 2 < i < 25:
            core.on_detections([D(10, 10)], t)               # confirmed at every check
        elif sent[-1]:
            core.on_detections([], t)                        # from image 25: the detector sees nothing -> lost
    return sent, dict(core.logic.counts), core.logic.mode


def test_same_images_and_decisions_as_before():
    assert run(True) == run(False)


def test_searching_every_image_tracking_every_nth():
    sent, counts, mode = run(True)
    assert sent[:3] == [True, True, True]
    assert sent[3:13] == [False, False, False, False, True] * 2
    assert counts["confirmed"] > 0 and counts["lost"] == 1 and mode == "search"
    assert all(sent[-3:])                                     # searching again: every image


def test_low_score_still_asks_the_detector_late():
    class LowScore(FakeTracker):
        def update(self, frame):
            ok, box, _ = super().update(frame)
            return ok, box, 0.1
    core = TrackerCore(LowScore(), TargetLogic(), lost_score=0.3, detector_every=5, stamp_tolerance=0.01)
    s = core.pre_image(0.0)
    core.on_image(None, 0.0, sent=s)
    core.on_detections([D(10, 10)], 0.0)
    s = core.pre_image(1 / 30)
    assert s is False                                         # not an N-th image
    assert core.on_image(None, 1 / 30, sent=s).send is True   # but the low score sends it (late)
