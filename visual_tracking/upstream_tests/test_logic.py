"""Tests of the decision logic and of the image/detection matching (no ROS, no model needed):
    python3 -m pytest test/  (from the package folder)"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from visual_tracking.core import TrackerCore  # noqa: E402
from visual_tracking.logic import Detection, TargetLogic  # noqa: E402


def D(x, y, w=20, h=10, s=0.9):
    return Detection((x, y, w, h), s, "car")


def test_first_target_is_the_most_confident():
    lg = TargetLogic()
    dec = lg.search([D(10, 10, s=0.5), D(300, 200, s=0.9)])
    assert dec.event == "init" and dec.init_box == (300, 200, 20, 10) and lg.mode == "track"


def test_confirm_correct_lost_reacquire():
    lg = TargetLogic(lost_after=3)
    lg.search([D(100, 100)])
    assert lg.check((101, 100, 20, 10), [D(100, 100)]).event == "confirm"
    dec = lg.check((130, 100, 20, 10), [D(104, 100)])      # tracker drifted 30 px, detection near: correct
    assert dec.event == "correct" and dec.init_box == (104, 100, 20, 10)
    for k in range(2):
        assert lg.check((104, 100, 20, 10), []).event == "unverified"
    assert lg.check((104, 100, 20, 10), []).event == "lost" and lg.mode == "search"
    dec = lg.search([D(400, 300), D(110, 102)])            # nearest to the last known position
    assert dec.event == "reacquire" and dec.init_box == (110, 102, 20, 10)


def test_switch_to_another_car_is_caught():
    lg = TargetLogic()
    lg.search([D(100, 100)])
    # the tracker jumped onto a car at 200,100 while the target is still detected near 100,100
    dec = lg.check((200, 100, 20, 10), [D(200, 100), D(102, 100)])
    assert dec.event == "correct" and dec.init_box == (102, 100, 20, 10) and lg.counts["switches"] == 1


class FakeTracker:
    """Moves its box 1 px to the right per update."""
    def __init__(self):
        self.box = None
        self.inits = []

    def init(self, frame, box):
        self.box = list(box)
        self.inits.append(tuple(box))

    def update(self, frame):
        self.box[0] += 1
        return True, tuple(self.box), None


def test_late_detections_are_matched_to_their_image_and_the_tracker_catches_up():
    tr = FakeTracker()
    core = TrackerCore(tr, TargetLogic(), max_catch_up=3, stamp_tolerance=0.01)
    for i in range(3):
        core.on_image(None, i * 0.033)
    core.on_detections([D(50, 50)], 0.0)                   # detections of the FIRST image arrive late
    assert core.ready and tr.inits == [(50, 50, 20, 10)]
    assert core.stats["caught_up"] == 2                    # rerun over images 2 and 3
    assert core.buf[-1].box == (52, 50, 20, 10)
    res = core.on_image(None, 3 * 0.033)
    assert res.box == (53, 50, 20, 10) and res.mode == "track"


def test_detections_before_their_image_wait_for_it():
    tr = FakeTracker()
    core = TrackerCore(tr, TargetLogic(), stamp_tolerance=0.01, detector_every=0)  # detector not driven by the node
    core.on_image(None, 0.0)
    assert core.on_detections([D(10, 10)], 0.033) is None  # its image has not arrived yet
    assert len(core.pending) == 1
    core.on_image(None, 0.033)
    assert not core.pending and core.ready and tr.inits == [(10, 10, 20, 10)]


def test_lost_after_seconds_does_not_depend_on_the_detector_rate():
    lg = TargetLogic(lost_after=3, lost_after_s=0.5)
    lg.search([D(100, 100)], stamp=0.0)
    for k in range(1, 15):                                   # 14 empty detection messages in 0.47 s: still tracking
        assert lg.check((100, 100, 20, 10), [], stamp=k / 30).event == "unverified"
    assert lg.check((100, 100, 20, 10), [], stamp=0.5).event == "lost"


def test_after_a_loss_a_far_car_is_not_taken_for_the_target():
    lg = TargetLogic(lost_after_s=0.5, search_growth=200.0)
    lg.search([D(100, 100)], stamp=0.0)
    assert lg.check((100, 100, 20, 10), [D(400, 100)], stamp=0.6).event == "lost"   # target hidden, other car far
    assert lg.search([D(400, 100)], stamp=0.7).event == "search"                       # 300 px away: not taken
    dec = lg.search([D(400, 100), D(115, 100)], stamp=0.9)                              # target back nearby
    assert dec.event == "reacquire" and dec.init_box == (115, 100, 20, 10)
    assert lg.search_limit(0.6 + 2.0) > 300                                             # after 2 s, far is allowed


def test_after_a_loss_a_detection_on_the_still_running_tracker_reacquires_at_once():
    lg = TargetLogic(lost_after_s=0.5, search_growth=200.0)
    lg.search([D(100, 100)], stamp=0.0)
    assert lg.check((100, 100, 20, 10), [], stamp=0.6).event == "lost"
    # the tracker kept following the car to x=160; the detector sees it there again, plus another car nearer to x=100
    dec = lg.search([D(118, 100), D(161, 100)], stamp=0.7, box=(160, 100, 20, 10))
    assert dec.event == "reacquire" and dec.init_box == (161, 100, 20, 10)


def test_while_the_tracker_is_trusted_other_cars_are_ignored():
    lg = TargetLogic(lost_after_s=0.5, search_growth=200.0, trust_tracker_s=2.0)
    lg.search([D(100, 100)], stamp=0.0)
    assert lg.check((100, 100, 20, 10), [D(180, 100)], stamp=0.6).event == "lost"
    # 0.7 s: the tracker is still at 100; a car 80 px away is inside the growing radius, but not near the tracker
    assert lg.search([D(180, 100)], stamp=0.7, box=(101, 100, 20, 10)).event == "search"
    assert lg.search([D(180, 100), D(103, 100)], stamp=0.8, box=(102, 100, 20, 10)).init_box == (103, 100, 20, 10)


def test_detector_every_n_images_while_tracking_and_every_image_while_searching():
    tr = FakeTracker()
    core = TrackerCore(tr, TargetLogic(lost_after_s=0.5), detector_every=5, stamp_tolerance=0.01)
    sent = [core.on_image(None, i / 30).send for i in range(3)]
    assert sent == [True, True, True]                        # searching: every image goes to the detector
    core.on_detections([D(10, 10)], 2 / 30)                   # target found on image 2
    sent = [core.on_image(None, i / 30).send for i in range(3, 14)]
    assert sent == [False, False, False, False, True] * 2 + [False]   # tracking: every 5th image


def test_detections_of_images_not_sent_are_ignored():
    tr = FakeTracker()
    core = TrackerCore(tr, TargetLogic(), detector_every=5, stamp_tolerance=0.01)
    core.on_image(None, 0.0)
    core.on_detections([D(10, 10)], 0.0)                      # tracking from image 0
    core.on_image(None, 1 / 30)                               # not sent (1st of 5)
    assert core.on_detections([D(90, 90)], 1 / 30) is None    # the detector ran on it anyway: ignored
    assert core.stats["ignored_detections"] == 1 and core.logic.counts["checks"] == 0
