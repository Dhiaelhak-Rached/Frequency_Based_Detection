import threading
import time
import unittest

import numpy as np

from geometry import RigGeometry
from main import create_vision
from test_vision import FakeModel, FakePrediction
from tracked_camera import LatestRequestQueue, TrackedDroneCamera, TrackingSettings
from vision_detector import DroneCamera
from visual_tracking.core import TrackerCore
from visual_tracking.logic import Detection, TargetLogic


class FakeTracker:
    def __init__(self):
        self.box = None
        self.inits = []
        self.updates = 0

    def init(self, frame, box):
        self.box = list(box)
        self.inits.append(tuple(box))

    def update(self, frame):
        self.box[0] += 1.0
        self.updates += 1
        return True, tuple(self.box), None


class StreamingCapture:
    def __init__(self):
        self.opened = True
        self.released = False

    def isOpened(self):
        return self.opened

    def read(self):
        time.sleep(0.005)
        return self.opened, np.zeros((120, 160, 3), dtype=np.uint8)

    def release(self):
        self.released = True
        self.opened = False


class BlockingModel(FakeModel):
    def __init__(self, rows):
        super().__init__(rows)
        self.started = threading.Event()
        self.release = threading.Event()

    def predict(self, frame, conf, verbose):
        self.calls += 1
        if self.calls > 1:
            self.started.set()
            if not self.release.wait(2.0):
                raise TimeoutError("Test did not release YOLO")
        return [FakePrediction(self.rows)]


def wait_until(predicate, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("Condition was not met before timeout")


class VisionTrackingTest(unittest.TestCase):
    def make_camera(self, model=None, tracker=None, capture=None, **tracker_config):
        config = {"detector_every": 1, "lost_after_s": 0.1}
        config.update(tracker_config)
        return TrackedDroneCamera(
            model_path="unused.pt",
            model=model or FakeModel([(50, 40, 80, 80, 0.87, 0)]),
            tracker=tracker or FakeTracker(),
            capture_factory=lambda source: capture or StreamingCapture(),
            inference_hz=100.0,
            horizontal_fov_deg=90.0,
            tracker_config=config,
        )

    def test_latest_request_queue_never_backlogs(self):
        requests = LatestRequestQueue()
        requests.put(np.zeros(1), 1.0)
        requests.put(np.zeros(2), 2.0)
        self.assertEqual(requests.size, 1)
        self.assertEqual(requests.dropped, 1)
        self.assertEqual(requests.get()[1], 2.0)

    def test_yolo_initializes_tracker_and_tracked_box_has_direction(self):
        tracker = FakeTracker()
        capture = StreamingCapture()
        camera = self.make_camera(tracker=tracker, capture=capture)
        camera.start()
        wait_until(lambda: camera.get_result().target is not None)
        wait_until(lambda: tracker.updates >= 2)
        result = camera.get_result()
        self.assertEqual(result.output_source, "tracker")
        self.assertEqual(result.tracking_state, "track")
        self.assertAlmostEqual(result.last_yolo_confidence, 0.87)
        self.assertIsNotNone(result.direction)
        self.assertGreater(result.direction.global_bearing_deg, 0.0)
        self.assertEqual(tracker.inits[0], (50.0, 40.0, 30.0, 40.0))
        camera.stop()
        self.assertTrue(capture.released)

    def test_loss_clears_target_and_direction(self):
        model = FakeModel([(50, 40, 80, 80, 0.87, 0)])
        camera = self.make_camera(model=model)
        camera.start()
        wait_until(lambda: camera.get_result().target is not None)
        model.rows = []
        wait_until(
            lambda: camera.get_result().target is None
            and camera.get_result().tracking_state == "search"
            and camera.get_result().tracking_event == "lost"
        )
        result = camera.get_result()
        self.assertIsNone(result.direction)
        camera.stop()

    def test_slow_yolo_does_not_block_tracker_updates(self):
        model = BlockingModel([(50, 40, 80, 80, 0.87, 0)])
        tracker = FakeTracker()
        camera = self.make_camera(model=model, tracker=tracker, detector_every=2)
        camera.start()
        wait_until(lambda: model.started.is_set())
        before = tracker.updates
        time.sleep(0.06)
        self.assertGreater(tracker.updates, before)
        self.assertLessEqual(camera._detection_requests.size, 1)
        model.release.set()
        camera.stop()

    def test_tracker_load_failure_is_visible(self):
        camera = self.make_camera(library_path="/missing/tracker/library")
        camera._tracker = None
        camera.start()
        wait_until(lambda: camera.get_result().error is not None)
        self.assertTrue(camera.get_result().error.startswith("Tracker error:"))
        camera.stop()

    def test_skipped_catch_up_does_not_publish_stale_box(self):
        tracker = FakeTracker()
        core = TrackerCore(tracker, TargetLogic(), max_catch_up=0, stamp_tolerance=0.01)
        first = np.zeros((120, 160, 3), dtype=np.uint8)
        second = np.zeros((120, 160, 3), dtype=np.uint8)
        core.on_image(first, 0.0)
        core.on_image(second, 0.033)
        decision = core.on_detections([Detection((50, 40, 30, 40), 0.9, "drone")], 0.0)
        self.assertEqual(decision.event, "init")
        self.assertEqual(core.stats["catch_up_skipped"], 1)
        self.assertIsNone(core.buf[-1].box)

    def test_explicit_yolo_only_fallback(self):
        geo = RigGeometry()
        tracked = create_vision(geo, camera=0)
        fallback = create_vision(geo, camera=0, no_tracker=True)
        self.assertIsInstance(tracked, TrackedDroneCamera)
        self.assertIsInstance(fallback, DroneCamera)
        self.assertNotIsInstance(fallback, TrackedDroneCamera)

    def test_tracking_settings_validation(self):
        with self.assertRaises(ValueError):
            TrackingSettings(detector_every=0)
        with self.assertRaises(ValueError):
            TrackingSettings(name="unknown")
        with self.assertRaises(ValueError):
            TrackingSettings(max_catch_up=3, buffer_size=4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
