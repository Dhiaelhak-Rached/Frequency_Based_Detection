import time
import unittest

import numpy as np

from vision_detector import (
    DetectionLock,
    DroneCamera,
    VisionDetection,
    normalize_camera_source,
    pixel_to_direction,
    validate_horizontal_fov,
)


class FakeTensor:
    def __init__(self, values):
        self.values = np.asarray(values)

    def cpu(self):
        return self

    def numpy(self):
        return self.values


class FakeBoxes:
    def __init__(self, rows):
        rows = list(rows)
        self.xyxy = FakeTensor([row[:4] for row in rows])
        self.conf = FakeTensor([row[4] for row in rows])
        self.cls = FakeTensor([row[5] for row in rows])


class FakePrediction:
    def __init__(self, rows):
        self.boxes = FakeBoxes(rows)


class FakeModel:
    names = {0: "drone", 1: "other"}

    def __init__(self, rows=()):
        self.rows = rows
        self.calls = 0

    def predict(self, frame, conf, verbose):
        self.calls += 1
        return [FakePrediction(self.rows)]


class FakeCapture:
    def __init__(self, opened=True):
        self.opened = opened
        self.released = False
        self.frames = 0

    def isOpened(self):
        return self.opened

    def read(self):
        self.frames += 1
        return True, np.full((48, 64, 3), self.frames % 255, dtype=np.uint8)

    def release(self):
        self.released = True


def box(x, y, score=0.8):
    return VisionDetection(
        bbox=(int(x - 10), int(y - 10), int(x + 10), int(y + 10)),
        confidence=score,
        class_id=0,
        label="drone",
    )


class VisionTest(unittest.TestCase):
    def make_camera(self, model=None, capture_factory=None, horizontal_fov=None):
        return DroneCamera(
            model_path="unused.pt",
            model=model or FakeModel(),
            capture_factory=capture_factory,
            inference_hz=50.0,
            horizontal_fov_deg=horizontal_fov,
        )

    def wait_for_frame(self, camera):
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            if camera.get_result().timestamp > 0.0:
                return
            time.sleep(0.01)
        self.fail(f"Camera did not produce a frame: {camera.get_result()}")

    def test_camera_source_normalization(self):
        self.assertEqual(normalize_camera_source(" 2 "), 2)
        self.assertEqual(normalize_camera_source("/dev/v4l/by-id/camera"), "/dev/v4l/by-id/camera")

    def test_drone_filtering_and_annotation(self):
        model = FakeModel([
            (10, 10, 40, 40, 0.87, 0),
            (50, 10, 70, 40, 0.99, 1),
        ])
        camera = self.make_camera(model)
        frame = np.zeros((80, 100, 3), dtype=np.uint8)
        result = camera.process_frame(frame)

        self.assertEqual(len(result.detections), 1)
        detection = result.detections[0]
        self.assertEqual(detection.label, "drone")
        self.assertEqual(detection.bbox, (10, 10, 40, 40))
        self.assertAlmostEqual(result.best_confidence, 0.87)
        self.assertTrue(np.any(result.frame != frame))

    def test_missing_drone_class_is_rejected(self):
        model = FakeModel()
        model.names = {0: "person"}
        camera = self.make_camera(model)
        with self.assertRaisesRegex(ValueError, "no 'drone' class"):
            camera.process_frame(np.zeros((32, 32, 3), dtype=np.uint8))

    def test_worker_publishes_latest_frame_and_stops(self):
        capture = FakeCapture()
        camera = self.make_camera(capture_factory=lambda source: capture)
        camera.start()
        self.wait_for_frame(camera)
        result = camera.get_result()
        self.assertIsNone(result.error)
        self.assertEqual(result.frame.shape, (48, 64, 3))
        camera.stop()
        self.assertTrue(capture.released)
        self.assertFalse(camera.is_running)

    def test_unavailable_camera_is_retried(self):
        captures = [FakeCapture(opened=False), FakeCapture(opened=True)]
        camera = self.make_camera(capture_factory=lambda source: captures.pop(0))
        camera.start()
        self.wait_for_frame(camera)
        self.assertIsNone(camera.get_result().error)
        self.assertFalse(captures)
        camera.stop()

    def test_fov_is_required_and_validated(self):
        self.assertIsNone(validate_horizontal_fov(None))
        self.assertIsNone(validate_horizontal_fov(""))
        self.assertEqual(validate_horizontal_fov("72.5"), 72.5)
        for invalid in (0, 180, -20, float("nan"), "invalid"):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    validate_horizontal_fov(invalid)

    def test_center_pixel_has_zero_direction(self):
        direction = pixel_to_direction(319.5, 239.5, 640, 480, 90)
        self.assertAlmostEqual(direction.local_bearing_deg, 0.0)
        self.assertAlmostEqual(direction.global_bearing_deg, 0.0)
        self.assertAlmostEqual(direction.elevation_deg, 0.0)
        self.assertAlmostEqual(np.linalg.norm(direction.ray), 1.0)

    def test_left_right_elevation_and_mount_rotation(self):
        left = pixel_to_direction(0, 239.5, 640, 480, 90, orientation_deg=350)
        right = pixel_to_direction(639, 239.5, 640, 480, 90)
        above = pixel_to_direction(319.5, 0, 640, 480, 90)
        self.assertLess(left.local_bearing_deg, 0)
        self.assertGreater(right.local_bearing_deg, 0)
        self.assertAlmostEqual(left.global_bearing_deg, 350 + left.local_bearing_deg)
        self.assertGreater(above.elevation_deg, 0)

    def test_fov_and_image_size_change_direction(self):
        narrow = pixel_to_direction(480, 239.5, 640, 480, 60)
        wide = pixel_to_direction(480, 239.5, 640, 480, 120)
        doubled = pixel_to_direction(960.5, 479.5, 1280, 960, 60)
        self.assertLess(abs(narrow.local_bearing_deg), abs(wide.local_bearing_deg))
        self.assertAlmostEqual(narrow.local_bearing_deg, doubled.local_bearing_deg)

    def test_detection_lock_confirms_and_rejects_far_jump(self):
        lock = DetectionLock()
        self.assertIsNone(lock.select([box(100, 200)], 0.0))
        self.assertIsNone(lock.select([box(104, 201)], 0.067))
        self.assertIsNotNone(lock.select([box(108, 202)], 0.134))
        self.assertIsNone(lock.select([box(620, 250)], 0.201))
        self.assertIsNotNone(lock.select([box(110, 203)], 0.268))

    def test_detection_lock_reacquires_after_gap(self):
        lock = DetectionLock()
        for stamp in (0.0, 0.067, 0.134):
            selected = lock.select([box(100, 200)], stamp)
        self.assertIsNotNone(selected)
        self.assertIsNone(lock.select([], 0.201))
        self.assertIsNone(lock.select([box(600, 300)], 1.0))
        self.assertIsNone(lock.select([box(602, 301)], 1.067))
        self.assertIsNotNone(lock.select([box(604, 302)], 1.134))

    def test_confirmed_detection_without_fov_has_no_direction(self):
        model = FakeModel([(300, 220, 340, 260, 0.9, 0)])
        camera = self.make_camera(model)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        results = [camera.process_frame(frame) for _ in range(3)]
        self.assertIsNotNone(results[-1].target)
        self.assertIsNone(results[-1].direction)

    def test_confirmed_detection_with_fov_has_direction(self):
        model = FakeModel([(440, 220, 480, 260, 0.9, 0)])
        camera = self.make_camera(model, horizontal_fov=90)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        results = [camera.process_frame(frame) for _ in range(3)]
        self.assertIsNotNone(results[-1].target)
        self.assertIsNotNone(results[-1].direction)
        self.assertGreater(results[-1].direction.global_bearing_deg, 20.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
