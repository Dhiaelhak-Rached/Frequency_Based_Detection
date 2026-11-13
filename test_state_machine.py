import unittest

import numpy as np

from classifier import ClassificationResult
from detector import DetectionResult
from doa_engine import DOAResult
from target_state_machine import (
    StateMachineConfig,
    TargetStateMachine,
    TrackingMode,
    bearing_elevation_to_vector,
    format_fused_state,
)
from triangulation import TriangulationResult
from vision_detector import CameraDirection, VisionDetection, VisionResult


def _make_vision(bearing=45.2, elevation=12.4, calibrated=True, error=None):
    target = VisionDetection(bbox=(10, 10, 50, 50), confidence=0.88, class_id=0, label="drone")
    direction = (
        CameraDirection(
            local_bearing_deg=bearing,
            global_bearing_deg=bearing,
            elevation_deg=elevation,
            ray=(0.0, 0.0, 1.0),
        )
        if calibrated
        else None
    )
    return VisionResult(
        target=target,
        direction=direction,
        tracking_state="track",
        error=error,
    )


def _make_detection(detected=True):
    return DetectionResult(
        detected=detected,
        peak_freq=1000.0 if detected else None,
        peak_db=-45.0 if detected else -90.0,
        threshold_db=-65.0,
        band=(200.0, 5000.0),
        freqs=np.array([1000.0]),
        mags=np.array([-45.0]),
    )


def _make_doa(bearing=120.0, elevation=15.0, prominence=3.0, error=None):
    vec = np.array(bearing_elevation_to_vector(bearing, elevation)) if elevation is not None else None
    return DOAResult(
        array_id="left",
        local_angle_deg=bearing,
        global_angle_deg=bearing,
        prominence=prominence,
        confidence=0.9,
        spectrum=None,
        error=error,
        stable=True,
        elevation_deg=elevation,
        elevation_prominence=2.5,
        elevation_confidence=0.9,
        elevation_stable=elevation is not None,
        direction_vector=vec,
    )


class TestTargetStateMachine(unittest.TestCase):
    def test_starts_in_searching_when_no_sensors_have_target(self):
        sm = TargetStateMachine()
        state = sm.from_pipeline_outputs(now=1.0)
        self.assertEqual(state.mode, TrackingMode.SEARCHING)
        self.assertEqual(state.active_source, "NONE")
        self.assertIsNone(state.bearing_deg)
        self.assertIsNone(state.direction_vector)
        self.assertFalse(state.vision_uncalibrated)

    def test_falls_back_to_sound_when_vision_absent(self):
        sm = TargetStateMachine()
        state = sm.from_pipeline_outputs(
            now=1.0,
            vision=VisionResult(tracking_state="search"),
            detection=_make_detection(True),
            single_doa=_make_doa(bearing=210.5, elevation=-5.0),
        )
        self.assertEqual(state.mode, TrackingMode.ACOUSTIC_TRACKING)
        self.assertEqual(state.active_source, "SOUND")
        self.assertAlmostEqual(state.bearing_deg, 210.5)
        self.assertAlmostEqual(state.elevation_deg, -5.0)
        self.assertIsNotNone(state.direction_vector)

    def test_vision_takes_priority_over_sound_and_keeps_tracking(self):
        sm = TargetStateMachine()
        det = _make_detection(True)
        doa = _make_doa(bearing=200.0, elevation=0.0)

        s1 = sm.from_pipeline_outputs(now=1.0, detection=det, single_doa=doa)
        self.assertEqual(s1.mode, TrackingMode.ACOUSTIC_TRACKING)

        # Vision sees drone -> immediately switches to VISION_TRACKING
        s2 = sm.from_pipeline_outputs(
            now=1.1,
            vision=_make_vision(bearing=45.2, elevation=12.4),
            detection=det,
            single_doa=doa,
        )
        self.assertEqual(s2.mode, TrackingMode.VISION_TRACKING)
        self.assertEqual(s2.active_source, "VISION")
        self.assertAlmostEqual(s2.bearing_deg, 45.2)
        self.assertAlmostEqual(s2.elevation_deg, 12.4)

        # Sound drops out -> vision keeps tracking
        s3 = sm.from_pipeline_outputs(
            now=1.2,
            vision=_make_vision(bearing=47.0, elevation=13.0),
            detection=_make_detection(False),
        )
        self.assertEqual(s3.mode, TrackingMode.VISION_TRACKING)
        self.assertEqual(s3.active_source, "VISION")
        self.assertAlmostEqual(s3.bearing_deg, 47.0)

    def test_vision_loss_falls_back_immediately_when_hold_is_zero(self):
        sm = TargetStateMachine(StateMachineConfig(vision_hold_s=0.0))
        det = _make_detection(True)
        doa = _make_doa(bearing=180.0, elevation=8.0)

        sm.from_pipeline_outputs(now=1.0, vision=_make_vision(bearing=30.0, elevation=5.0), detection=det, single_doa=doa)
        s2 = sm.from_pipeline_outputs(now=1.1, vision=VisionResult(tracking_state="search"), detection=det, single_doa=doa)
        self.assertEqual(s2.mode, TrackingMode.ACOUSTIC_TRACKING)
        self.assertEqual(s2.active_source, "SOUND")
        self.assertAlmostEqual(s2.bearing_deg, 180.0)

    def test_optional_vision_hold_and_reacquire(self):
        sm = TargetStateMachine(StateMachineConfig(vision_hold_s=0.5))
        det = _make_detection(True)
        doa = _make_doa(bearing=180.0, elevation=8.0)

        sm.from_pipeline_outputs(now=1.0, vision=_make_vision(bearing=30.0, elevation=5.0), detection=det, single_doa=doa)
        held = sm.from_pipeline_outputs(now=1.2, vision=VisionResult(tracking_state="search"), detection=det, single_doa=doa)
        self.assertEqual(held.mode, TrackingMode.VISION_HOLD)
        self.assertEqual(held.active_source, "VISION")
        self.assertAlmostEqual(held.bearing_deg, 30.0)

        # Reacquired before timeout
        reacq = sm.from_pipeline_outputs(now=1.4, vision=_make_vision(bearing=32.0, elevation=6.0), detection=det, single_doa=doa)
        self.assertEqual(reacq.mode, TrackingMode.VISION_TRACKING)

        # Expired hold falls back to sound
        sm.from_pipeline_outputs(now=1.5, vision=VisionResult(tracking_state="search"), detection=det, single_doa=doa)
        expired = sm.from_pipeline_outputs(now=2.1, vision=VisionResult(tracking_state="search"), detection=det, single_doa=doa)
        self.assertEqual(expired.mode, TrackingMode.ACOUSTIC_TRACKING)
        self.assertEqual(expired.active_source, "SOUND")

    def test_uncalibrated_camera_does_not_enter_vision_tracking(self):
        sm = TargetStateMachine()
        uncal_vision = _make_vision(calibrated=False)

        # With valid sound -> stays in ACOUSTIC_TRACKING and exposes vision_uncalibrated=True
        s1 = sm.from_pipeline_outputs(
            now=1.0,
            vision=uncal_vision,
            detection=_make_detection(True),
            single_doa=_make_doa(bearing=90.0, elevation=10.0),
        )
        self.assertEqual(s1.mode, TrackingMode.ACOUSTIC_TRACKING)
        self.assertEqual(s1.active_source, "SOUND")
        self.assertTrue(s1.vision_uncalibrated)

        # Without sound -> remains in SEARCHING and exposes vision_uncalibrated=True
        s2 = sm.from_pipeline_outputs(now=1.1, vision=uncal_vision, detection=_make_detection(False))
        self.assertEqual(s2.mode, TrackingMode.SEARCHING)
        self.assertEqual(s2.active_source, "NONE")
        self.assertTrue(s2.vision_uncalibrated)

    def test_require_drone_for_audio_gating(self):
        sm = TargetStateMachine(StateMachineConfig(require_drone_for_audio=True))
        det = _make_detection(True)
        tri = TriangulationResult(True, 0.5, 1.5, 1.58, 18.4, 0.9, 2, {"front": 18.4, "left": 18.4})

        non_drone = ClassificationResult(False, 0.2, 0.2, 0, 5, 0.75, (0.8, 0.2), True)
        s1 = sm.from_pipeline_outputs(now=1.0, detection=det, triangulation=tri, classification=non_drone)
        self.assertEqual(s1.mode, TrackingMode.SEARCHING)

        confirmed = ClassificationResult(True, 0.9, 0.9, 3, 5, 0.75, (0.1, 0.9), True)
        s2 = sm.from_pipeline_outputs(now=1.1, detection=det, triangulation=tri, classification=confirmed)
        self.assertEqual(s2.mode, TrackingMode.ACOUSTIC_TRACKING)
        self.assertEqual(s2.active_source, "SOUND")
        self.assertAlmostEqual(s2.bearing_deg, 18.4)
        self.assertTrue(s2.audio_drone_confirmed)

    def test_suppressed_acoustic_direction_still_allows_vision_tracking(self):
        sm = TargetStateMachine()
        s1 = sm.from_pipeline_outputs(
            now=1.0,
            detection=_make_detection(True),
            single_doa=_make_doa(bearing=60.0),
            suppressed=True,
        )
        self.assertEqual(s1.mode, TrackingMode.SEARCHING)

        s2 = sm.from_pipeline_outputs(
            now=1.1,
            vision=_make_vision(bearing=45.2, elevation=12.4),
            detection=_make_detection(True),
            single_doa=_make_doa(bearing=60.0),
            suppressed=True,
        )
        self.assertEqual(s2.mode, TrackingMode.VISION_TRACKING)
        self.assertEqual(s2.active_source, "VISION")

    def test_format_fused_state_output(self):
        sm = TargetStateMachine()
        state = sm.from_pipeline_outputs(
            now=1.0,
            vision=_make_vision(bearing=45.2, elevation=12.4),
        )
        text = format_fused_state(state)
        lines = text.splitlines()
        self.assertEqual(len(lines), 5)
        self.assertEqual(lines[0], "MODE: VISION_TRACKING")
        self.assertEqual(lines[1], "SOURCE: VISION")
        self.assertEqual(lines[2], "BEARING: 45.2°")
        self.assertEqual(lines[3], "ELEVATION: 12.4°")
        self.assertTrue(lines[4].startswith("DIRECTION_VECTOR: ("))


if __name__ == "__main__":
    unittest.main()
