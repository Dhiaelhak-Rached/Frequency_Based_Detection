import unittest

import numpy as np

from detector import DetectionResult, FrequencyDetector
from direction_suppression import (
    DirectionSuppression,
    displayable_doa,
    select_front_only_target,
)
from doa_engine import ArrayDOA, DOAResult, SOUND_SPEED
from geometry import RigGeometry


class FakeDOA:
    def __init__(self, bearings):
        self.bearings = bearings
        self.calls = []

    def estimate(self, block, target_freq, band, track=True):
        self.calls.append((target_freq, track))
        return DOAResult(
            array_id="left",
            local_angle_deg=self.bearings[target_freq],
            global_angle_deg=self.bearings[target_freq],
            prominence=5.0,
            confidence=0.5,
            spectrum=None,
        )


def detection_with_candidates(candidates):
    peak_freq, peak_db = candidates[0]
    return DetectionResult(
        detected=True,
        peak_freq=peak_freq,
        peak_db=peak_db,
        threshold_db=-65.0,
        band=(800.0, 1200.0),
        freqs=np.array([]),
        mags=np.array([]),
        peak_candidates=tuple(candidates),
    )


def two_source_uma16_block(arr):
    sample_rate = 44100
    n_samples = 8820
    t = np.arange(n_samples) / sample_rate
    block = np.zeros((n_samples, 16), dtype=np.float64)
    sources = (
        (900.0, 30.0, 8.0, 1.0),
        (1100.0, 330.0, -12.0, 0.6),
    )
    for frequency, azimuth_deg, elevation_deg, amplitude in sources:
        azimuth = np.deg2rad(azimuth_deg)
        elevation = np.deg2rad(elevation_deg)
        direction = np.array([
            np.cos(elevation) * np.sin(azimuth),
            np.cos(elevation) * np.cos(azimuth),
            np.sin(elevation),
        ])
        for ch in arr.local_mics:
            position = arr.mic_positions[ch]
            delay = -(
                np.dot(position, direction[:2])
                + arr.mic_heights_local[ch] * direction[2]
            ) / SOUND_SPEED
            block[:, ch] += amplitude * np.sin(2.0 * np.pi * frequency * (t - delay))
    return block


class DirectionSuppressionTest(unittest.TestCase):
    def test_wrap_safe_exclusion_zone(self):
        suppression = DirectionSuppression(True, 0.0, 10.0)
        for bearing in (0.0, 359.0, 1.0, 350.0, 10.0, 360.0):
            with self.subTest(bearing=bearing):
                self.assertTrue(suppression.contains(bearing))
        for bearing in (10.1, 349.9, 90.0, 180.0):
            with self.subTest(bearing=bearing):
                self.assertFalse(suppression.contains(bearing))
        self.assertFalse(suppression.contains(None))
        self.assertFalse(DirectionSuppression(False, 0.0, 10.0).contains(0.0))

    def test_configuration_validation_and_normalization(self):
        suppression = DirectionSuppression.from_config(
            {"enabled": False, "tolerance_deg": 8.0},
            bearing_override=-10.0,
        )
        self.assertTrue(suppression.enabled)
        self.assertEqual(suppression.bearing_deg, 350.0)
        for tolerance in (0.0, -1.0, 90.1):
            with self.subTest(tolerance=tolerance):
                with self.assertRaises(ValueError):
                    DirectionSuppression(True, 0.0, tolerance)
        with self.assertRaises(ValueError):
            DirectionSuppression(True, 0.0, 10.0, 0)

    def test_stronger_suppressed_peak_falls_back_to_other_source(self):
        detection = detection_with_candidates(((900.0, -20.0), (1100.0, -30.0)))
        engine = FakeDOA({900.0: 30.0, 1100.0: 330.0})
        selection = select_front_only_target(
            detection,
            None,
            engine,
            (800.0, 1200.0),
            DirectionSuppression(True, 30.0, 10.0),
        )
        self.assertFalse(selection.suppressed)
        self.assertEqual(selection.detection.peak_freq, 1100.0)
        self.assertEqual(selection.doa.global_angle_deg, 330.0)
        self.assertEqual(engine.calls, [(900.0, False), (1100.0, False), (1100.0, True)])

    def test_all_suppressed_candidates_have_no_displayable_output_or_tracking(self):
        detection = detection_with_candidates(((900.0, -20.0), (1100.0, -30.0)))
        engine = FakeDOA({900.0: 30.0, 1100.0: 330.0})
        selection = select_front_only_target(
            detection,
            None,
            engine,
            (800.0, 1200.0),
            DirectionSuppression(True, 0.0, 90.0),
        )
        self.assertTrue(selection.suppressed)
        self.assertFalse(selection.detection.detected)
        self.assertIsNone(selection.detection.peak_freq)
        self.assertIsNone(selection.doa)
        self.assertTrue(all(not track for _, track in engine.calls))

    def test_disabled_filter_preserves_strongest_peak(self):
        detection = detection_with_candidates(((900.0, -20.0), (1100.0, -30.0)))
        engine = FakeDOA({900.0: 30.0, 1100.0: 330.0})
        selection = select_front_only_target(
            detection,
            None,
            engine,
            (800.0, 1200.0),
            DirectionSuppression(False, 30.0, 10.0),
        )
        self.assertEqual(selection.detection.peak_freq, 900.0)
        self.assertEqual(engine.calls, [(900.0, True)])

    def test_final_display_guard(self):
        result = DOAResult("left", 355.0, 355.0, 5.0, 0.9, None)
        suppression = DirectionSuppression(True, 0.0, 10.0)
        self.assertFalse(displayable_doa(result, suppression))
        self.assertTrue(displayable_doa(result, DirectionSuppression(False, 0.0, 10.0)))
        self.assertFalse(displayable_doa(None, suppression))

    def test_real_doa_selects_weaker_tone_without_rejected_tracker_history(self):
        geo = RigGeometry()
        geo.set_uma16_front_only(True)
        arr = geo.arrays["left"]
        block = two_source_uma16_block(arr)
        detector = FrequencyDetector(
            sample_rate=geo.sample_rate,
            block_duration=0.2,
            average_blocks=1,
            threshold_dbfs=-65.0,
        )
        detection = detector.process(block[:, 0], 800.0, 1200.0)
        self.assertAlmostEqual(detection.peak_freq, 900.0, delta=1.0)
        self.assertGreaterEqual(len(detection.peak_candidates), 2)
        engine = ArrayDOA(
            arr.array_id,
            arr.local_mics,
            arr.orientation_deg,
            sample_rate=geo.sample_rate,
            mount=arr.mount,
            mic_heights_local=arr.mic_heights_local,
        )
        selection = select_front_only_target(
            detection,
            block,
            engine,
            (800.0, 1200.0),
            DirectionSuppression(True, 30.0, 10.0),
        )
        self.assertFalse(selection.suppressed)
        self.assertAlmostEqual(selection.detection.peak_freq, 1100.0, delta=1.0)
        self.assertAlmostEqual(selection.doa.global_angle_deg, 330.0, delta=2.0)
        self.assertEqual(len(engine.tracker.estimates), 1)
        tracked_angle = np.rad2deg(engine.tracker.estimates[0]) % 360.0
        self.assertAlmostEqual(tracked_angle, 330.0, delta=2.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
