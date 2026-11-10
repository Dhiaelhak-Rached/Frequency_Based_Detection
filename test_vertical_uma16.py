import unittest

import numpy as np

from doa_engine import ArrayDOA, CircularTracker, SOUND_SPEED
from geometry import RigGeometry, bearing_to_vector


UMA16_LAYOUT = {
    0: (3, 1), 1: (3, 0), 2: (2, 1), 3: (2, 0),
    4: (1, 1), 5: (1, 0), 6: (0, 1), 7: (0, 0),
    8: (0, 3), 9: (0, 2), 10: (1, 3), 11: (1, 2),
    12: (2, 3), 13: (2, 2), 14: (3, 3), 15: (3, 2),
}
UMA8_ANGLES = {4: 10.0, 5: 70.0, 6: 130.0, 1: 190.0, 2: 250.0, 3: 310.0}


def angle_error(actual, expected):
    return abs((actual - expected + 180.0) % 360.0 - 180.0)


def wave_block(channels, positions, heights, normal, azimuth_deg, elevation_deg, seed, snr_db=20.0):
    sample_rate = 44100
    frequency = 1010.0
    n_samples = 8820
    right = np.array([normal[1], -normal[0]])
    azimuth = np.deg2rad(azimuth_deg)
    elevation = np.deg2rad(elevation_deg)
    direction = np.cos(elevation) * (
        np.cos(azimuth) * normal + np.sin(azimuth) * right
    )
    vertical_direction = np.sin(elevation)
    t = np.arange(n_samples) / sample_rate
    rng = np.random.default_rng(seed)
    block = np.zeros((n_samples, 16), dtype=np.float64)

    for ch in channels:
        delay = -(
            np.dot(positions[ch], direction) + heights[ch] * vertical_direction
        ) / SOUND_SPEED
        tone = np.sin(2.0 * np.pi * frequency * (t - delay))
        block[:, ch] = tone + rng.normal(0.0, 10.0 ** (-snr_db / 20.0), n_samples)
    return block


def vertical_uma16_block(arr, grid_m, azimuth_deg, elevation_deg, seed, snr_db=20.0):
    normal = bearing_to_vector(arr.orientation_deg)
    right = np.array([normal[1], -normal[0]])
    positions = {}
    heights = {}
    for ch, (row, col) in UMA16_LAYOUT.items():
        positions[ch] = (col - 1.5) * grid_m * right
        heights[ch] = (1.5 - row) * grid_m
    return wave_block(
        range(16), positions, heights, normal, azimuth_deg, elevation_deg, seed, snr_db
    )


def vertical_uma8_block(arr, azimuth_deg, elevation_deg, seed, snr_db=20.0):
    radius = 0.04275
    normal = bearing_to_vector(arr.orientation_deg)
    right = np.array([normal[1], -normal[0]])
    positions = {0: np.zeros(2)}
    heights = {0: 0.0}
    for ch, deg in UMA8_ANGLES.items():
        angle = np.deg2rad(deg)
        positions[ch] = radius * np.sin(angle) * right
        heights[ch] = radius * np.cos(angle)
    return wave_block(
        range(7), positions, heights, normal, azimuth_deg, elevation_deg, seed, snr_db
    )[:, :8]


def make_engine(arr):
    return ArrayDOA(
        arr.array_id,
        arr.local_mics,
        arr.orientation_deg,
        sample_rate=44100,
        mount=arr.mount,
    )


def estimate(arr, block):
    return make_engine(arr).estimate(block, target_freq=1010.0, band=(980.0, 1040.0))


class VerticalArrayTest(unittest.TestCase):
    def assert_vertical_cases(self, arr, block_factory):
        for azimuth in (-60.0, -30.0, 0.0, 30.0, 60.0):
            estimates = []
            for elevation in (-15.0, 15.0):
                with self.subTest(array=arr.array_id, azimuth=azimuth, elevation=elevation):
                    block = block_factory(azimuth, elevation)
                    result = estimate(arr, block)
                    print(
                        f"{arr.array_id} az={azimuth:+.0f} el={elevation:+.0f} -> "
                        f"local={result.local_angle_deg:.3f} global={result.global_angle_deg:.3f} "
                        f"prominence={result.prominence:.2f}",
                        flush=True,
                    )
                    self.assertIsNotNone(result.local_angle_deg)
                    estimates.append(result.local_angle_deg)
                    self.assertLessEqual(angle_error(result.local_angle_deg, azimuth), 5.0)
            with self.subTest(array=arr.array_id, azimuth=azimuth, elevation="pair"):
                self.assertLessEqual(angle_error(estimates[0], estimates[1]), 2.0)

    def test_vertical_uma16_azimuth_is_independent_of_elevation(self):
        geo = RigGeometry()
        arr = geo.arrays["left"]
        grid_m = geo.cfg["arrays"]["left"]["grid_m"]
        self.assert_vertical_cases(
            arr,
            lambda az, el: vertical_uma16_block(
                arr, grid_m, az, el, seed=int(1000 + az * 10 + el)
            ),
        )

    def test_vertical_uma8_azimuth_is_independent_of_elevation(self):
        arr = RigGeometry().arrays["front"]
        self.assert_vertical_cases(
            arr,
            lambda az, el: vertical_uma8_block(arr, az, el, seed=int(2000 + az * 10 + el)),
        )

    def test_global_normals(self):
        geo = RigGeometry()
        self.assertEqual(geo.local_to_global_bearing("front", 0.0), 0.0)
        self.assertEqual(geo.local_to_global_bearing("left", 0.0), 270.0)

    def test_uma16_front_only_geometry_and_restore(self):
        geo = RigGeometry()
        original_center = geo.arrays["left"].center.copy()
        original_positions = {
            ch: pos.copy() for ch, pos in geo.arrays["left"].mic_positions.items()
        }

        geo.set_uma16_front_only(True)
        arr = geo.arrays["left"]
        np.testing.assert_allclose(arr.center, [0.0, 0.0])
        self.assertEqual(arr.orientation_deg, 0.0)
        self.assertEqual(arr.name, "UMA-16 Front")
        self.assertEqual(geo.local_to_global_bearing("left", 0.0), 0.0)

        block = vertical_uma16_block(arr, 0.042, 30.0, 15.0, seed=130)
        result = estimate(arr, block)
        self.assertLessEqual(angle_error(result.global_angle_deg, 30.0), 5.0)

        geo.set_uma16_front_only(False)
        restored = geo.arrays["left"]
        np.testing.assert_allclose(restored.center, original_center)
        self.assertEqual(restored.orientation_deg, 270.0)
        self.assertEqual(restored.name, "UMA-16 Left")
        for ch, pos in original_positions.items():
            np.testing.assert_allclose(restored.mic_positions[ch], pos)

    def test_vertical_grid_boundary_and_alias_limit(self):
        geo = RigGeometry()
        arr = geo.arrays["left"]
        engine = make_engine(arr)
        self.assertEqual(len(engine.angles), 181)
        self.assertEqual(engine.angles[0], 270.0)
        self.assertEqual(engine.angles[-1], 90.0)
        self.assertAlmostEqual(engine.max_direction_freq_hz, 343.0 / 0.084)
        block = vertical_uma16_block(arr, 0.042, 90.0, 0.0, seed=90, snr_db=30.0)
        result = engine.estimate(block, target_freq=1010.0, band=(980.0, 1040.0))
        self.assertLessEqual(angle_error(result.local_angle_deg, 90.0), 2.0)

    def test_low_prominence_requires_temporal_stability(self):
        geo = RigGeometry()
        arr = geo.arrays["left"]
        engine = make_engine(arr)
        block = vertical_uma16_block(arr, 0.042, 30.0, 15.0, seed=30)
        results = [
            engine.estimate(block, target_freq=1010.0, band=(980.0, 1040.0))
            for _ in range(3)
        ]
        self.assertLess(results[-1].prominence, 1.4)
        self.assertFalse(results[0].stable)
        self.assertFalse(results[1].stable)
        self.assertTrue(results[2].stable)
        self.assertGreaterEqual(results[2].confidence, 0.85)

    def test_jump_does_not_become_stable(self):
        tracker = CircularTracker(window=5, min_prominence=0.45, min_samples=3)
        tracker.add(10.0, 0.6)
        tracker.add(12.0, 0.6)
        tracker.add(120.0, 0.6)
        self.assertFalse(tracker.ready)
        self.assertEqual(len(tracker.estimates), 1)

    def test_flat_uma8_ring_regression(self):
        expected = 35.0
        expected_rad = np.deg2rad(expected)
        direction = np.array([np.cos(expected_rad), np.sin(expected_rad)])
        radius = 0.04275
        local_positions = {0: np.zeros(2)}
        for ch, deg in UMA8_ANGLES.items():
            angle = np.deg2rad(deg)
            local_positions[ch] = radius * np.array([np.cos(angle), np.sin(angle)])

        sample_rate = 44100
        frequency = 1010.0
        t = np.arange(8820) / sample_rate
        rng = np.random.default_rng(35)
        block = np.zeros((8820, 8), dtype=np.float64)
        for ch, pos in local_positions.items():
            delay = -np.dot(pos, direction) / SOUND_SPEED
            tone = np.sin(2.0 * np.pi * frequency * (t - delay))
            block[:, ch] = tone + rng.normal(0.0, 0.05, 8820)

        engine = ArrayDOA("front", local_positions, 0.0, sample_rate=sample_rate, mount="flat")
        result = engine.estimate(block, target_freq=1010.0, band=(980.0, 1040.0))
        print(f"flat UMA8 az={expected:+.0f} -> global={result.global_angle_deg:.3f}", flush=True)
        self.assertLessEqual(angle_error(result.global_angle_deg, expected), 5.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
