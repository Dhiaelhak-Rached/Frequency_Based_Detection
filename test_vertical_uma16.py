import unittest

import numpy as np

from doa_engine import ArrayDOA, SOUND_SPEED
from geometry import RigGeometry, bearing_to_vector


UMA16_LAYOUT = {
    0: (3, 1), 1: (3, 0), 2: (2, 1), 3: (2, 0),
    4: (1, 1), 5: (1, 0), 6: (0, 1), 7: (0, 0),
    8: (0, 3), 9: (0, 2), 10: (1, 3), 11: (1, 2),
    12: (2, 3), 13: (2, 2), 14: (3, 3), 15: (3, 2),
}


def angle_error(actual, expected):
    return abs((actual - expected + 180.0) % 360.0 - 180.0)


def vertical_uma16_block(arr, grid_m, azimuth_deg, elevation_deg, seed, snr_db=20.0):
    sample_rate = 44100
    frequency = 1010.0
    n_samples = 8820
    normal = bearing_to_vector(arr.orientation_deg)
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

    for ch, (row, col) in UMA16_LAYOUT.items():
        horizontal = (col - 1.5) * grid_m
        height = (1.5 - row) * grid_m
        relative = horizontal * right
        delay = -(np.dot(relative, direction) + height * vertical_direction) / SOUND_SPEED
        tone = np.sin(2.0 * np.pi * frequency * (t - delay))
        noise = rng.normal(0.0, 10.0 ** (-snr_db / 20.0), n_samples)
        block[:, ch] = tone + noise
    return block


def estimate(arr, block):
    kwargs = {"mount": arr.mount} if hasattr(arr, "mount") else {}
    engine = ArrayDOA(
        arr.array_id,
        arr.local_mics,
        arr.orientation_deg,
        sample_rate=44100,
        **kwargs,
    )
    return engine.estimate(block, target_freq=1010.0, band=(980.0, 1040.0))


class VerticalUMA16Test(unittest.TestCase):
    def test_vertical_azimuth_is_independent_of_elevation(self):
        geo = RigGeometry()
        arr = geo.arrays["left"]
        grid_m = geo.cfg["arrays"]["left"]["grid_m"]

        for azimuth in (-60.0, -30.0, 0.0, 30.0, 60.0):
            estimates = []
            for elevation in (-15.0, 15.0):
                with self.subTest(azimuth=azimuth, elevation=elevation):
                    block = vertical_uma16_block(
                        arr, grid_m, azimuth, elevation,
                        seed=int(1000 + azimuth * 10 + elevation),
                    )
                    result = estimate(arr, block)
                    print(
                        f"UMA16 az={azimuth:+.0f} el={elevation:+.0f} -> "
                        f"local={result.local_angle_deg} prominence={result.prominence:.2f}",
                        flush=True,
                    )
                    self.assertIsNotNone(result.local_angle_deg)
                    if result.local_angle_deg is not None:
                        estimates.append(result.local_angle_deg)
                        self.assertLessEqual(angle_error(result.local_angle_deg, azimuth), 5.0)
            if len(estimates) == 2:
                with self.subTest(azimuth=azimuth, elevation="pair"):
                    self.assertLessEqual(angle_error(estimates[0], estimates[1]), 2.0)

    def test_uma8_flat_ring_regression(self):
        geo = RigGeometry()
        arr = geo.arrays["front"]
        expected = 35.0
        direction = bearing_to_vector(expected)
        sample_rate = 44100
        frequency = 1010.0
        n_samples = 8820
        t = np.arange(n_samples) / sample_rate
        rng = np.random.default_rng(35)
        block = np.zeros((n_samples, 8), dtype=np.float64)
        for ch, pos in arr.mic_positions.items():
            delay = -np.dot(pos - arr.center, direction) / SOUND_SPEED
            tone = np.sin(2.0 * np.pi * frequency * (t - delay))
            block[:, ch] = tone + rng.normal(0.0, 0.05, n_samples)

        result = estimate(arr, block)
        print(
            f"UMA8 az={expected:+.0f} -> global={result.global_angle_deg} "
            f"prominence={result.prominence:.2f}",
            flush=True,
        )
        self.assertIsNotNone(result.global_angle_deg)
        self.assertLessEqual(angle_error(result.global_angle_deg, expected), 5.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
