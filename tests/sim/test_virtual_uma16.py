"""Unit tests for the virtual 3D UMA-16 acoustic sensor and isolation guarantees."""

import numpy as np
import unittest

from frequency_id.control.guidance import AcousticObservation
from frequency_id.control.px4 import validate_observation_metadata
from sim.sensors.virtual_uma16 import AcousticSensorConfig, VirtualUMA16Sensor, ideal_body_angles, ned_to_body_rig


class VirtualUMA16SensorTest(unittest.TestCase):
    def test_ned_to_body_rig_signs_and_yaw(self):
        # Target North-East (+10, +10), same alt -> at yaw=0: right=+10, fwd=+10, up=0
        rel = ned_to_body_rig(np.array([10.0, 10.0, 0.0]), 0.0, 0.0, 0.0)
        np.testing.assert_allclose(rel, [10.0, 10.0, 0.0], atol=1e-6)
        az, el, _ = ideal_body_angles(rel)
        self.assertAlmostEqual(az, 45.0, delta=1e-4)
        self.assertAlmostEqual(el, 0.0, delta=1e-4)

        # Facing East (yaw=+90): target at North-East (+10, +10) is now 45 deg LEFT (az=-45)
        rel90 = ned_to_body_rig(np.array([10.0, 10.0, -5.0]), 0.0, 0.0, 90.0)
        az90, el90, _ = ideal_body_angles(rel90)
        self.assertAlmostEqual(az90, -45.0, delta=1e-4)
        self.assertGreater(el90, 0.0)

    def test_payload_contains_only_acoustic_metadata_never_target_coords(self):
        sensor = VirtualUMA16Sensor()
        for k in range(5):
            payload, _ = sensor.sample(
                1.0 + k * 0.2,
                np.array([0.0, 0.0, -5.0]),
                (0.0, 0.0, 0.0),
                np.array([8.0, 4.0, -7.0]),
            )
        validate_observation_metadata(payload, "uma16-front-body-v1")
        for forbidden in ("target_n", "target_e", "target_alt", "true_range_m", "drone_n"):
            self.assertNotIn(forbidden, payload)
        obs = AcousticObservation.from_mapping(payload)
        self.assertTrue(obs.azimuth_stable)
        self.assertTrue(obs.elevation_stable)
        self.assertGreater(obs.azimuth_deg, 20.0)
        self.assertGreater(obs.elevation_deg, 10.0)

    def test_dropout_and_low_elevation_prominence_fail_closed(self):
        sensor = VirtualUMA16Sensor(sensor_config=AcousticSensorConfig(force_dropout=True))
        payload, _ = sensor.sample(1.0, np.zeros(3), (0.0, 0.0, 0.0), np.array([8.0, 0.0, -5.0]))
        self.assertIsNone(payload["azimuth_deg"])

        sensor2 = VirtualUMA16Sensor(sensor_config=AcousticSensorConfig(force_low_elev_prominence=True))
        payload2, _ = sensor2.sample(1.0, np.zeros(3), (0.0, 0.0, 0.0), np.array([8.0, 0.0, -7.0]))
        self.assertLess(payload2["elevation_prominence"], 0.45)
        self.assertFalse(payload2["elevation_stable"])


if __name__ == "__main__":
    unittest.main()
