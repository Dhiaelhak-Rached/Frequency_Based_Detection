"""Offline unit tests for the 2-drone Gazebo + real UMA-16 HIL runner."""

import numpy as np
import unittest

from frequency_id.control.guidance import AcousticObservation, GuidanceCommand, GuidanceState
from sim.sensors.acoustic_dsp import AcousticWavefieldDOASensor
from sim.runners.dual_drone_uma16 import (
    apply_fixed_follow_distance,
    build_parser,
    compute_auto_target_command,
    compute_target_drone_command,
)
from sim.harness.sitl_harness import PX4InstanceConfig, assert_sitl_endpoint


class DualDroneUMA16Test(unittest.TestCase):
    def test_instance_ports_and_poses(self):
        d1 = PX4InstanceConfig(instance_id=0, pose_enu="0,0,0,0,0,0", udp_port=14540, mavsdk_port=50051)
        d2 = PX4InstanceConfig(instance_id=1, pose_enu="0,8,0,0,0,0", udp_port=14541, mavsdk_port=50052)
        self.assertEqual(d1.connect_address, "udp://:14540")
        self.assertEqual(d2.connect_address, "udp://:14541")
        assert_sitl_endpoint(d1.connect_address)
        assert_sitl_endpoint(d2.connect_address)
        with self.assertRaises(RuntimeError):
            assert_sitl_endpoint("serial:///dev/ttyUSB0:921600")

    def test_target_drone_hovers_when_silent_or_stale(self):
        cmd_none = compute_target_drone_command(None, 10.0, 0.0, 0.0, 5.0, 0.0)
        self.assertFalse(cmd_none.active)
        self.assertEqual(cmd_none.forward_m_s, 0.0)
        self.assertEqual(cmd_none.yaw_rate_deg_s, 0.0)

        stale_obs = AcousticObservation(
            captured_at=5.0,
            sequence=1,
            azimuth_deg=20.0,
            elevation_deg=10.0,
            azimuth_prominence=2.0,
            elevation_prominence=0.8,
            azimuth_stable=True,
            elevation_stable=True,
        )
        cmd_stale = compute_target_drone_command(stale_obs, 10.0, 0.0, 0.0, 5.0, 0.0, timeout_s=0.8)
        self.assertFalse(cmd_stale.active)

    def test_target_drone_moves_left_up_and_respects_radius_fence(self):
        obs = AcousticObservation(
            captured_at=10.0,
            sequence=2,
            azimuth_deg=-35.0,
            elevation_deg=20.0,
            azimuth_prominence=2.0,
            elevation_prominence=0.8,
            azimuth_stable=True,
            elevation_stable=True,
        )
        cmd = compute_target_drone_command(obs, 10.1, 0.0, 0.0, 5.0, 0.0, max_radius_m=15.0)
        self.assertTrue(cmd.active)
        self.assertLess(cmd.right_m_s, 0.0)   # left
        self.assertLess(cmd.down_m_s, 0.0)    # up
        self.assertGreater(cmd.forward_m_s, 0.0)

        # At 16 m North facing North (yaw=0), outward forward must clamp to 0
        cmd_fence = compute_target_drone_command(obs, 10.1, 16.0, 0.0, 5.0, 0.0, max_radius_m=15.0)
        self.assertEqual(cmd_fence.forward_m_s, 0.0)

    def test_fixed_follow_distance_holds_and_tracks(self):
        base = GuidanceCommand(forward_m_s=0.5, state=GuidanceState.FOLLOW, reason="following")
        cmd_far = apply_fixed_follow_distance(base, horiz_dist_m=8.0, follow_distance_m=5.0)
        self.assertGreater(cmd_far.forward_m_s, 0.5)
        cmd_at = apply_fixed_follow_distance(base, horiz_dist_m=5.0, follow_distance_m=5.0)
        self.assertEqual(cmd_at.forward_m_s, 0.0)
        cmd_close = apply_fixed_follow_distance(base, horiz_dist_m=3.5, follow_distance_m=5.0)
        self.assertLess(cmd_close.forward_m_s, 0.0)

    def test_auto_target_and_cli_defaults(self):
        cmd = compute_auto_target_command(elapsed_s=2.0, alt_m=5.0)
        self.assertTrue(cmd.active)
        self.assertGreater(cmd.forward_m_s, 0.0)
        args = build_parser().parse_args([])
        self.assertEqual(args.follow_mode, "acoustic-dsp")
        self.assertEqual(args.target_mode, "shared-mic")
        self.assertFalse(args.with_classifier)
        self.assertFalse(args.headless)

    def test_acoustic_wavefield_dsp_sensor_computes_real_3d_doa(self):
        sensor = AcousticWavefieldDOASensor()
        payload = None
        for k in range(5):
            payload, diag = sensor.sample(
                now=1.0 + k * 0.2,
                drone_ned=np.array([0.0, 0.0, -5.0]),
                drone_rpy_deg=(0.0, 0.0, 0.0),
                target_ned=np.array([8.0, 4.0, -7.0]),
                target_active=True,
            )
        self.assertTrue(diag["detected"])
        self.assertTrue(payload["azimuth_stable"])
        self.assertTrue(payload["elevation_stable"])
        az = (payload["azimuth_deg"] + 180.0) % 360.0 - 180.0
        self.assertGreater(az, 15.0)
        self.assertGreater(payload["elevation_deg"], 5.0)


if __name__ == "__main__":
    unittest.main()
