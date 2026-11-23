import unittest

import numpy as np

from audio_manager import AudioBlock, MultiDeviceAudioManager
from doa_engine import DOAResult
from geometry import RigGeometry
from guidance import (
    AcousticObservation,
    FlightState,
    GuidanceConfig,
    GuidanceController,
    GuidanceState,
    wrap_angle_deg,
)
from main import build_guidance_payload


def observation(sequence, stamp, azimuth=0.0, elevation=0.0, **changes):
    values = dict(
        captured_at=stamp,
        sequence=sequence,
        azimuth_deg=azimuth,
        elevation_deg=elevation,
        azimuth_prominence=2.0,
        elevation_prominence=2.0,
        azimuth_stable=True,
        elevation_stable=True,
        error=None,
        suppressed=False,
    )
    values.update(changes)
    return AcousticObservation(**values)


def flight(altitude=10.0, distance=0.0, battery=80.0, offboard=True, fresh=True, geofence=True):
    return FlightState(
        altitude_m=altitude,
        distance_from_start_m=distance,
        battery_percent=battery,
        offboard_active=offboard,
        telemetry_fresh=fresh,
        geofence_ok=geofence,
    )


def make_controller(**changes):
    values = dict(
        required_stable_samples=2,
        filter_alpha=1.0,
        max_vertical_accel_m_s2=100.0,
        max_horizontal_accel_m_s2=100.0,
    )
    values.update(changes)
    controller = GuidanceController(GuidanceConfig(**values))
    controller.enable()
    return controller


def drive_to_follow(controller, start=1.0):
    for index in range(5):
        stamp = start + index * 0.2
        command = controller.update(stamp, observation(index + 1, stamp), flight())
    return command


class GuidanceTest(unittest.TestCase):
    def test_disabled_and_search_are_hover(self):
        controller = GuidanceController()
        command = controller.update(1.0, observation(1, 1.0), flight())
        self.assertEqual(command.state, GuidanceState.DISABLED)
        self.assertEqual(command.forward_m_s, 0.0)
        controller.enable()
        command = controller.update(1.0, observation(1, 1.0), flight())
        self.assertEqual(command.state, GuidanceState.HOVER_SEARCH)
        self.assertEqual(command.down_m_s, 0.0)
        self.assertEqual(command.yaw_rate_deg_s, 0.0)

    def test_stable_samples_require_distinct_blocks(self):
        controller = make_controller(required_stable_samples=3)
        first = observation(1, 1.0)
        controller.update(1.0, first, flight())
        for index in range(1, 5):
            command = controller.update(1.0 + index * 0.05, first, flight())
            self.assertEqual(command.state, GuidanceState.HOVER_SEARCH)
        command = controller.update(1.3, observation(2, 1.3), flight())
        self.assertEqual(command.state, GuidanceState.HOVER_SEARCH)
        command = controller.update(1.5, observation(3, 1.5), flight())
        self.assertEqual(command.state, GuidanceState.ALIGN_YAW)

    def test_yaw_wrap_and_saturation(self):
        self.assertAlmostEqual(wrap_angle_deg(350.0), -10.0)
        controller = make_controller(required_stable_samples=1)
        controller.update(1.0, observation(1, 1.0, azimuth=350.0), flight())
        command = controller.update(1.2, observation(2, 1.2, azimuth=350.0), flight())
        self.assertEqual(command.state, GuidanceState.ALIGN_YAW)
        self.assertAlmostEqual(command.yaw_rate_deg_s, -8.0)
        self.assertEqual(command.down_m_s, 0.0)
        command = controller.update(1.4, observation(3, 1.4, azimuth=80.0), flight())
        self.assertEqual(command.yaw_rate_deg_s, 20.0)

    def test_vertical_waits_for_yaw_and_uses_elevation_sign(self):
        controller = make_controller()
        controller.update(1.0, observation(1, 1.0, azimuth=30.0, elevation=20.0), flight())
        controller.update(1.2, observation(2, 1.2, azimuth=30.0, elevation=20.0), flight())
        command = controller.update(1.4, observation(3, 1.4, azimuth=30.0, elevation=20.0), flight())
        self.assertEqual(command.state, GuidanceState.ALIGN_YAW)
        self.assertEqual(command.down_m_s, 0.0)
        command = controller.update(1.6, observation(4, 1.6, azimuth=0.0, elevation=20.0), flight())
        self.assertEqual(command.state, GuidanceState.ALIGN_VERTICAL)
        self.assertAlmostEqual(command.down_m_s, -0.3)
        command = controller.update(1.8, observation(5, 1.8, azimuth=0.0, elevation=-20.0), flight())
        self.assertAlmostEqual(command.down_m_s, 0.3)

    def test_forward_motion_requires_stable_horizontal_and_vertical_alignment(self):
        controller = make_controller()
        for index, state in enumerate(
            (
                GuidanceState.HOVER_SEARCH,
                GuidanceState.ALIGN_YAW,
                GuidanceState.ALIGN_VERTICAL,
                GuidanceState.FOLLOW,
                GuidanceState.FOLLOW,
            )
        ):
            stamp = 1.0 + index * 0.2
            command = controller.update(stamp, observation(index + 1, stamp), flight())
            self.assertEqual(command.state, state)
            if state != GuidanceState.FOLLOW or index == 3:
                self.assertEqual(command.forward_m_s, 0.0)
        self.assertEqual(command.forward_m_s, 0.5)
        self.assertEqual(command.down_m_s, 0.0)

    def test_alignment_samples_require_distinct_blocks(self):
        controller = make_controller(required_stable_samples=2)
        for index in range(3):
            stamp = 1.0 + index * 0.2
            command = controller.update(stamp, observation(index + 1, stamp), flight())
        self.assertEqual(command.state, GuidanceState.ALIGN_VERTICAL)
        for index in range(4):
            command = controller.update(1.65 + index * 0.05, observation(3, 1.4), flight())
            self.assertNotEqual(command.state, GuidanceState.FOLLOW)
        command = controller.update(1.9, observation(4, 1.9), flight())
        self.assertEqual(command.state, GuidanceState.FOLLOW)

    def test_stale_invalid_and_lost_observations_hover(self):
        for change in (
            {"elevation_deg": None},
            {"elevation_stable": False},
            {"elevation_prominence": 0.1},
            {"error": "spatial_aliasing"},
            {"suppressed": True},
        ):
            with self.subTest(change=change):
                controller = make_controller(required_stable_samples=1)
                command = controller.update(1.0, observation(1, 1.0, **change), flight())
                self.assertEqual(command.state, GuidanceState.TARGET_LOST_HOLD)
                self.assertEqual(command.down_m_s, 0.0)
                self.assertEqual(command.forward_m_s, 0.0)

        controller = make_controller()
        drive_to_follow(controller)
        command = controller.update(4.0, None, flight())
        self.assertEqual(command.state, GuidanceState.TARGET_LOST_HOLD)
        self.assertEqual(command.forward_m_s, 0.0)

    def test_repeated_observation_cannot_outlive_timeout(self):
        controller = make_controller()
        drive_to_follow(controller)
        repeated = observation(5, 1.8)
        command = controller.update(2.0, repeated, flight())
        self.assertEqual(command.state, GuidanceState.FOLLOW)
        command = controller.update(2.5, repeated, flight())
        self.assertEqual(command.state, GuidanceState.TARGET_LOST_HOLD)
        self.assertEqual(command.forward_m_s, 0.0)

    def test_future_and_old_timestamps_are_rejected(self):
        for stamp in (0.0, 2.0):
            with self.subTest(stamp=stamp):
                controller = make_controller()
                command = controller.update(1.0, observation(1, stamp), flight())
                self.assertEqual(command.state, GuidanceState.TARGET_LOST_HOLD)
                self.assertEqual(command.down_m_s, 0.0)

    def test_altitude_and_vertical_travel_limits_stop_vertical_motion(self):
        for altitude, elevation in ((20.0, 20.0), (2.0, -20.0)):
            with self.subTest(altitude=altitude):
                controller = make_controller()
                controller.update(1.0, observation(1, 1.0, elevation=elevation), flight(altitude=altitude))
                controller.update(1.2, observation(2, 1.2, elevation=elevation), flight(altitude=altitude))
                command = controller.update(1.4, observation(3, 1.4, elevation=elevation), flight(altitude=altitude))
                self.assertEqual(command.down_m_s, 0.0)

        controller = make_controller()
        controller.update(1.0, observation(1, 1.0, elevation=20.0), flight())
        controller.update(1.2, observation(2, 1.2, elevation=20.0), flight())
        controller.update(1.4, observation(3, 1.4, elevation=20.0), flight())
        command = controller.update(1.6, observation(4, 1.6, elevation=20.0), flight(altitude=13.0))
        self.assertEqual(command.down_m_s, 0.0)

    def test_follow_distance_limit_latches(self):
        controller = make_controller()
        drive_to_follow(controller)
        command = controller.update(2.0, observation(6, 2.0), flight(distance=15.0))
        self.assertEqual(command.state, GuidanceState.TARGET_LOST_HOLD)
        self.assertEqual(command.forward_m_s, 0.0)
        command = controller.update(2.2, observation(7, 2.2), flight())
        self.assertEqual(command.state, GuidanceState.TARGET_LOST_HOLD)
        self.assertEqual(command.forward_m_s, 0.0)

    def test_manual_override_and_flight_failures_abort(self):
        failures = (
            flight(offboard=False),
            flight(fresh=False),
            flight(geofence=False),
            flight(battery=10.0),
            flight(altitude=30.0),
        )
        for failed in failures:
            with self.subTest(failed=failed):
                controller = make_controller()
                drive_to_follow(controller)
                command = controller.update(2.0, observation(6, 2.0), failed)
                self.assertEqual(command.state, GuidanceState.ABORTED)
                self.assertEqual(command.forward_m_s, 0.0)
                command = controller.update(2.2, observation(7, 2.2), flight())
                self.assertEqual(command.state, GuidanceState.ABORTED)

    def test_horizontal_command_acceleration_is_limited(self):
        controller = make_controller(max_horizontal_accel_m_s2=0.5)
        for index in range(4):
            stamp = 1.0 + index * 0.2
            controller.update(stamp, observation(index + 1, stamp), flight())
        command = controller.update(1.65, observation(4, 1.6), flight())
        self.assertAlmostEqual(command.forward_m_s, 0.025)

    def test_config_rejects_invalid_limits(self):
        with self.assertRaises(ValueError):
            GuidanceConfig(yaw_exit_deg=1.0, yaw_deadband_deg=5.0)
        with self.assertRaises(ValueError):
            GuidanceConfig(max_altitude_m=1.0, min_altitude_m=2.0)


class GuidanceTelemetryTest(unittest.TestCase):
    def test_simulated_audio_blocks_have_fresh_sequence_metadata(self):
        manager = MultiDeviceAudioManager(RigGeometry(), simulate=True)
        first = manager.get_latest_blocks()["left"]
        second = manager.get_latest_blocks()["left"]
        self.assertIsInstance(first, AudioBlock)
        self.assertEqual(first.samples.shape, (8820, 16))
        self.assertGreater(second.sequence, first.sequence)
        self.assertGreaterEqual(second.captured_at, first.captured_at)

    def test_guidance_payload_preserves_angles_quality_and_capture_identity(self):
        block = AudioBlock(np.zeros((8, 16), dtype=np.float32), captured_at=123.0, sequence=7)
        doa = DOAResult(
            array_id="left",
            local_angle_deg=10.0,
            global_angle_deg=10.0,
            prominence=2.0,
            confidence=0.9,
            spectrum=None,
            stable=True,
            elevation_deg=15.0,
            elevation_prominence=2.2,
            elevation_stable=True,
        )
        payload = build_guidance_payload(
            "uma16_front_only",
            block,
            doa,
            mount_calibration_id="test-mount",
        )
        self.assertEqual(payload["sequence"], 7)
        self.assertEqual(payload["captured_at"], 123.0)
        self.assertEqual(payload["azimuth_deg"], 10.0)
        self.assertEqual(payload["elevation_deg"], 15.0)
        self.assertEqual(payload["elevation_prominence"], 2.2)
        self.assertEqual(payload["mount_calibration_id"], "test-mount")
        self.assertIsNone(build_guidance_payload("dual", block, doa))


if __name__ == "__main__":
    unittest.main()
