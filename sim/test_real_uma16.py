"""Offline unit tests for the real UMA-16 HIL bridge, visualization, and safety guards."""

import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from guidance import AcousticObservation, FlightState, GuidanceConfig, GuidanceController, GuidanceState
from px4_controller import validate_observation_metadata
from sim.real_uma16 import RealUMA16Config, RealUMA16Source
from sim.run_single_drone_mic import (
    DirectionSmoother,
    build_parser as build_single_drone_parser,
    select_single_drone_command,
)
from sim.sitl_harness import (
    DEFAULT_FOLLOW_OFFSET_XYZ,
    assert_sitl_endpoint,
    set_gazebo_follow_target,
    spawn_gazebo_motion_reference_arena,
)
from sim.visualizer import Live3DGuidanceVisualizer, SimRecord, format_desk_alignment_cue


def _make_telemetry_line(
    sequence: int,
    captured_at: float,
    az_deg: float = 15.0,
    el_deg: float = 10.0,
    az_prom: float = 2.1,
    el_prom: float = 0.85,
) -> str:
    payload = {
        "type": "telemetry",
        "guidance": {
            "array_id": "left",
            "array_mode": "uma16_front_only",
            "mount_calibration_id": "uma16-front-body-v1",
            "captured_at": captured_at,
            "sequence": sequence,
            "azimuth_deg": az_deg,
            "elevation_deg": el_deg,
            "azimuth_prominence": az_prom,
            "elevation_prominence": el_prom,
            "azimuth_stable": True,
            "elevation_stable": True,
            "error": None,
            "suppressed": False,
        },
    }
    return "@JSON:" + json.dumps(payload, separators=(",", ":"))


class RealUMA16BridgeTest(unittest.TestCase):
    def test_build_command_defaults_to_local_uma16_no_classifier(self):
        src = RealUMA16Source(RealUMA16Config())
        cmd = src.build_command()
        self.assertIn("cli", cmd)
        self.assertIn("--uma16-front-only", cmd)
        self.assertIn("--stream-json", cmd)
        self.assertIn("--no-vision", cmd)
        self.assertIn("--no-classifier", cmd)

    def test_ingest_clock_normalization_and_staleness_detection(self):
        fake_now = [1000.0]
        src = RealUMA16Source(
            RealUMA16Config(source_mode="stdin", normalize_clock=True),
            clock_fn=lambda: fake_now[0],
        )
        # Remote host uptime is only ~50 s while local monotonic clock is 1000 s
        ok = src.ingest_line(_make_telemetry_line(sequence=1, captured_at=50.0, az_deg=20.0, el_deg=0.0))
        self.assertTrue(ok)
        payload, diag = src.sample()
        self.assertEqual(diag["packets_received"], 1)
        validate_observation_metadata(payload, "uma16-front-body-v1")
        self.assertAlmostEqual(payload["captured_at"], 1000.0, delta=1e-6)

        ctrl = GuidanceController(GuidanceConfig(observation_timeout_s=0.6))
        ctrl.enable()
        flight = FlightState(
            altitude_m=5.0,
            distance_from_start_m=0.0,
            battery_percent=95.0,
            offboard_active=True,
            telemetry_fresh=True,
            geofence_ok=True,
        )
        obs = AcousticObservation.from_mapping(payload)
        cmd = ctrl.update(fake_now[0], obs, flight)
        self.assertNotEqual(cmd.state, GuidanceState.TARGET_LOST_HOLD)

        # Advance local time without advancing remote captured_at/sequence -> must fail closed as stale
        fake_now[0] += 1.0
        src.ingest_line(_make_telemetry_line(sequence=1, captured_at=50.0, az_deg=20.0, el_deg=0.0))
        stale_payload, _ = src.sample()
        cmd_stale = ctrl.update(fake_now[0], AcousticObservation.from_mapping(stale_payload), flight)
        self.assertEqual(cmd_stale.state, GuidanceState.TARGET_LOST_HOLD)

    def test_stream_reader_waits_and_stops(self):
        lines = "\n".join([
            "not json",
            _make_telemetry_line(sequence=4, captured_at=12.3, az_deg=-12.0, el_deg=6.0),
        ]) + "\n"
        src = RealUMA16Source(
            RealUMA16Config(source_mode="stdin"),
            stream=io.StringIO(lines),
        )
        src.start()
        src.wait_ready(timeout_s=2.0)
        payload, _ = src.sample()
        self.assertEqual(payload["sequence"], 4)
        self.assertAlmostEqual(payload["azimuth_deg"], -12.0)
        src.close()

    def test_visualizer_handles_real_sensor_records_without_target(self):
        rec1 = SimRecord(
            t_s=0.1,
            drone_n=0.0,
            drone_e=0.0,
            drone_alt=5.0,
            drone_yaw_deg=10.0,
            drone_vz_m_s=0.0,
            meas_az_deg=22.5,
            meas_el_deg=8.0,
            az_prominence=1.9,
            el_prominence=0.75,
            state="ALIGN_YAW",
            reason="aligning_yaw",
            cmd_yaw_rate_deg_s=12.0,
            offboard_active=True,
            telemetry_fresh=True,
            geofence_ok=True,
        )
        self.assertIn("Turn UMA-16 RIGHT", format_desk_alignment_cue(rec1))

        viz = Live3DGuidanceVisualizer(title="Real UMA-16 HIL Test", live=False)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "hil_plot.png"
            viz.render([rec1], force=True)
            viz.save(out)
            viz.close()
            self.assertTrue(out.exists())
            self.assertGreater(out.stat().st_size, 1000)

    def test_sitl_endpoint_guard_rejects_hardware_serial(self):
        with self.assertRaises(RuntimeError):
            assert_sitl_endpoint("serial:///dev/ttyACM0:921600")
        assert_sitl_endpoint("udp://:14540")

    def test_build_command_with_vision_enabled_omits_no_vision(self):
        src = RealUMA16Source(
            RealUMA16Config(
                enable_vision=True,
                camera="/dev/video0",
                camera_fov=75.0,
                no_tracker=False,
            )
        )
        cmd = src.build_command()
        self.assertNotIn("--no-vision", cmd)
        self.assertIn("--camera", cmd)
        self.assertIn("/dev/video0", cmd)
        self.assertIn("--camera-fov", cmd)
        self.assertIn("75.0", cmd)

    def test_ingest_fused_vision_during_audio_waiting(self):
        fake_now = [250.0]
        src = RealUMA16Source(
            RealUMA16Config(source_mode="stdin", enable_vision=True),
            clock_fn=lambda: fake_now[0],
        )
        line = "@JSON:" + json.dumps(
            {
                "type": "telemetry",
                "audio_waiting": True,
                "array_mode": "uma16_front_only",
                "fused": {
                    "mode": "VISION_TRACKING",
                    "source": "VISION",
                    "bearing_deg": 335.0,
                    "elevation_deg": 12.0,
                },
                "camera": {
                    "status": "DRONE TRACKING",
                    "target_detected": True,
                    "bearing_deg": 335.0,
                    "elevation_deg": 12.0,
                },
                "guidance": None,
            },
            separators=(",", ":"),
        )
        self.assertTrue(src.ingest_line(line))
        guidance_payload, fused_payload, camera_payload, diag = src.sample_fused()
        self.assertIsNone(guidance_payload)
        self.assertEqual(fused_payload["source"], "VISION")
        self.assertEqual(camera_payload["status"], "DRONE TRACKING")
        self.assertEqual(diag["last_fused_received_at"], 250.0)

    def test_select_single_drone_command_vision_priority_and_mic_fallback(self):
        now = 100.0
        guidance_payload = {
            "array_id": "left",
            "array_mode": "uma16_front_only",
            "mount_calibration_id": "uma16-front-body-v1",
            "captured_at": 99.9,
            "sequence": 7,
            "azimuth_deg": 45.0,
            "elevation_deg": 0.0,
            "azimuth_prominence": 2.2,
            "elevation_prominence": 0.9,
            "azimuth_stable": True,
            "elevation_stable": True,
            "error": None,
            "suppressed": False,
        }
        fused_vision = {
            "mode": "VISION_TRACKING",
            "source": "VISION",
            "bearing_deg": 330.0,
            "elevation_deg": 10.0,
        }
        camera_payload = {"status": "DRONE TRACKING"}

        # 1. Vision active & fresh -> camera direction (-30 deg left, +10 deg up) wins over +45 deg mic
        cmd_v, src_v, az_v, el_v, detail_v = select_single_drone_command(
            guidance_payload=guidance_payload,
            fused_payload=fused_vision,
            camera_payload=camera_payload,
            now=now,
            last_fused_received_at=99.8,
            north_m=0.0,
            east_m=0.0,
            alt_m=5.0,
            yaw_deg=0.0,
        )
        self.assertTrue(cmd_v.active)
        self.assertEqual(src_v, "VISION")
        self.assertAlmostEqual(az_v, -30.0)
        self.assertAlmostEqual(el_v, 10.0)
        self.assertAlmostEqual(cmd_v.target_yaw_deg, -30.0)
        self.assertGreater(cmd_v.north_m_s, 0.0)
        self.assertLess(cmd_v.east_m_s, 0.0)
        self.assertLess(cmd_v.right_m_s, 0.0)
        self.assertLess(cmd_v.down_m_s, 0.0)
        self.assertEqual(detail_v, "DRONE TRACKING")

        # 2. Camera loses target (source == SOUND) -> falls back to UMA-16 mic (+45 deg right)
        cmd_m, src_m, az_m, el_m, _ = select_single_drone_command(
            guidance_payload=guidance_payload,
            fused_payload={"mode": "ACOUSTIC_TRACKING", "source": "SOUND", "bearing_deg": 45.0, "elevation_deg": 0.0},
            camera_payload={"status": "SEARCHING"},
            now=now,
            last_fused_received_at=99.8,
            north_m=0.0,
            east_m=0.0,
            alt_m=5.0,
            yaw_deg=0.0,
        )
        self.assertTrue(cmd_m.active)
        self.assertEqual(src_m, "MIC")
        self.assertAlmostEqual(az_m, 45.0)
        self.assertAlmostEqual(cmd_m.target_yaw_deg, 45.0)
        self.assertGreater(cmd_m.north_m_s, 0.0)
        self.assertGreater(cmd_m.east_m_s, 0.0)
        self.assertGreater(cmd_m.right_m_s, 0.0)

        # 3. Vision telemetry goes stale -> falls back to mic
        cmd_stale, src_stale, _, _, _ = select_single_drone_command(
            guidance_payload=guidance_payload,
            fused_payload=fused_vision,
            camera_payload=camera_payload,
            now=now,
            last_fused_received_at=98.0,
            north_m=0.0,
            east_m=0.0,
            alt_m=5.0,
            yaw_deg=0.0,
            vision_timeout_s=0.8,
        )
        self.assertEqual(src_stale, "MIC")
        self.assertTrue(cmd_stale.active)

        # 4. Geofence radius respected in vision mode
        cmd_fence, src_fence, _, _, _ = select_single_drone_command(
            guidance_payload=None,
            fused_payload={"mode": "VISION_TRACKING", "source": "VISION", "bearing_deg": 0.0, "elevation_deg": 0.0},
            camera_payload=camera_payload,
            now=now,
            last_fused_received_at=99.9,
            north_m=25.0,
            east_m=0.0,
            alt_m=5.0,
            yaw_deg=0.0,
            max_radius_m=25.0,
        )
        self.assertEqual(src_fence, "VISION")
        self.assertAlmostEqual(cmd_fence.forward_m_s, 0.0)
        self.assertAlmostEqual(cmd_fence.right_m_s, 0.0)

    def test_single_drone_mic_default_takeoff_altitude_is_10m(self):
        args = build_single_drone_parser().parse_args([])
        self.assertAlmostEqual(args.hover_alt, 10.0)

    def test_set_gazebo_follow_target_configures_third_person_follow(self):
        ok_proc = MagicMock(returncode=0, stdout="data: true\n")
        with patch("sim.sitl_harness.subprocess.run", side_effect=[ok_proc, ok_proc]) as run_mock, patch(
            "sim.sitl_harness.time.sleep"
        ) as sleep_mock:
            self.assertTrue(set_gazebo_follow_target("x500_0"))
        self.assertEqual(run_mock.call_count, 2)
        sleep_mock.assert_called_once_with(0.15)
        self.assertEqual(DEFAULT_FOLLOW_OFFSET_XYZ, (-1.8, 0.0, 0.55))
        self.assertIn("/gui/follow", run_mock.call_args_list[0].args[0])
        self.assertIn('data: "x500_0"', run_mock.call_args_list[0].args[0])
        self.assertIn("/gui/follow/offset", run_mock.call_args_list[1].args[0])
        self.assertIn("x: -1.80, y: 0.00, z: 0.55", run_mock.call_args_list[1].args[0])

        fail_proc = MagicMock(returncode=0, stdout="data: false\n")
        with patch("sim.sitl_harness.subprocess.run", return_value=fail_proc):
            self.assertFalse(set_gazebo_follow_target("x500_0"))

    def test_mic_fallback_works_when_elevation_is_none(self):
        guidance_payload = {
            "array_id": "left",
            "array_mode": "uma16_front_only",
            "mount_calibration_id": "uma16-front-body-v1",
            "captured_at": 99.9,
            "sequence": 11,
            "azimuth_deg": 35.0,
            "elevation_deg": None,
            "azimuth_prominence": 2.0,
            "elevation_prominence": 0.4,
            "azimuth_stable": True,
            "elevation_stable": False,
            "error": None,
            "suppressed": False,
        }
        cmd, src, az, el, detail = select_single_drone_command(
            guidance_payload=guidance_payload,
            fused_payload=None,
            camera_payload={"status": "SEARCHING"},
            now=100.0,
            last_fused_received_at=None,
            north_m=0.0,
            east_m=0.0,
            alt_m=10.0,
            yaw_deg=0.0,
        )
        self.assertTrue(cmd.active)
        self.assertEqual(src, "MIC")
        self.assertAlmostEqual(az, 35.0)
        self.assertAlmostEqual(el, 0.0)
        self.assertAlmostEqual(cmd.down_m_s, 0.0)
        self.assertGreater(cmd.east_m_s, 0.0)
        self.assertAlmostEqual(cmd.target_yaw_deg, 35.0)
        self.assertEqual(detail, "acoustic-az-only")

    def test_direction_smoother_wraps_angles_and_resets_on_source_change(self):
        smoother = DirectionSmoother(alpha=0.5, reset_after_s=0.6)
        az1, el1 = smoother.update(10.0, "VISION", 175.0, 5.0)
        self.assertAlmostEqual(az1, 175.0)
        self.assertAlmostEqual(el1, 5.0)

        # Crossing +/-180 deg boundary moves smoothly across -180 instead of jumping via 0
        az2, el2 = smoother.update(10.1, "VISION", -175.0, 15.0)
        self.assertAlmostEqual(abs(az2), 180.0, places=5)
        self.assertAlmostEqual(el2, 10.0)

        # Source change resets immediately to the new source angle
        az3, el3 = smoother.update(10.2, "MIC", 30.0, -4.0)
        self.assertAlmostEqual(az3, 30.0)
        self.assertAlmostEqual(el3, -4.0)

    def test_spawn_gazebo_motion_reference_arena_creates_visual_only_landmarks(self):
        ok_proc = MagicMock(returncode=0, stdout="data: true\n")
        with patch("sim.sitl_harness.subprocess.run", return_value=ok_proc) as run_mock:
            self.assertTrue(spawn_gazebo_motion_reference_arena("default", hover_alt_m=10.0))
        run_mock.assert_called_once()
        cmd = run_mock.call_args.args[0]
        self.assertIn("/world/default/create", cmd)
        req_arg = cmd[cmd.index("--req") + 1]
        self.assertIn("motion_reference_arena", req_arg)
        self.assertIn("<static>true</static>", req_arg)
        self.assertNotIn("<collision", req_arg)


if __name__ == "__main__":
    unittest.main()
