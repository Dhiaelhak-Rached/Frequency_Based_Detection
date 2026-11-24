import asyncio
import contextlib
import io
import json
import os
import signal
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from frequency_id.control import px4
from frequency_id.control.guidance import GuidanceConfig
from frequency_id.control.px4 import (
    PX4FlightMonitor,
    TakeoffTestConfig,
    build_parser,
    main,
    parse_telemetry_line,
    run_takeoff_test,
    validate_observation_metadata,
)


class TelemetryParsingTest(unittest.TestCase):
    def test_extracts_only_guidance_observations(self):
        observation = {"sequence": 3, "captured_at": 10.0}
        line = "@JSON:" + json.dumps({"type": "telemetry", "guidance": observation})
        self.assertEqual(parse_telemetry_line(line), observation)

    def test_ignores_plain_malformed_and_non_guidance_lines(self):
        self.assertIsNone(parse_telemetry_line("MODE: SEARCHING"))
        self.assertIsNone(parse_telemetry_line("@JSON:not-json"))
        self.assertIsNone(parse_telemetry_line("@JSON:{\"type\":\"other\"}"))
        self.assertIsNone(parse_telemetry_line("@JSON:{\"type\":\"telemetry\"}"))

    def test_rejects_wrong_mode_or_mount_calibration(self):
        observation = {
            "array_mode": "uma16_front_only",
            "array_id": "left",
            "mount_calibration_id": "mount-a",
        }
        validate_observation_metadata(observation, "mount-a")
        with self.assertRaises(ValueError):
            validate_observation_metadata(observation, "mount-b")
        observation["array_mode"] = "dual"
        with self.assertRaises(ValueError):
            validate_observation_metadata(observation, "mount-a")


class FlightMonitorTest(unittest.TestCase):
    def test_builds_bounded_flight_state_from_px4_telemetry(self):
        config = GuidanceConfig(geofence_radius_m=5.0)
        monitor = PX4FlightMonitor(None, config)
        monitor.flight_mode = SimpleNamespace(name="OFFBOARD")
        monitor.position = SimpleNamespace(relative_altitude_m=10.0)
        monitor.start_position = SimpleNamespace(north_m=0.0, east_m=0.0)
        monitor.local_position = SimpleNamespace(north_m=3.0, east_m=4.0)
        monitor.battery_percent = 75.0
        monitor.received_at = {name: 10.0 for name in ("flight_mode", "position", "local_position", "battery")}

        state = monitor.flight_state(10.5)
        self.assertTrue(state.offboard_active)
        self.assertTrue(state.telemetry_fresh)
        self.assertEqual(state.distance_from_start_m, 5.0)
        self.assertEqual(state.altitude_m, 10.0)
        self.assertTrue(state.geofence_ok)

        monitor.local_position = SimpleNamespace(north_m=4.0, east_m=4.0)
        state = monitor.flight_state(10.5)
        self.assertFalse(state.geofence_ok)
        state = monitor.flight_state(13.0)
        self.assertFalse(state.telemetry_fresh)


class FakeDrone:
    """Async fake MAVSDK System double for standalone takeoff tests."""

    def __init__(
        self,
        *,
        battery_percent: float = 85.0,
        health=None,
        status_text: str = "",
        takeoff_mode: str = "succeed",
        arm_succeeds: bool = True,
        on_hover=None,
    ):
        self.connected_address = None
        self.actions = []
        self.armed = False
        self.in_air = False
        self.altitude_m = 0.0
        self.vz_m_s = 0.0
        self.flight_mode = "HOLD"
        self.battery_percent = battery_percent
        self.status_text = status_text
        self.target_takeoff_alt = 2.5
        self.takeoff_mode = takeoff_mode
        self.arm_succeeds = arm_succeeds
        self.on_hover = on_hover
        self._hover_triggered = False
        self.health = health or SimpleNamespace(
            is_gyrometer_calibration_ok=True,
            is_accelerometer_calibration_ok=True,
            is_magnetometer_calibration_ok=True,
            is_local_position_ok=True,
            is_global_position_ok=True,
            is_home_position_ok=True,
            is_armable=True,
        )
        self.core = self._Core(self)
        self.telemetry = self._Telemetry(self)
        self.action = self._Action(self)

    async def connect(self, system_address=None):
        self.connected_address = system_address

    class _Core:
        def __init__(self, parent):
            self.parent = parent

        async def connection_state(self):
            yield SimpleNamespace(is_connected=True)

    class _Telemetry:
        def __init__(self, parent):
            self.parent = parent

        async def armed(self):
            while True:
                yield self.parent.armed
                await asyncio.sleep(0.005)

        async def in_air(self):
            while True:
                yield self.parent.in_air
                await asyncio.sleep(0.005)

        async def landed_state(self):
            while True:
                state = "IN_AIR" if self.parent.in_air else "ON_GROUND"
                yield SimpleNamespace(name=state)
                await asyncio.sleep(0.005)

        async def position_velocity_ned(self):
            while True:
                yield SimpleNamespace(
                    position=SimpleNamespace(north_m=0.0, east_m=0.0, down_m=-self.parent.altitude_m),
                    velocity=SimpleNamespace(north_m_s=0.0, east_m_s=0.0, down_m_s=self.parent.vz_m_s),
                )
                await asyncio.sleep(0.005)

        async def flight_mode(self):
            while True:
                yield SimpleNamespace(name=self.parent.flight_mode)
                await asyncio.sleep(0.005)

        async def battery(self):
            while True:
                yield SimpleNamespace(remaining_percent=self.parent.battery_percent)
                await asyncio.sleep(0.005)

        async def health(self):
            while True:
                yield self.parent.health
                await asyncio.sleep(0.005)

        async def status_text(self):
            if self.parent.status_text:
                yield SimpleNamespace(text=self.parent.status_text)
            while True:
                await asyncio.sleep(0.05)

    class _Action:
        def __init__(self, parent):
            self.parent = parent

        async def arm(self):
            self.parent.actions.append("arm")
            if not self.parent.arm_succeeds:
                raise RuntimeError("Command denied")
            self.parent.armed = True

        async def disarm(self):
            self.parent.actions.append("disarm")
            self.parent.armed = False

        async def set_takeoff_altitude(self, alt):
            self.parent.actions.append(("set_takeoff_altitude", float(alt)))
            self.parent.target_takeoff_alt = float(alt)

        async def takeoff(self):
            self.parent.actions.append("takeoff")
            if self.parent.takeoff_mode == "stall_ground":
                return
            if self.parent.takeoff_mode == "stall_airborne":
                self.parent.in_air = True
                self.parent.altitude_m = 0.2
                self.parent.vz_m_s = -0.1
                return
            self.parent.flight_mode = "TAKEOFF"
            self.parent.in_air = True
            self.parent.altitude_m = self.parent.target_takeoff_alt * 0.5
            self.parent.vz_m_s = -0.8

            async def finish_climb():
                await asyncio.sleep(0.02)
                self.parent.altitude_m = self.parent.target_takeoff_alt
                self.parent.vz_m_s = 0.0
                self.parent.flight_mode = "HOLD"
                if self.parent.on_hover is not None and not self.parent._hover_triggered:
                    self.parent._hover_triggered = True
                    await asyncio.sleep(0.03)
                    self.parent.on_hover(self.parent)

            asyncio.create_task(finish_climb())

        async def land(self):
            self.parent.actions.append("land")
            self.parent.flight_mode = "LAND"
            await asyncio.sleep(0.01)
            self.parent.altitude_m = 0.0
            self.parent.vz_m_s = 0.0
            self.parent.in_air = False
            self.parent.armed = False

        async def kill(self):
            raise AssertionError("action.kill() must never be called")


def _parse_events(stdout_text: str):
    events = []
    for line in stdout_text.splitlines():
        if line.startswith("@PX4_TEST:"):
            events.append(json.loads(line[10:]))
    return events


def _make_args(config_path, **overrides):
    defaults = {
        "connect": "udp://:14540",
        "config": str(config_path),
        "connect_timeout_s": 2.0,
        "takeoff_timeout_s": 1.0,
        "authorize_offboard": False,
        "enable_motion": False,
        "test_takeoff": True,
        "authorize_arm": True,
        "arm_only": False,
        "takeoff_alt": None,
        "hover_s": 0.08,
        "auto_land": True,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


class PX4TakeoffTest(unittest.TestCase):
    def _write_config(self, directory, payload):
        path = os.path.join(directory, "rig_config.json")
        with open(path, "w") as f:
            json.dump(payload, f)
        return path

    def _run_with_capture(self, coro):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = asyncio.run(coro)
        return rc, _parse_events(buf.getvalue())

    def test_takeoff_test_config_validation(self):
        cfg = TakeoffTestConfig(takeoff_altitude_m=2.5, hover_duration_s=5.0)
        self.assertEqual(cfg.takeoff_altitude_m, 2.5)
        with self.assertRaises(ValueError):
            TakeoffTestConfig(takeoff_altitude_m=0.4)
        with self.assertRaises(ValueError):
            TakeoffTestConfig(takeoff_altitude_m=5.5)
        with self.assertRaises(ValueError):
            TakeoffTestConfig(takeoff_altitude_m=3.5, guidance_max_altitude_m=3.0)
        with self.assertRaises(ValueError):
            TakeoffTestConfig(takeoff_altitude_m=float("nan"))
        with self.assertRaises(ValueError):
            TakeoffTestConfig(hover_duration_s=31.0)
        with self.assertRaises(ValueError):
            TakeoffTestConfig(connect_timeout_s=0.0)
        with self.assertRaises(ValueError):
            TakeoffTestConfig(takeoff_timeout_s=-1.0)
        with self.assertRaises(ValueError):
            TakeoffTestConfig(min_battery_percent=101.0)

    def test_refuses_without_authorize_arm(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_path = self._write_config(tmpdir, {"guidance": {"max_altitude_m": 20.0}})
            args = _make_args(cfg_path, authorize_arm=False)
            with self.assertRaises(ValueError):
                asyncio.run(run_takeoff_test(args, drone_factory=lambda: FakeDrone()))

            err_buf = io.StringIO()
            with contextlib.redirect_stderr(err_buf):
                rc = main(["--test-takeoff", "--config", cfg_path])
            self.assertEqual(rc, 2)
            self.assertIn("--authorize-arm", err_buf.getvalue())

    def test_rejects_invalid_altitude_and_low_battery_before_arm(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_path = self._write_config(tmpdir, {"guidance": {"max_altitude_m": 20.0}})
            bad_args = _make_args(cfg_path, takeoff_alt=6.0)
            with self.assertRaises(ValueError):
                asyncio.run(run_takeoff_test(bad_args, drone_factory=lambda: FakeDrone()))

            drone = FakeDrone(battery_percent=20.0)
            args = _make_args(cfg_path, takeoff_alt=2.0)
            with patch.object(px4, "TAKEOFF_POLL_INTERVAL_S", 0.01):
                rc, events = self._run_with_capture(
                    run_takeoff_test(args, drone_factory=lambda: drone)
                )
            self.assertNotEqual(rc, 0)
            self.assertEqual(drone.actions, [])
            self.assertEqual(events[-1]["phase"], "FAILED")
            self.assertIn("low_battery", events[-1]["message"])

    def test_unreported_negative_battery_passes_preflight_while_low_battery_fails(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_path = self._write_config(tmpdir, {"guidance": {"max_altitude_m": 20.0}})
            # -1.0 means unknown/unreported on hardware without a calibrated power module
            drone_unknown = FakeDrone(battery_percent=-1.0)
            args = _make_args(cfg_path, takeoff_alt=2.0, hover_s=0.05)
            with patch.object(px4, "TAKEOFF_POLL_INTERVAL_S", 0.01), patch.object(
                px4, "TAKEOFF_EMIT_INTERVAL_S", 0.01
            ):
                rc, events = self._run_with_capture(
                    run_takeoff_test(args, drone_factory=lambda: drone_unknown)
                )
            self.assertEqual(rc, 0)
            self.assertEqual(events[-1]["phase"], "PASSED")
            self.assertIn("arm", drone_unknown.actions)

            # Explicit low battery (15.0%) still fails preflight without arming
            drone_low = FakeDrone(battery_percent=15.0)
            with patch.object(px4, "TAKEOFF_POLL_INTERVAL_S", 0.01):
                rc_low, events_low = self._run_with_capture(
                    run_takeoff_test(args, drone_factory=lambda: drone_low)
                )
            self.assertNotEqual(rc_low, 0)
            self.assertEqual(drone_low.actions, [])
            self.assertEqual(events_low[-1]["phase"], "FAILED")
            self.assertIn("low_battery", events_low[-1]["message"])

    def test_unarmable_health_reports_flags_and_status_text_without_arming(self):
        unhealthy = SimpleNamespace(
            is_gyrometer_calibration_ok=True,
            is_accelerometer_calibration_ok=True,
            is_magnetometer_calibration_ok=False,
            is_local_position_ok=False,
            is_global_position_ok=True,
            is_home_position_ok=False,
            is_armable=False,
        )
        drone = FakeDrone(
            health=unhealthy,
            status_text="Preflight Fail: Mag 0 uncalibrated",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_path = self._write_config(tmpdir, {"guidance": {"max_altitude_m": 20.0}})
            args = _make_args(cfg_path)
            with patch.object(px4, "TAKEOFF_POLL_INTERVAL_S", 0.01):
                rc, events = self._run_with_capture(
                    run_takeoff_test(args, drone_factory=lambda: drone)
                )
        self.assertNotEqual(rc, 0)
        self.assertNotIn("arm", drone.actions)
        failed = events[-1]
        self.assertEqual(failed["phase"], "FAILED")
        for expected in ("magnetometer", "local_position", "home_position", "armable"):
            self.assertIn(expected, failed["message"])
        self.assertIn("Preflight Fail: Mag 0 uncalibrated", failed["message"])
        self.assertEqual(failed["px4_status_text"], "Preflight Fail: Mag 0 uncalibrated")

    def test_arm_only_flow_arms_and_disarms_without_takeoff(self):
        # Also test missing "px4" section fallback and arm_only not requiring local/home position
        bench_health = SimpleNamespace(
            is_gyrometer_calibration_ok=True,
            is_accelerometer_calibration_ok=True,
            is_magnetometer_calibration_ok=True,
            is_local_position_ok=False,
            is_global_position_ok=False,
            is_home_position_ok=False,
            is_armable=True,
        )
        drone = FakeDrone(health=bench_health)
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_path = self._write_config(tmpdir, {})
            args = _make_args(cfg_path, arm_only=True)
            with patch.object(px4, "ARM_ONLY_HOLD_S", 0.04), patch.object(
                px4, "TAKEOFF_POLL_INTERVAL_S", 0.01
            ):
                rc, events = self._run_with_capture(
                    run_takeoff_test(args, drone_factory=lambda: drone)
                )
        self.assertEqual(rc, 0)
        self.assertEqual(drone.actions, ["arm", "disarm"])
        phases = [e["phase"] for e in events]
        self.assertIn("ENGINES_RUNNING", phases)
        self.assertIn("DISARMING", phases)
        self.assertEqual(phases[-1], "PASSED")
        self.assertTrue(events[-1]["engines_started"])
        self.assertFalse(events[-1]["takeoff_succeeded"])
        self.assertTrue(events[-1]["arm_only"])

    def test_full_takeoff_hover_and_auto_land_flow(self):
        drone = FakeDrone()
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_path = self._write_config(
                tmpdir,
                {
                    "guidance": {"max_altitude_m": 20.0},
                    "px4": {
                        "connect": "udp://:14540",
                        "takeoff_altitude_m": 2.5,
                        "hover_duration_s": 0.08,
                        "auto_land": True,
                    },
                },
            )
            args = _make_args(cfg_path, takeoff_alt=None, hover_s=None, auto_land=None)
            with patch.object(px4, "TAKEOFF_POLL_INTERVAL_S", 0.01), patch.object(
                px4, "TAKEOFF_EMIT_INTERVAL_S", 0.01
            ):
                rc, events = self._run_with_capture(
                    run_takeoff_test(args, drone_factory=lambda: drone)
                )
        self.assertEqual(rc, 0)
        self.assertEqual(
            drone.actions,
            ["arm", ("set_takeoff_altitude", 2.5), "takeoff", "land"],
        )
        phases = [e["phase"] for e in events]
        for required_phase in (
            "CONNECTING",
            "PREFLIGHT_CHECK",
            "ARMING",
            "ENGINES_RUNNING",
            "TAKING_OFF",
            "AIRBORNE_HOVER",
            "LANDING",
            "PASSED",
        ):
            self.assertIn(required_phase, phases)
        self.assertEqual(phases[-1], "PASSED")
        self.assertTrue(events[-1]["engines_started"])
        self.assertTrue(events[-1]["takeoff_succeeded"])

    def test_takeoff_no_auto_land_holds_hover(self):
        drone = FakeDrone()
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_path = self._write_config(tmpdir, {})
            args = _make_args(cfg_path, takeoff_alt=2.0, hover_s=0.05, auto_land=False)
            with patch.object(px4, "TAKEOFF_POLL_INTERVAL_S", 0.01), patch.object(
                px4, "TAKEOFF_EMIT_INTERVAL_S", 0.01
            ):
                rc, events = self._run_with_capture(
                    run_takeoff_test(args, drone_factory=lambda: drone)
                )
        self.assertEqual(rc, 0)
        self.assertNotIn("land", drone.actions)
        self.assertEqual(events[-1]["phase"], "PASSED")
        self.assertTrue(events[-1]["hovering"])

    def test_takeoff_timeout_recovers_safely_on_ground_and_airborne(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_path = self._write_config(tmpdir, {})
            # Ground stall -> disarm
            ground_drone = FakeDrone(takeoff_mode="stall_ground")
            args = _make_args(cfg_path, takeoff_timeout_s=0.08)
            with patch.object(px4, "TAKEOFF_POLL_INTERVAL_S", 0.01), patch.object(
                px4, "TAKEOFF_EMIT_INTERVAL_S", 0.02
            ):
                rc, events = self._run_with_capture(
                    run_takeoff_test(args, drone_factory=lambda: ground_drone)
                )
            self.assertNotEqual(rc, 0)
            self.assertIn("disarm", ground_drone.actions)
            self.assertEqual(events[-1]["phase"], "FAILED")

            # Airborne stall -> land
            airborne_drone = FakeDrone(takeoff_mode="stall_airborne")
            with patch.object(px4, "TAKEOFF_POLL_INTERVAL_S", 0.01), patch.object(
                px4, "TAKEOFF_EMIT_INTERVAL_S", 0.02
            ):
                rc, events = self._run_with_capture(
                    run_takeoff_test(args, drone_factory=lambda: airborne_drone)
                )
            self.assertNotEqual(rc, 0)
            self.assertIn("land", airborne_drone.actions)
            self.assertEqual(events[-1]["phase"], "FAILED")

    def test_cancel_sigint_mid_hover_triggers_land_and_emits_aborted(self):
        def send_sigint(_drone):
            os.kill(os.getpid(), signal.SIGINT)

        drone = FakeDrone(on_hover=send_sigint)
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_path = self._write_config(tmpdir, {})
            args = _make_args(cfg_path, hover_s=2.0, auto_land=False)
            with patch.object(px4, "TAKEOFF_POLL_INTERVAL_S", 0.01), patch.object(
                px4, "TAKEOFF_EMIT_INTERVAL_S", 0.01
            ):
                rc, events = self._run_with_capture(
                    run_takeoff_test(args, drone_factory=lambda: drone)
                )
        self.assertNotEqual(rc, 0)
        self.assertIn("land", drone.actions)
        phases = [e["phase"] for e in events]
        self.assertIn("AIRBORNE_HOVER", phases)
        self.assertIn("LANDING", phases)
        self.assertEqual(phases[-1], "ABORTED")

    def test_manual_rc_takeover_stops_commanding_and_emits_aborted(self):
        def switch_to_posctl(d):
            d.flight_mode = "POSCTL"

        drone = FakeDrone(on_hover=switch_to_posctl)
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_path = self._write_config(tmpdir, {})
            args = _make_args(cfg_path, hover_s=2.0, auto_land=True)
            with patch.object(px4, "TAKEOFF_POLL_INTERVAL_S", 0.01), patch.object(
                px4, "TAKEOFF_EMIT_INTERVAL_S", 0.01
            ):
                rc, events = self._run_with_capture(
                    run_takeoff_test(args, drone_factory=lambda: drone)
                )
        self.assertNotEqual(rc, 0)
        self.assertNotIn("land", drone.actions)
        self.assertEqual(events[-1]["phase"], "ABORTED")
        self.assertIn("POSCTL", events[-1]["message"])

    def test_cli_parser_backwards_compatibility_and_new_flags(self):
        parser = build_parser()
        legacy = parser.parse_args(["--authorize-offboard", "--enable-motion"])
        self.assertTrue(legacy.authorize_offboard)
        self.assertTrue(legacy.enable_motion)
        self.assertFalse(legacy.test_takeoff)

        diag = parser.parse_args([
            "--test-takeoff",
            "--authorize-arm",
            "--arm-only",
            "--takeoff-alt",
            "1.8",
            "--hover-s",
            "3.0",
            "--no-auto-land",
        ])
        self.assertTrue(diag.test_takeoff)
        self.assertTrue(diag.authorize_arm)
        self.assertTrue(diag.arm_only)
        self.assertEqual(diag.takeoff_alt, 1.8)
        self.assertEqual(diag.hover_s, 3.0)
        self.assertIs(diag.auto_land, False)


if __name__ == "__main__":
    unittest.main()
