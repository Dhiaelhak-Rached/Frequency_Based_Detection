import json
from types import SimpleNamespace
import unittest

from guidance import GuidanceConfig
from px4_controller import PX4FlightMonitor, parse_telemetry_line, validate_observation_metadata


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


if __name__ == "__main__":
    unittest.main()
