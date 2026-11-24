import io
import json
from pathlib import Path
import queue
import sys
import tempfile
import unittest
from unittest.mock import patch

from frequency_id.gui import (
    TacticalGUI,
    build_px4_test_command,
    parse_px4_test_line,
    px4_launcher,
)
from frequency_id.gui.px4_launcher import format_px4_test_status
from frequency_id.gui.widgets import ACCENT, OK, UNKNOWN_AMBER, WARN


class Widget:
    def __init__(self):
        self.options = {}

    def configure(self, **options):
        self.options.update(options)


class FakeVar:
    def __init__(self, value):
        self._value = value

    def get(self):
        return self._value

    def set(self, value):
        self._value = value


class Process:
    def __init__(self, lines=(), return_code=None):
        self.return_code = return_code
        self.pid = 4321
        self.stdout = io.StringIO("".join(lines))

    def poll(self):
        return self.return_code

    def wait(self, timeout=None):
        return self.return_code


class Root:
    def __init__(self):
        self.scheduled = []

    def after(self, delay, callback):
        self.scheduled.append((delay, callback))


class PX4LauncherTest(unittest.TestCase):
    def make_gui(self, process=None, connect="udp://:14540", alt="2.5", auto_land=True, arm_only=False):
        gui = object.__new__(TacticalGUI)
        gui.px4_test_process = process
        gui._px4_queue = queue.Queue()
        gui._px4_terminal_event = None
        gui._px4_reader_thread = None
        gui.var_px4_connect = FakeVar(connect)
        gui.var_px4_alt = FakeVar(alt)
        gui.var_px4_auto_land = FakeVar(auto_land)
        gui.var_px4_arm_only = FakeVar(arm_only)
        gui.px4_test_button = Widget()
        gui.px4_stop_button = Widget()
        gui.px4_status_lbl = Widget()
        gui.root = Root()
        gui.save_config_calls = 0
        gui.save_config = lambda: setattr(gui, "save_config_calls", gui.save_config_calls + 1)
        return gui

    def test_build_px4_test_command_formats_flags(self):
        with patch(
            "frequency_id.gui.px4_launcher.get_project_root",
            return_value=Path("/opt/freq"),
        ):
            cmd = build_px4_test_command(
                "serial:///dev/ttyTHS1:921600",
                takeoff_alt_m=1.5,
                hover_s=4.0,
                auto_land=False,
                arm_only=True,
                config_path="/tmp/rig.json",
            )
        self.assertEqual(
            cmd,
            [
                sys.executable,
                "-u",
                "/opt/freq/px4_controller.py",
                "--test-takeoff",
                "--authorize-arm",
                "--connect",
                "serial:///dev/ttyTHS1:921600",
                "--takeoff-alt",
                "1.5",
                "--hover-s",
                "4.0",
                "--no-auto-land",
                "--arm-only",
                "--config",
                "/tmp/rig.json",
            ],
        )

    def test_parse_px4_test_line_and_format_status(self):
        self.assertIsNone(parse_px4_test_line("plain log line"))
        self.assertIsNone(parse_px4_test_line("@PX4_TEST:not-json"))
        self.assertIsNone(parse_px4_test_line('@PX4_TEST:{"type":"other"}'))

        raw = "@PX4_TEST:" + json.dumps(
            {"type": "px4_test", "phase": "TAKING_OFF", "message": "Climbing", "altitude_m": 1.42}
        )
        parsed = parse_px4_test_line(raw)
        self.assertEqual(parsed["phase"], "TAKING_OFF")
        text, color = format_px4_test_status(parsed)
        self.assertIn("1.42m", text)
        self.assertEqual(color, ACCENT)

        for phase, expected_color in (
            ("CONNECTING", ACCENT),
            ("PREFLIGHT_CHECK", ACCENT),
            ("ARMING", UNKNOWN_AMBER),
            ("ENGINES_RUNNING", UNKNOWN_AMBER),
            ("AIRBORNE_HOVER", OK),
            ("LANDING", UNKNOWN_AMBER),
            ("DISARMING", UNKNOWN_AMBER),
            ("PASSED", OK),
            ("FAILED", WARN),
            ("ABORTED", WARN),
        ):
            label, col = format_px4_test_status(
                {"type": "px4_test", "phase": phase, "message": "detail", "altitude_m": 2.5}
            )
            self.assertTrue(label)
            self.assertEqual(col, expected_color)

    def test_confirmation_dialog_cancel_prevents_subprocess_launch(self):
        gui = self.make_gui()
        with patch("frequency_id.gui.px4_launcher.messagebox.askyesno", return_value=False) as ask, patch(
            "frequency_id.gui.px4_launcher.subprocess.Popen"
        ) as popen:
            gui.on_test_arm_takeoff()
        ask.assert_called_once()
        popen.assert_not_called()
        self.assertIsNone(gui.px4_test_process)

    def test_invalid_inputs_rejected_before_confirmation(self):
        gui_bad_uri = self.make_gui(connect="   ")
        with patch("frequency_id.gui.px4_launcher.messagebox.askyesno") as ask:
            gui_bad_uri.on_test_arm_takeoff()
        ask.assert_not_called()
        self.assertEqual(gui_bad_uri.px4_status_lbl.options["fg"], WARN)

        gui_bad_alt = self.make_gui(alt="9.0")
        with patch("frequency_id.gui.px4_launcher.messagebox.askyesno") as ask:
            gui_bad_alt.on_test_arm_takeoff()
        ask.assert_not_called()
        self.assertEqual(gui_bad_alt.px4_status_lbl.options["fg"], WARN)

    def test_confirmed_launch_streams_events_disables_duplicates_and_reenables_on_exit(self):
        events = [
            "@PX4_TEST:" + json.dumps({"type": "px4_test", "phase": "ENGINES_RUNNING", "message": "Armed"}) + "\n",
            "@PX4_TEST:" + json.dumps({"type": "px4_test", "phase": "AIRBORNE_HOVER", "message": "Hover", "altitude_m": 2.48}) + "\n",
            "@PX4_TEST:" + json.dumps({"type": "px4_test", "phase": "PASSED", "message": "Done"}) + "\n",
        ]
        process = Process(lines=events, return_code=None)
        gui = self.make_gui()
        with tempfile.TemporaryDirectory() as directory, patch(
            "frequency_id.gui.px4_launcher.get_project_root", return_value=Path(directory)
        ), patch(
            "frequency_id.gui.px4_launcher.messagebox.askyesno", return_value=True
        ), patch(
            "frequency_id.gui.px4_launcher.subprocess.Popen", return_value=process
        ) as popen:
            gui.on_test_arm_takeoff()
            gui.on_test_arm_takeoff()  # duplicate click ignored
            gui._px4_reader_thread.join(timeout=2.0)

        popen.assert_called_once()
        self.assertEqual(gui.save_config_calls, 1)
        self.assertEqual(gui.px4_test_button.options["state"], "disabled")
        self.assertEqual(gui.px4_stop_button.options["state"], "normal")
        self.assertEqual(gui.root.scheduled[-1], (100, gui._poll_px4_test))

        # Poll while still running drains events
        gui._poll_px4_test()
        self.assertEqual(gui.px4_status_lbl.options["text"], "PX4 TEST PASSED")
        self.assertEqual(gui.px4_status_lbl.options["fg"], OK)
        self.assertIsNotNone(gui.px4_test_process)

        # Process exits -> button re-enabled, stop disabled
        process.return_code = 0
        gui._poll_px4_test()
        self.assertIsNone(gui.px4_test_process)
        self.assertEqual(gui.px4_test_button.options["state"], "normal")
        self.assertEqual(gui.px4_stop_button.options["state"], "disabled")

    def test_on_stop_and_close_send_sigint_to_process_group(self):
        process = Process(return_code=None)
        gui = self.make_gui(process=process)
        with patch("frequency_id.gui.px4_launcher.os.killpg") as killpg:
            gui.on_stop_px4_test()
        killpg.assert_called_once_with(process.pid, px4_launcher.signal.SIGINT)
        self.assertEqual(gui.px4_status_lbl.options["text"], "STOPPING / LANDING...")

        with patch("frequency_id.gui.px4_launcher.os.killpg") as killpg:
            gui._stop_px4_test()
        killpg.assert_called_once_with(process.pid, px4_launcher.signal.SIGINT)
        self.assertIsNone(gui.px4_test_process)
        self.assertEqual(gui.px4_stop_button.options["state"], "disabled")


if __name__ == "__main__":
    unittest.main()
