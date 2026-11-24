from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import tactical_gui


class Widget:
    def __init__(self):
        self.options = {}

    def configure(self, **options):
        self.options.update(options)


class Process:
    def __init__(self, return_code=None):
        self.return_code = return_code
        self.pid = 1234

    def poll(self):
        return self.return_code

    def wait(self, timeout):
        return self.return_code


class Root:
    def __init__(self):
        self.scheduled = []

    def after(self, delay, callback):
        self.scheduled.append((delay, callback))


class SimulationLauncherTest(unittest.TestCase):
    def make_gui(self, process):
        gui = object.__new__(tactical_gui.TacticalGUI)
        gui.sim_process = process
        gui.sim_button = Widget()
        gui.sim_status_lbl = Widget()
        gui.root = Root()
        return gui

    def test_command_runs_full_gazebo_simulation_without_shell(self):
        command = tactical_gui.build_simulation_command()
        self.assertEqual(
            command,
            [
                sys.executable,
                "-m",
                "sim.runners.single_drone_mic",
                "--low",
                "2900",
                "--high",
                "3000",
                "--threshold",
                "-55",
                "--no-classifier",
                "--duration",
                "1200",
            ],
        )

    def test_button_starts_expected_process_and_disables_duplicate_launches(self):
        process = Process()
        gui = self.make_gui(None)
        gui.sim_stop_button = Widget()
        with tempfile.TemporaryDirectory() as directory, patch(
            "tactical_gui.Path"
        ) as path_class, patch(
            "tactical_gui.subprocess.Popen", return_value=process
        ) as popen, patch.object(
            tactical_gui.RealUMA16Source, "kill_stale"
        ) as uma_kill_stale:
            path_class.return_value.resolve.return_value.parent = Path(directory)
            gui.on_run_simulation()
            gui.on_run_simulation()
        popen.assert_called_once()
        uma_kill_stale.assert_called_once_with()
        args, kwargs = popen.call_args
        self.assertEqual(args[0], tactical_gui.build_simulation_command())
        self.assertEqual(kwargs["cwd"], Path(directory))
        self.assertTrue(kwargs["start_new_session"])
        self.assertEqual(gui.sim_process, process)
        self.assertEqual(gui.sim_button.options["state"], "disabled")
        self.assertEqual(gui.sim_stop_button.options["state"], "normal")

    def test_running_process_is_polled_and_completion_reenables_button(self):
        process = Process()
        gui = self.make_gui(process)
        gui.sim_stop_button = Widget()
        gui._poll_simulation()
        self.assertEqual(gui.sim_process, process)
        self.assertEqual(gui.root.scheduled, [(500, gui._poll_simulation)])

        process.return_code = 0
        with patch.object(tactical_gui.RealUMA16Source, "kill_stale") as uma_kill_stale:
            gui._poll_simulation()
        uma_kill_stale.assert_called_once_with()
        self.assertIsNone(gui.sim_process)
        self.assertEqual(gui.sim_button.options["state"], "normal")
        self.assertEqual(gui.sim_stop_button.options["state"], "disabled")
        self.assertEqual(gui.sim_status_lbl.options["text"], "SIM PASSED")

    def test_close_interrupts_process_group(self):
        process = Process()
        gui = self.make_gui(process)
        gui.sim_stop_button = Widget()
        with patch("tactical_gui.os.killpg") as killpg, patch.object(
            tactical_gui.PX4SitlProcess, "kill_stale"
        ) as kill_stale, patch.object(
            tactical_gui.RealUMA16Source, "kill_stale"
        ) as uma_kill_stale:
            gui._stop_simulation()
        killpg.assert_called_once_with(process.pid, tactical_gui.signal.SIGINT)
        kill_stale.assert_called_once_with()
        uma_kill_stale.assert_called_once_with()
        self.assertEqual(gui.sim_stop_button.options["state"], "disabled")
        self.assertIsNone(gui.sim_process)

    def test_simulation_pauses_and_restores_gui_vision(self):
        class FakeVision:
            def __init__(self):
                self.is_running = True
                self.starts = 0
                self.stops = 0

            def start(self):
                self.is_running = True
                self.starts += 1

            def stop(self):
                self.is_running = False
                self.stops += 1

        class FakeVar:
            def __init__(self, value):
                self._value = value

            def get(self):
                return self._value

        process = Process()
        gui = self.make_gui(None)
        gui.vision = FakeVision()
        gui.var_vision_enabled = FakeVar(True)
        gui.is_active = False
        gui.audio_mgr = None

        with tempfile.TemporaryDirectory() as directory, patch(
            "tactical_gui.Path"
        ) as path_class, patch(
            "tactical_gui.subprocess.Popen", return_value=process
        ), patch.object(
            tactical_gui.RealUMA16Source, "kill_stale"
        ):
            path_class.return_value.resolve.return_value.parent = Path(directory)
            gui.on_run_simulation()

        self.assertEqual(gui.vision.stops, 1)
        self.assertFalse(gui.vision.is_running)

        process.return_code = 0
        with patch.object(tactical_gui.RealUMA16Source, "kill_stale"):
            gui._poll_simulation()
        self.assertEqual(gui.vision.starts, 1)
        self.assertTrue(gui.vision.is_running)


if __name__ == "__main__":
    unittest.main()
