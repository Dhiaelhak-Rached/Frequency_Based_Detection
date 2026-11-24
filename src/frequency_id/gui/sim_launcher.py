"""PX4/Gazebo simulation subprocess builder and lifecycle mixin for the Tactical GUI."""

from __future__ import annotations

import os
import signal
import subprocess
import sys

from frequency_id.gui.widgets import ACCENT, MUTED, OK, WARN
from frequency_id.paths import get_project_root
from sim.harness.sitl_harness import PX4SitlProcess
from sim.sensors.real_uma16 import RealUMA16Source


def build_simulation_command():
    return [
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
    ]


class SimulationLauncherMixin:
    """Mixin encapsulating PX4/Gazebo simulation launch, polling, and teardown."""

    def _release_devices_for_simulation(self):
        vision = getattr(self, "vision", None)
        if vision is not None and getattr(vision, "is_running", False):
            vision.stop()
            self.vision_result = None
            self._sim_paused_vision = True
        else:
            self._sim_paused_vision = False

        if getattr(self, "is_active", False) and getattr(self, "audio_mgr", None) is not None:
            self.on_toggle_listening()
            self._sim_paused_audio = True
        else:
            self._sim_paused_audio = False

    def _restore_devices_after_simulation(self):
        if getattr(self, "_sim_paused_vision", False):
            self._sim_paused_vision = False
            vision = getattr(self, "vision", None)
            var_vision = getattr(self, "var_vision_enabled", None)
            if vision is not None and (var_vision is None or var_vision.get()):
                vision.start()

        if getattr(self, "_sim_paused_audio", False):
            self._sim_paused_audio = False
            if not getattr(self, "is_active", False) and getattr(self, "audio_mgr", None) is not None:
                self.on_toggle_listening()

    def on_run_simulation(self):
        if self.sim_process is not None:
            return
        if hasattr(self, "var_low"):
            self.save_config()
        self._release_devices_for_simulation()
        RealUMA16Source.kill_stale()
        project_dir = get_project_root()
        reports_dir = project_dir / "sim" / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        try:
            with open(reports_dir / "gui_simulation.log", "w") as log_file:
                self.sim_process = subprocess.Popen(
                    build_simulation_command(),
                    cwd=project_dir,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
        except OSError as exc:
            self.sim_process = None
            self._restore_devices_after_simulation()
            self.sim_status_lbl.configure(text="SIM START FAILED", fg=WARN)
            print(f"[ERROR] Could not start simulation: {exc}")
            return
        self.sim_button.configure(state="disabled", text="SIM RUNNING")
        stop_btn = getattr(self, "sim_stop_button", None)
        if stop_btn is not None:
            stop_btn.configure(state="normal")
        self.sim_status_lbl.configure(text=f"SIM PID {self.sim_process.pid}", fg=ACCENT)
        self.root.after(250, self._poll_simulation)

    def on_stop_simulation(self):
        self._stop_simulation(restore_devices=True)
        self.sim_button.configure(state="normal", text="RUN SIMULATION")
        stop_btn = getattr(self, "sim_stop_button", None)
        if stop_btn is not None:
            stop_btn.configure(state="disabled")
        self.sim_status_lbl.configure(text="SIM STOPPED", fg=MUTED)

    def _poll_simulation(self):
        process = self.sim_process
        if process is None:
            return
        return_code = process.poll()
        if return_code is None:
            self.root.after(500, self._poll_simulation)
            return
        self.sim_process = None
        RealUMA16Source.kill_stale()
        stop_btn = getattr(self, "sim_stop_button", None)
        if stop_btn is not None:
            stop_btn.configure(state="disabled")
        self._restore_devices_after_simulation()
        self.sim_button.configure(state="normal", text="RUN SIMULATION")
        if return_code == 0:
            self.sim_status_lbl.configure(text="SIM PASSED", fg=OK)
        else:
            self.sim_status_lbl.configure(text=f"SIM FAILED ({return_code})", fg=WARN)

    def _stop_simulation(self, restore_devices: bool = True):
        process = self.sim_process
        self.sim_process = None
        if process is None or process.poll() is not None:
            return
        try:
            os.killpg(process.pid, signal.SIGINT)
            process.wait(timeout=8.0)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                pass
        PX4SitlProcess.kill_stale()
        RealUMA16Source.kill_stale()
        stop_btn = getattr(self, "sim_stop_button", None)
        if stop_btn is not None:
            stop_btn.configure(state="disabled")
        if restore_devices:
            self._restore_devices_after_simulation()
