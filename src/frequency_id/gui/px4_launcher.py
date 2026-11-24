"""PX4 arm/takeoff diagnostic subprocess builder, event parser, and GUI lifecycle mixin."""

from __future__ import annotations

import json
import math
import os
import queue
import signal
import subprocess
import sys
import threading
from tkinter import messagebox
from typing import Dict, List, Optional, Tuple

from frequency_id.gui.widgets import ACCENT, MUTED, OK, UNKNOWN_AMBER, WARN
from frequency_id.paths import get_project_root

PX4_URI_PRESETS = (
    "udp://:14540",
    "udp://:14550",
    "serial:///dev/ttyACM0:57600",
    "serial:///dev/ttyTHS1:921600",
)

_TERMINAL_PHASES = frozenset({"PASSED", "FAILED", "ABORTED"})


def build_px4_test_command(
    connect_uri: str,
    takeoff_alt_m: float = 2.5,
    hover_s: float = 5.0,
    auto_land: bool = True,
    arm_only: bool = False,
    virtual_rc: bool = False,
    config_path=None,
) -> List[str]:
    cmd = [
        sys.executable,
        "-u",
        str(get_project_root() / "px4_controller.py"),
        "--test-takeoff",
        "--authorize-arm",
        "--connect",
        str(connect_uri),
        "--takeoff-alt",
        str(takeoff_alt_m),
        "--hover-s",
        str(hover_s),
        "--auto-land" if auto_land else "--no-auto-land",
    ]
    if arm_only:
        cmd.append("--arm-only")
    if virtual_rc:
        cmd.append("--virtual-rc")
    if config_path is not None:
        cmd.extend(["--config", str(config_path)])
    return cmd


def parse_px4_test_line(line: str) -> Optional[Dict]:
    text = line.strip()
    if not text.startswith("@PX4_TEST:"):
        return None
    try:
        payload = json.loads(text[10:])
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict) or payload.get("type") != "px4_test":
        return None
    return payload


def format_px4_test_status(event: Dict) -> Tuple[str, str]:
    phase = str(event.get("phase", "")).upper()
    message = str(event.get("message", "")).strip()
    alt = event.get("altitude_m")
    alt_str = f" ({float(alt):.2f}m)" if isinstance(alt, (int, float)) and math.isfinite(alt) else ""

    if phase == "CONNECTING":
        return ("PX4 CONNECTING...", ACCENT)
    if phase == "PREFLIGHT_CHECK":
        return ("PX4 PREFLIGHT...", ACCENT)
    if phase == "ARMING":
        return ("PX4 ARMING...", UNKNOWN_AMBER)
    if phase == "ENGINES_RUNNING":
        return ("PX4 ENGINES RUNNING", UNKNOWN_AMBER)
    if phase == "TAKING_OFF":
        return (f"PX4 TAKING OFF{alt_str}", ACCENT)
    if phase == "AIRBORNE_HOVER":
        return (f"PX4 HOVERING{alt_str}", OK)
    if phase == "LANDING":
        return (f"PX4 LANDING{alt_str}", UNKNOWN_AMBER)
    if phase == "DISARMING":
        return ("PX4 DISARMING...", UNKNOWN_AMBER)
    if phase == "PASSED":
        if event.get("arm_only"):
            return ("PX4 ARM TEST PASSED", OK)
        return ("PX4 TEST PASSED", OK)
    if phase == "FAILED":
        short_msg = message[:140] if message else ""
        return (f"PX4 FAILED: {short_msg}" if short_msg else "PX4 TEST FAILED", WARN)
    if phase == "ABORTED":
        short_msg = message[:140] if message else ""
        return (f"PX4 ABORTED: {short_msg}" if short_msg else "PX4 TEST ABORTED", WARN)
    return (f"PX4 {phase}" if phase else "PX4 IDLE", MUTED)


def build_px4_confirm_prompt(
    connect_uri: str,
    takeoff_alt_m: float,
    auto_land: bool,
    arm_only: bool,
    virtual_rc: bool = False,
    remote_label: Optional[str] = None,
) -> str:
    mode_text = "Arm Only (Bench)" if arm_only else "Arm & Takeoff"
    auto_land_text = "Yes" if auto_land else "No (Hold Hover)"
    target_line = f"Target: {remote_label}\n" if remote_label else ""
    virtual_rc_line = (
        "Virtual RC (Bench): Yes (neutral 0,0,0,0)\n" if virtual_rc else ""
    )
    return (
        f"{target_line}"
        f"PX4 URI: {connect_uri}\n"
        f"Mode: {mode_text}\n"
        f"{virtual_rc_line}"
        f"Target Altitude: {takeoff_alt_m:.2f} m\n"
        f"Auto-Land: {auto_land_text}\n\n"
        "WARNING: Motors will spin immediately upon arming.\n"
        "Verify propellers are clear (or removed for bench test) and keep "
        "an RC transmitter ready to override at any time.\n\n"
        "Proceed with PX4 arm/takeoff test?"
    )


def sync_px4_bench_toggles(var_arm_only, var_virtual_rc, changed: str) -> None:
    """Keep Arm Only and Virtual RC bench checkboxes mutually consistent."""
    if changed == "virtual_rc" and var_virtual_rc.get():
        var_arm_only.set(True)
    elif changed == "arm_only" and not var_arm_only.get():
        var_virtual_rc.set(False)


class PX4FlightTestMixin:
    """Mixin encapsulating PX4 arm/takeoff diagnostic launch, polling, and graceful stop."""

    def on_test_arm_takeoff(self):
        if self.px4_test_process is not None:
            return

        connect_uri = self.var_px4_connect.get().strip()
        if not connect_uri:
            self.px4_status_lbl.configure(text="INVALID PX4 URI", fg=WARN)
            return

        try:
            alt = float(self.var_px4_alt.get().strip())
        except (TypeError, ValueError):
            self.px4_status_lbl.configure(text="INVALID ALT (0.5-5.0m)", fg=WARN)
            return
        if not math.isfinite(alt) or not (0.5 <= alt <= 5.0):
            self.px4_status_lbl.configure(text="INVALID ALT (0.5-5.0m)", fg=WARN)
            return

        auto_land = bool(self.var_px4_auto_land.get())
        arm_only = bool(self.var_px4_arm_only.get())
        virtual_rc = bool(getattr(self, "var_px4_virtual_rc", None) and self.var_px4_virtual_rc.get())
        if virtual_rc and not arm_only:
            self.px4_status_lbl.configure(text="VIRTUAL RC REQUIRES ARM ONLY", fg=WARN)
            return

        confirmed = messagebox.askyesno(
            "Confirm PX4 Motor Arm & Takeoff",
            build_px4_confirm_prompt(
                connect_uri=connect_uri,
                takeoff_alt_m=alt,
                auto_land=auto_land,
                arm_only=arm_only,
                virtual_rc=virtual_rc,
            ),
        )
        if not confirmed:
            return

        if hasattr(self, "save_config"):
            self.save_config()

        geo = getattr(self, "geo", None)
        px4_cfg = (
            geo.cfg.get("px4", {})
            if geo is not None and isinstance(getattr(geo, "cfg", None), dict)
            else {}
        )
        hover_s = float(px4_cfg.get("hover_duration_s", 5.0)) if isinstance(px4_cfg, dict) else 5.0
        config_path = getattr(geo, "config_path", None)

        project_dir = get_project_root()
        reports_dir = project_dir / "sim" / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        log_path = reports_dir / "px4_takeoff_test.log"

        cmd = build_px4_test_command(
            connect_uri=connect_uri,
            takeoff_alt_m=alt,
            hover_s=hover_s,
            auto_land=auto_land,
            arm_only=arm_only,
            virtual_rc=virtual_rc,
            config_path=config_path,
        )

        self._px4_terminal_event = None
        while True:
            try:
                self._px4_queue.get_nowait()
            except queue.Empty:
                break

        try:
            log_file = open(log_path, "w")
        except OSError as exc:
            self.px4_status_lbl.configure(text="PX4 LOG OPEN FAILED", fg=WARN)
            print(f"[ERROR] Could not open PX4 test log: {exc}")
            return

        try:
            process = subprocess.Popen(
                cmd,
                cwd=project_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                start_new_session=True,
            )
        except OSError as exc:
            log_file.close()
            self.px4_test_process = None
            self.px4_status_lbl.configure(text="PX4 START FAILED", fg=WARN)
            print(f"[ERROR] Could not start PX4 takeoff test: {exc}")
            return

        self.px4_test_process = process

        def reader():
            try:
                if process.stdout is not None:
                    for line in process.stdout:
                        log_file.write(line)
                        log_file.flush()
                        self._px4_queue.put(line)
            finally:
                log_file.close()

        reader_thread = threading.Thread(target=reader, daemon=True)
        self._px4_reader_thread = reader_thread
        reader_thread.start()

        self.px4_test_button.configure(state="disabled", text="PX4 TESTING...")
        stop_btn = getattr(self, "px4_stop_button", None)
        if stop_btn is not None:
            stop_btn.configure(state="normal")
        self.px4_status_lbl.configure(text="PX4 CONNECTING...", fg=ACCENT)
        self.root.after(100, self._poll_px4_test)

    def _drain_px4_queue(self):
        while True:
            try:
                line = self._px4_queue.get_nowait()
            except queue.Empty:
                break
            event = parse_px4_test_line(line)
            if event is not None:
                text, color = format_px4_test_status(event)
                self.px4_status_lbl.configure(text=text, fg=color)
                if str(event.get("phase", "")).upper() in _TERMINAL_PHASES:
                    self._px4_terminal_event = event

    def _poll_px4_test(self):
        process = self.px4_test_process
        if process is None:
            return
        self._drain_px4_queue()
        return_code = process.poll()
        if return_code is None:
            self.root.after(100, self._poll_px4_test)
            return

        reader_thread = getattr(self, "_px4_reader_thread", None)
        if reader_thread is not None and reader_thread.is_alive():
            reader_thread.join(timeout=0.5)
        self._px4_reader_thread = None
        self._drain_px4_queue()

        self.px4_test_process = None
        self.px4_test_button.configure(state="normal", text="TEST ARM & TAKEOFF")
        stop_btn = getattr(self, "px4_stop_button", None)
        if stop_btn is not None:
            stop_btn.configure(state="disabled")

        if self._px4_terminal_event is None:
            if return_code == 0:
                self.px4_status_lbl.configure(text="PX4 TEST PASSED", fg=OK)
            else:
                self.px4_status_lbl.configure(text=f"PX4 FAILED ({return_code})", fg=WARN)

    def on_stop_px4_test(self):
        process = self.px4_test_process
        if process is None or process.poll() is not None:
            return
        try:
            os.killpg(process.pid, signal.SIGINT)
        except (ProcessLookupError, OSError):
            pass
        self.px4_status_lbl.configure(text="STOPPING / LANDING...", fg=UNKNOWN_AMBER)

    def _stop_px4_test(self):
        process = self.px4_test_process
        self.px4_test_process = None
        if process is None or process.poll() is not None:
            return
        try:
            os.killpg(process.pid, signal.SIGINT)
            process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                pass
        except ProcessLookupError:
            pass
        stop_btn = getattr(self, "px4_stop_button", None)
        if stop_btn is not None:
            stop_btn.configure(state="disabled")
        test_btn = getattr(self, "px4_test_button", None)
        if test_btn is not None:
            test_btn.configure(state="normal", text="TEST ARM & TAKEOFF")
