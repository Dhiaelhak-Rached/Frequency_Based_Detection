#!/usr/bin/env python3
"""Laptop-side remote control and configuration CLI for the Jetson Orin rig.

Allows syncing code, inspecting audio/camera devices, editing Jetson
rig_config.json, running tests, and streaming live `main.py cli` output over SSH.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import pty
import select
import shlex
import signal
import subprocess
import sys
from typing import Any, Callable, Dict, List, Optional, Sequence


DEFAULT_HOST = "41.228.129.118"
DEFAULT_PORT = 2222
DEFAULT_USER = "meltem"
DEFAULT_REMOTE_DIR = "/home/meltem/frequency_id"
DEFAULT_LOG_PATH = "/tmp/frequency_id_cli.log"
LOCAL_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class JetsonTarget:
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    user: str = DEFAULT_USER
    remote_dir: str = DEFAULT_REMOTE_DIR

    @property
    def destination(self) -> str:
        return f"{self.user}@{self.host}"


def build_ssh_command(
    target: JetsonTarget,
    remote_cmd: str,
    tty: bool = False,
) -> List[str]:
    """Build an OpenSSH command targeting the Jetson."""
    cmd = ["ssh"]
    if tty:
        cmd.append("-tt")
    cmd.extend([
        "-o", "ConnectTimeout=10",
        "-o", "StrictHostKeyChecking=accept-new",
        "-p", str(target.port),
        target.destination,
        remote_cmd,
    ])
    return cmd


def build_rsync_command(
    target: JetsonTarget,
    local_dir: Path = LOCAL_ROOT,
    include_config: bool = False,
    dry_run: bool = False,
) -> List[str]:
    """Build an rsync command to sync the project to the Jetson."""
    cmd = [
        "rsync",
        "-avz",
        "-e",
        f"ssh -p {target.port} -o StrictHostKeyChecking=accept-new",
        "--exclude", ".git",
        "--exclude", "__pycache__",
        "--exclude", "*.pyc",
        "--exclude", ".venv",
        "--exclude", ".pytest_cache",
        "--exclude", "pnmap_scan_results.txt",
    ]
    if not include_config:
        cmd.extend(["--exclude", "rig_config.json"])
    if dry_run:
        cmd.append("--dry-run")
    src = str(local_dir).rstrip("/") + "/"
    dst = f"{target.destination}:{target.remote_dir.rstrip('/')}/"
    cmd.extend([src, dst])
    return cmd


def build_config_patch(args: argparse.Namespace) -> Dict[str, Any]:
    """Validate CLI config flags locally and return a structured patch dict."""
    patch: Dict[str, Any] = {}

    if args.dev_front is not None:
        patch["dev_front"] = int(args.dev_front)
    if args.dev_left is not None:
        patch["dev_left"] = int(args.dev_left)

    if args.low is not None or args.high is not None:
        if args.low is not None and args.high is not None and args.low >= args.high:
            raise ValueError("--low must be strictly less than --high")
        if args.low is not None:
            patch["low"] = float(args.low)
        if args.high is not None:
            patch["high"] = float(args.high)

    if args.threshold is not None:
        patch["threshold"] = float(args.threshold)
    if getattr(args, "average", None) is not None:
        if int(args.average) < 1:
            raise ValueError("--average must be at least 1")
        patch["average"] = int(args.average)

    if args.camera is not None:
        if not str(args.camera).strip():
            raise ValueError("--camera cannot be empty")
        patch["camera"] = str(args.camera).strip()

    if args.camera_fov is not None:
        fov = float(args.camera_fov)
        if not (0.0 < fov < 180.0):
            raise ValueError("--camera-fov must be between 0 and 180 degrees")
        patch["camera_fov"] = fov

    if args.suppress_enable and args.suppress_disable:
        raise ValueError("Cannot specify both --suppress-enable and --suppress-disable")
    if args.suppress_enable:
        patch["suppress_enabled"] = True
    elif args.suppress_disable:
        patch["suppress_enabled"] = False
    if args.suppress_bearing is not None:
        patch["suppress_bearing"] = float(args.suppress_bearing) % 360.0
    if args.suppress_tolerance is not None:
        tol = float(args.suppress_tolerance)
        if tol < 0.0 or tol >= 180.0:
            raise ValueError("--suppress-tolerance must be in [0, 180)")
        patch["suppress_tolerance"] = tol

    if args.classifier_on and args.classifier_off:
        raise ValueError("Cannot specify both --classifier-on and --classifier-off")
    if args.classifier_on:
        patch["classifier_enabled"] = True
    elif args.classifier_off:
        patch["classifier_enabled"] = False

    if args.tracker_on and args.tracker_off:
        raise ValueError("Cannot specify both --tracker-on and --tracker-off")
    if args.tracker_on:
        patch["tracker_enabled"] = True
    elif args.tracker_off:
        patch["tracker_enabled"] = False

    if not patch:
        raise ValueError("No configuration changes specified")
    return patch


def build_remote_cli_command(
    target: JetsonTarget,
    cli_args: Sequence[str],
    detach: bool = False,
    log_path: str = DEFAULT_LOG_PATH,
) -> str:
    """Build the remote shell command to run `main.py cli` inside Jetson's .venv."""
    quoted_dir = shlex.quote(target.remote_dir)
    args_str = " ".join(shlex.quote(arg) for arg in cli_args)
    base = f"cd {quoted_dir} && PYTHONUNBUFFERED=1 .venv/bin/python -u main.py cli"
    if args_str:
        base = f"{base} {args_str}"
    if detach:
        quoted_log = shlex.quote(log_path)
        return f"nohup bash -lc {shlex.quote(base)} > {quoted_log} 2>&1 < /dev/null & echo $!"
    return base


def _run_cmd(cmd: List[str], dry_run: bool = False) -> int:
    if dry_run:
        print("[DRY-RUN]", " ".join(shlex.quote(c) for c in cmd))
        return 0
    proc = subprocess.run(cmd)
    return proc.returncode


def run_command_streaming(
    cmd: List[str],
    on_line: Callable[[str], None],
    password: Optional[str] = None,
    stop_flag: Optional[Callable[[], bool]] = None,
) -> int:
    """Run a command (SSH or rsync) and stream output lines to `on_line`.

    Uses a pseudo-TTY when `password` is provided or for interactive SSH
    streaming, and handles password prompts automatically when `password` is set.
    """
    pid, fd = pty.fork()
    if pid == 0:
        os.execvp(cmd[0], cmd)

    buffer = bytearray()
    pw_sent = False
    pw_bytes = (password + "\n").encode() if password else None

    try:
        while True:
            if stop_flag is not None and stop_flag():
                try:
                    os.kill(pid, signal.SIGINT)
                except ProcessLookupError:
                    pass
                break

            ready, _, _ = select.select([fd], [], [], 0.15)
            if not ready:
                finished_pid, status = os.waitpid(pid, os.WNOHANG)
                if finished_pid == pid:
                    if buffer:
                        on_line(buffer.decode(errors="replace").rstrip("\r\n"))
                    return os.waitstatus_to_exitcode(status)
                continue

            try:
                chunk = os.read(fd, 4096)
            except OSError:
                break
            if not chunk:
                break

            buffer.extend(chunk)
            if pw_bytes is not None and not pw_sent and b"password:" in buffer.lower():
                os.write(fd, pw_bytes)
                pw_sent = True
                buffer.clear()
                continue

            while b"\n" in buffer:
                line_bytes, _, rest = buffer.partition(b"\n")
                buffer = bytearray(rest)
                text = line_bytes.decode(errors="replace").rstrip("\r")
                if text.strip().endswith("password:") and pw_bytes is not None:
                    continue
                on_line(text)

        if buffer:
            on_line(buffer.decode(errors="replace").rstrip("\r\n"))
        _, status = os.waitpid(pid, 0)
        return os.waitstatus_to_exitcode(status)
    finally:
        try:
            os.close(fd)
        except OSError:
            pass


def parse_cli_block(lines: Sequence[str]) -> Optional[Dict[str, str]]:
    """Parse a 5-line `format_fused_state` block into a dict if complete."""
    parsed: Dict[str, str] = {}
    for line in lines:
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip().upper()
        if key in {"MODE", "SOURCE", "BEARING", "ELEVATION", "DIRECTION_VECTOR"}:
            parsed[key] = value.strip()
    if len(parsed) == 5:
        return parsed
    return None


def build_remote_discovery_script(remote_dir: str) -> str:
    """Build a remote command that outputs Jetson config, audio devices, and cameras as @DISCOVERY JSON."""
    py_code = """
import json, socket
from pathlib import Path
from audio_manager import MultiDeviceAudioManager
from vision_detector import discover_cameras

cfg = json.loads(Path("rig_config.json").read_text())
devs = []
for d in MultiDeviceAudioManager.list_available_devices():
    devs.append({
        "index": d["index"],
        "name": d["name"],
        "channels": d["channels"],
        "samplerate": d["samplerate"],
        "label": f"[{d['index']}] {d['name']} ({d['channels']}ch)",
    })

cams = []
try:
    with socket.create_connection(("127.0.0.1", 8554), timeout=0.25):
        cams.append("rtsp://127.0.0.1:8554/drone")
except OSError:
    pass

try:
    for c in discover_cameras():
        if c.source not in cams:
            cams.append(c.source)
except Exception:
    pass

by_id = Path("/dev/v4l/by-id")
if by_id.is_dir():
    for p in sorted(by_id.iterdir()):
        if str(p) not in cams:
            cams.append(str(p))

cur_cam = str(cfg.get("vision", {}).get("camera", "")).strip()
if cur_cam in ("/dev/video0", "0") and "/dev/video0" not in cams and "rtsp://127.0.0.1:8554/drone" in cams:
    cfg.setdefault("vision", {})["camera"] = "rtsp://127.0.0.1:8554/drone"
    cur_cam = "rtsp://127.0.0.1:8554/drone"

if cur_cam and cur_cam not in cams and not cur_cam.startswith("/dev/video0/dev/"):
    cams.append(cur_cam)

print("@DISCOVERY:" + json.dumps({"config": cfg, "audio_devices": devs, "cameras": cams}))
""".strip()
    return f"cd {shlex.quote(remote_dir)} && .venv/bin/python -c {shlex.quote(py_code)}"


def build_remote_camera_preview_command(target: JetsonTarget, camera: str) -> str:
    """Build a lightweight remote camera preview command emitting @PREVIEW:<b64> lines."""
    src = str(camera).strip() or "rtsp://127.0.0.1:8554/drone"
    py_code = f"""
import base64, socket, time
import cv2

src = {src!r}
cap = cv2.VideoCapture(src, cv2.CAP_V4L2) if src.startswith("/dev/") else cv2.VideoCapture(int(src) if src.isdigit() else src)
if not cap.isOpened() and src in ("/dev/video0", "0"):
    cap.release()
    try:
        with socket.create_connection(("127.0.0.1", 8554), timeout=0.25):
            cap = cv2.VideoCapture("rtsp://127.0.0.1:8554/drone")
    except OSError:
        pass
try:
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok or frame is None:
            time.sleep(0.2)
            continue
        thumb = cv2.resize(frame, (384, 216), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", thumb, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
        if ok:
            print("@PREVIEW:" + base64.b64encode(buf.tobytes()).decode("ascii"), flush=True)
        time.sleep(0.12)
finally:
    cap.release()
""".strip()
    return f"cd {shlex.quote(target.remote_dir)} && .venv/bin/python -u -c {shlex.quote(py_code)}"


def _remote_config_update_script(remote_dir: str, patch: Dict[str, Any]) -> str:
    patch_json = json.dumps(patch)
    cfg_path = f"{remote_dir.rstrip('/')}/rig_config.json"
    py_code = f"""
import json
from pathlib import Path

cfg_path = Path({cfg_path!r})
cfg = json.loads(cfg_path.read_text())
patch = json.loads({patch_json!r})

if "dev_front" in patch:
    cfg.setdefault("arrays", {{}}).setdefault("front", {{}})["device_index"] = patch["dev_front"]
if "dev_left" in patch:
    cfg.setdefault("arrays", {{}}).setdefault("left", {{}})["device_index"] = patch["dev_left"]

det = cfg.setdefault("default_detection", {{}})
if "low" in patch:
    det["target_low_hz"] = patch["low"]
if "high" in patch:
    det["target_high_hz"] = patch["high"]
if "threshold" in patch:
    det["threshold_dbfs"] = patch["threshold"]
if "average" in patch:
    det["average_blocks"] = patch["average"]

vis = cfg.setdefault("vision", {{}})
if "camera" in patch:
    vis["camera"] = patch["camera"]
if "camera_fov" in patch:
    vis["horizontal_fov_deg"] = patch["camera_fov"]
    cam_key = str(patch.get("camera") or vis.get("camera") or "")
    if cam_key:
        vis.setdefault("camera_fovs", {{}})[cam_key] = patch["camera_fov"]

sup = cfg.setdefault("uma16_front_only", {{}}).setdefault("direction_suppression", {{}})
if "suppress_enabled" in patch:
    sup["enabled"] = patch["suppress_enabled"]
if "suppress_bearing" in patch:
    sup["bearing_deg"] = patch["suppress_bearing"]
if "suppress_tolerance" in patch:
    sup["tolerance_deg"] = patch["suppress_tolerance"]

if "classifier_enabled" in patch:
    cfg.setdefault("classifier", {{}})["enabled"] = patch["classifier_enabled"]
if "tracker_enabled" in patch:
    vis.setdefault("tracker", {{}})["enabled"] = patch["tracker_enabled"]

tmp_path = cfg_path.with_suffix(".json.tmp")
tmp_path.write_text(json.dumps(cfg, indent=2) + "\\n")
tmp_path.replace(cfg_path)
print("Updated", cfg_path)
print(json.dumps(cfg, indent=2))
""".strip()
    return f"python3 -c {shlex.quote(py_code)}"


def cmd_test_conn(target: JetsonTarget, args: argparse.Namespace) -> int:
    remote_cmd = (
        f"hostname && uname -m && "
        f"test -x {shlex.quote(target.remote_dir)}/.venv/bin/python && "
        f"{shlex.quote(target.remote_dir)}/.venv/bin/python --version && "
        f"(pgrep -af 'main.py' || echo 'No running main.py process')"
    )
    return _run_cmd(build_ssh_command(target, remote_cmd), dry_run=args.dry_run)


def cmd_setup_ssh(target: JetsonTarget, args: argparse.Namespace) -> int:
    key_path = Path.home() / ".ssh" / "id_ed25519"
    if not key_path.exists():
        rc = _run_cmd(
            ["ssh-keygen", "-t", "ed25519", "-f", str(key_path), "-N", ""],
            dry_run=args.dry_run,
        )
        if rc != 0:
            return rc
    return _run_cmd(
        ["ssh-copy-id", "-p", str(target.port), target.destination],
        dry_run=args.dry_run,
    )


def cmd_sync(target: JetsonTarget, args: argparse.Namespace) -> int:
    cmd = build_rsync_command(
        target,
        local_dir=LOCAL_ROOT,
        include_config=args.include_config,
        dry_run=args.dry_run,
    )
    return _run_cmd(cmd, dry_run=args.dry_run)


def cmd_devices(target: JetsonTarget, args: argparse.Namespace) -> int:
    remote_cmd = f"cd {shlex.quote(target.remote_dir)} && .venv/bin/python main.py list-devices"
    return _run_cmd(build_ssh_command(target, remote_cmd), dry_run=args.dry_run)


def cmd_cameras(target: JetsonTarget, args: argparse.Namespace) -> int:
    remote_cmd = f"cd {shlex.quote(target.remote_dir)} && .venv/bin/python main.py list-cameras"
    return _run_cmd(build_ssh_command(target, remote_cmd), dry_run=args.dry_run)


def cmd_config_show(target: JetsonTarget, args: argparse.Namespace) -> int:
    remote_cmd = f"cat {shlex.quote(target.remote_dir)}/rig_config.json"
    return _run_cmd(build_ssh_command(target, remote_cmd), dry_run=args.dry_run)


def cmd_config_set(target: JetsonTarget, args: argparse.Namespace) -> int:
    patch = build_config_patch(args)
    remote_cmd = _remote_config_update_script(target.remote_dir, patch)
    return _run_cmd(build_ssh_command(target, remote_cmd), dry_run=args.dry_run)


def cmd_run(target: JetsonTarget, args: argparse.Namespace) -> int:
    cli_args = list(args.cli_args or [])
    if cli_args and cli_args[0] == "--":
        cli_args = cli_args[1:]
    remote_cmd = build_remote_cli_command(target, cli_args, detach=args.detach)
    ssh_cmd = build_ssh_command(target, remote_cmd, tty=not args.detach)
    return _run_cmd(ssh_cmd, dry_run=args.dry_run)


def cmd_status(target: JetsonTarget, args: argparse.Namespace) -> int:
    remote_cmd = "pgrep -af 'main.py' || echo 'No running main.py process'"
    return _run_cmd(build_ssh_command(target, remote_cmd), dry_run=args.dry_run)


def cmd_logs(target: JetsonTarget, args: argparse.Namespace) -> int:
    follow = "-f " if args.follow else ""
    remote_cmd = f"tail -n {int(args.lines)} {follow}{shlex.quote(DEFAULT_LOG_PATH)}"
    return _run_cmd(build_ssh_command(target, remote_cmd, tty=args.follow), dry_run=args.dry_run)


def cmd_stop(target: JetsonTarget, args: argparse.Namespace) -> int:
    remote_cmd = "pkill -INT -f 'main.py (cli|gui)' && echo 'Stopped main.py' || echo 'No running main.py process'"
    return _run_cmd(build_ssh_command(target, remote_cmd), dry_run=args.dry_run)


def cmd_test(target: JetsonTarget, args: argparse.Namespace) -> int:
    remote_cmd = (
        f"cd {shlex.quote(target.remote_dir)} && "
        ".venv/bin/python -m unittest "
        "test_state_machine test_classifier test_vertical_uma16 "
        "test_vision test_direction_suppression test_vision_tracking"
    )
    return _run_cmd(build_ssh_command(target, remote_cmd), dry_run=args.dry_run)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Remote control and configuration CLI for the Jetson Orin rig."
    )
    parser.add_argument("--host", default=os.environ.get("JETSON_HOST", DEFAULT_HOST))
    parser.add_argument("--port", type=int, default=int(os.environ.get("JETSON_PORT", DEFAULT_PORT)))
    parser.add_argument("--user", default=os.environ.get("JETSON_USER", DEFAULT_USER))
    parser.add_argument("--remote-dir", default=os.environ.get("JETSON_DIR", DEFAULT_REMOTE_DIR))
    parser.add_argument("--dry-run", action="store_true", help="Print SSH/rsync commands without executing")

    sub = parser.add_subparsers(dest="command", required=True)

    p_conn = sub.add_parser("test-conn", help="Verify SSH connectivity and Jetson environment")
    p_conn.set_defaults(func=cmd_test_conn)

    p_ssh = sub.add_parser("setup-ssh", help="Install laptop SSH public key on the Jetson")
    p_ssh.set_defaults(func=cmd_setup_ssh)

    p_sync = sub.add_parser("sync", help="Sync local project files to the Jetson")
    p_sync.add_argument("--include-config", action="store_true", help="Also overwrite Jetson rig_config.json")
    p_sync.set_defaults(func=cmd_sync)

    p_dev = sub.add_parser("devices", help="List audio devices on the Jetson")
    p_dev.set_defaults(func=cmd_devices)

    p_cam = sub.add_parser("cameras", help="List cameras on the Jetson")
    p_cam.set_defaults(func=cmd_cameras)

    p_cfg = sub.add_parser("config", help="Inspect or update Jetson rig_config.json")
    cfg_sub = p_cfg.add_subparsers(dest="config_command", required=True)
    p_cfg_show = cfg_sub.add_parser("show", help="Print Jetson rig_config.json")
    p_cfg_show.set_defaults(func=cmd_config_show)

    p_cfg_set = cfg_sub.add_parser("set", help="Update fields in Jetson rig_config.json")
    p_cfg_set.add_argument("--dev-front", type=int, default=None)
    p_cfg_set.add_argument("--dev-left", type=int, default=None)
    p_cfg_set.add_argument("--low", type=float, default=None)
    p_cfg_set.add_argument("--high", type=float, default=None)
    p_cfg_set.add_argument("--threshold", type=float, default=None)
    p_cfg_set.add_argument("--average", type=int, default=None)
    p_cfg_set.add_argument("--camera", default=None)
    p_cfg_set.add_argument("--camera-fov", type=float, default=None)
    p_cfg_set.add_argument("--suppress-enable", action="store_true")
    p_cfg_set.add_argument("--suppress-disable", action="store_true")
    p_cfg_set.add_argument("--suppress-bearing", type=float, default=None)
    p_cfg_set.add_argument("--suppress-tolerance", type=float, default=None)
    p_cfg_set.add_argument("--classifier-on", action="store_true")
    p_cfg_set.add_argument("--classifier-off", action="store_true")
    p_cfg_set.add_argument("--tracker-on", action="store_true")
    p_cfg_set.add_argument("--tracker-off", action="store_true")
    p_cfg_set.set_defaults(func=cmd_config_set)

    p_run = sub.add_parser("run", help="Start main.py cli on the Jetson and stream output")
    p_run.add_argument("--detach", action="store_true", help="Run in background on the Jetson")
    p_run.add_argument("cli_args", nargs=argparse.REMAINDER, help="Arguments forwarded to main.py cli")
    p_run.set_defaults(func=cmd_run)

    p_status = sub.add_parser("status", help="Check if main.py is running on the Jetson")
    p_status.set_defaults(func=cmd_status)

    p_logs = sub.add_parser("logs", help="View detached Jetson CLI logs")
    p_logs.add_argument("-f", "--follow", action="store_true")
    p_logs.add_argument("-n", "--lines", type=int, default=40)
    p_logs.set_defaults(func=cmd_logs)

    p_stop = sub.add_parser("stop", help="Stop running main.py on the Jetson")
    p_stop.set_defaults(func=cmd_stop)

    p_test = sub.add_parser("test", help="Run unit tests on the Jetson")
    p_test.set_defaults(func=cmd_test)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args, extra = parser.parse_known_args(argv)
    if extra:
        if args.command == "run":
            args.cli_args = list(extra) + list(args.cli_args or [])
        else:
            parser.error(f"unrecognized arguments: {' '.join(extra)}")
    target = JetsonTarget(
        host=args.host,
        port=args.port,
        user=args.user,
        remote_dir=args.remote_dir,
    )
    try:
        return int(args.func(target, args))
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
