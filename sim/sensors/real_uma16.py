"""Real UMA-16 V2 audio/DOA observation bridge for PX4 SITL + Gazebo HIL."""

from __future__ import annotations

import atexit
from dataclasses import dataclass
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from typing import Callable, Dict, IO, List, Optional, Tuple


def parse_telemetry_envelope(line: str) -> Optional[Dict]:
    """Parse a full @JSON: telemetry payload dict."""
    text = line.strip()
    if not text.startswith("@JSON:"):
        return None
    try:
        payload = json.loads(text[6:])
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict) or payload.get("type") != "telemetry":
        return None
    return payload


from frequency_id.paths import PROJECT_ROOT

REPO_ROOT = PROJECT_ROOT


@dataclass(frozen=True)
class RealUMA16Config:
    source_mode: str = "local"  # "local" (spawn main.py) or "stdin" (piped @JSON: stream)
    device: Optional[str] = None
    no_classifier: bool = True
    low_hz: Optional[float] = None
    high_hz: Optional[float] = None
    threshold_dbfs: Optional[float] = None
    average_blocks: Optional[int] = None
    suppress_bearing_deg: Optional[float] = None
    suppress_tolerance_deg: Optional[float] = None
    mount_calibration_id: str = "uma16-front-body-v1"
    normalize_clock: bool = True
    startup_timeout_s: float = 12.0
    enable_vision: bool = False
    camera: Optional[str] = None
    camera_fov: Optional[float] = None
    no_tracker: bool = False


class RealUMA16Source:
    """Thread-safe reader for live UMA-16 front-only @JSON: telemetry."""

    def __init__(
        self,
        config: Optional[RealUMA16Config] = None,
        stream: Optional[IO[str]] = None,
        clock_fn: Callable[[], float] = time.monotonic,
    ):
        self.config = config or RealUMA16Config()
        if self.config.source_mode not in ("local", "stdin"):
            raise ValueError(f"Unsupported source_mode: {self.config.source_mode!r}")
        self._external_stream = stream
        self._clock_fn = clock_fn

        self._lock = threading.Lock()
        self._ready_event = threading.Event()
        self._stop_event = threading.Event()
        self._proc: Optional[subprocess.Popen] = None
        self._reader_thread: Optional[threading.Thread] = None
        self._stderr_thread: Optional[threading.Thread] = None
        self._stderr_lines: List[str] = []

        self._clock_offset: Optional[float] = None
        self._last_sequence: Optional[int] = None
        self._latest_payload: Optional[Dict] = None
        self._latest_fused_payload: Optional[Dict] = None
        self._latest_camera_payload: Optional[Dict] = None
        self._packets_received = 0
        self._last_received_at: Optional[float] = None
        self._last_fused_received_at: Optional[float] = None

    def build_command(self) -> List[str]:
        cmd = [
            sys.executable,
            "-u",
            str(REPO_ROOT / "main.py"),
            "cli",
            "--uma16-front-only",
            "--stream-json",
        ]
        if not self.config.enable_vision:
            cmd.append("--no-vision")
        else:
            if self.config.camera is not None:
                cmd.extend(["--camera", str(self.config.camera)])
            if self.config.camera_fov is not None:
                cmd.extend(["--camera-fov", str(self.config.camera_fov)])
            if self.config.no_tracker:
                cmd.append("--no-tracker")
        if self.config.no_classifier:
            cmd.append("--no-classifier")
        if self.config.device is not None:
            cmd.extend(["--dev-left", str(self.config.device)])
        if self.config.low_hz is not None:
            cmd.extend(["--low", str(self.config.low_hz)])
        if self.config.high_hz is not None:
            cmd.extend(["--high", str(self.config.high_hz)])
        if self.config.threshold_dbfs is not None:
            cmd.extend(["--threshold", str(self.config.threshold_dbfs)])
        if self.config.average_blocks is not None:
            cmd.extend(["--average", str(self.config.average_blocks)])
        if self.config.suppress_bearing_deg is not None:
            cmd.extend(["--suppress-bearing", str(self.config.suppress_bearing_deg)])
        if self.config.suppress_tolerance_deg is not None:
            cmd.extend(["--suppress-tolerance", str(self.config.suppress_tolerance_deg)])
        return cmd

    @staticmethod
    def kill_stale():
        subprocess.run(
            ["pkill", "-9", "-f", "[m]ain.py cli --uma16-front-only --stream-json"],
            capture_output=True,
        )

    def start(self):
        if self._reader_thread is not None:
            return
        self._stop_event.clear()

        if self._external_stream is not None:
            stream = self._external_stream
        elif self.config.source_mode == "stdin":
            stream = sys.stdin
        else:
            self.kill_stale()
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            self._proc = subprocess.Popen(
                self.build_command(),
                cwd=str(REPO_ROOT),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                start_new_session=True,
            )
            atexit.register(self.close)
            stream = self._proc.stdout
            self._stderr_thread = threading.Thread(
                target=self._read_stderr,
                args=(self._proc.stderr,),
                daemon=True,
            )
            self._stderr_thread.start()

        self._reader_thread = threading.Thread(
            target=self._read_stdout,
            args=(stream,),
            daemon=True,
        )
        self._reader_thread.start()

    def _read_stderr(self, stream: Optional[IO[str]]):
        if stream is None:
            return
        try:
            for line in stream:
                if self._stop_event.is_set():
                    break
                text = line.rstrip()
                if text:
                    with self._lock:
                        self._stderr_lines.append(text)
                        if len(self._stderr_lines) > 50:
                            self._stderr_lines = self._stderr_lines[-50:]
        except Exception:
            pass

    def _read_stdout(self, stream: Optional[IO[str]]):
        if stream is None:
            return
        try:
            for line in stream:
                if self._stop_event.is_set():
                    break
                self.ingest_line(line)
        except Exception:
            pass

    def ingest_line(self, line: str) -> bool:
        """Parse one telemetry line and update latest UMA-16 guidance and fused vision payloads."""
        envelope = parse_telemetry_envelope(line)
        if envelope is None:
            return False

        obs = envelope.get("guidance")
        fused = envelope.get("fused")
        camera = envelope.get("camera")

        has_valid_guidance = (
            isinstance(obs, dict)
            and obs.get("array_mode") == "uma16_front_only"
            and obs.get("array_id") == "left"
        )
        has_valid_fused = (
            self.config.enable_vision
            and envelope.get("array_mode") == "uma16_front_only"
            and (isinstance(fused, dict) or isinstance(camera, dict))
        )
        if not has_valid_guidance and not has_valid_fused:
            return False

        recv_now = float(self._clock_fn())

        with self._lock:
            if isinstance(fused, dict):
                self._latest_fused_payload = dict(fused)
                self._last_fused_received_at = recv_now
            if isinstance(camera, dict):
                self._latest_camera_payload = dict(camera)

            if has_valid_guidance:
                payload = dict(obs)
                seq = payload.get("sequence")
                raw_captured_at = payload.get("captured_at")
                is_new_seq = isinstance(seq, int) and (
                    self._last_sequence is None or seq > self._last_sequence
                )
                if (
                    self.config.normalize_clock
                    and raw_captured_at is not None
                ):
                    raw_ts = float(raw_captured_at)
                    if self._clock_offset is None and is_new_seq:
                        self._clock_offset = recv_now - raw_ts
                    if self._clock_offset is not None:
                        payload["captured_at"] = raw_ts + self._clock_offset

                if is_new_seq:
                    self._last_sequence = seq
                self._latest_payload = payload

            self._packets_received += 1
            self._last_received_at = recv_now
            self._ready_event.set()
        return True

    def wait_ready(self, timeout_s: Optional[float] = None):
        timeout = self.config.startup_timeout_s if timeout_s is None else float(timeout_s)
        deadline = self._clock_fn() + timeout
        while self._clock_fn() < deadline:
            if self._ready_event.wait(timeout=0.1):
                return
            if self._proc is not None and self._proc.poll() is not None:
                with self._lock:
                    tail = "\n".join(self._stderr_lines[-10:])
                raise RuntimeError(
                    f"UMA-16 stream process exited early with code {self._proc.returncode}.\n{tail}"
                )
        with self._lock:
            tail = "\n".join(self._stderr_lines[-10:])
        raise TimeoutError(
            f"Timed out after {timeout:.1f}s waiting for UMA-16 front-only @JSON telemetry.\n{tail}"
        )

    def sample(
        self,
        now: Optional[float] = None,
        drone_ned=None,
        drone_rpy_deg=None,
    ) -> Tuple[Optional[Dict], Dict]:
        """Return latest UMA-16 observation dict and diagnostic metadata."""
        del now, drone_ned, drone_rpy_deg
        with self._lock:
            payload = dict(self._latest_payload) if self._latest_payload is not None else None
            diag = {
                "packets_received": self._packets_received,
                "last_sequence": self._last_sequence,
                "last_received_at": self._last_received_at,
            }
        return payload, diag

    def sample_fused(
        self,
    ) -> Tuple[Optional[Dict], Optional[Dict], Optional[Dict], Dict]:
        """Return latest guidance, fused vision/audio, and camera payloads with diagnostics."""
        with self._lock:
            guidance_payload = (
                dict(self._latest_payload) if self._latest_payload is not None else None
            )
            fused_payload = (
                dict(self._latest_fused_payload)
                if self._latest_fused_payload is not None
                else None
            )
            camera_payload = (
                dict(self._latest_camera_payload)
                if self._latest_camera_payload is not None
                else None
            )
            diag = {
                "packets_received": self._packets_received,
                "last_sequence": self._last_sequence,
                "last_received_at": self._last_received_at,
                "last_fused_received_at": self._last_fused_received_at,
            }
        return guidance_payload, fused_payload, camera_payload, diag

    def close(self):
        self._stop_event.set()
        if self._proc is not None:
            try:
                os.killpg(os.getpgid(self._proc.pid), signal.SIGTERM)
                self._proc.wait(timeout=2.0)
            except Exception:
                try:
                    os.killpg(os.getpgid(self._proc.pid), signal.SIGKILL)
                except Exception:
                    pass
            for pipe in (self._proc.stdout, self._proc.stderr):
                if pipe is not None:
                    try:
                        pipe.close()
                    except Exception:
                        pass
            self._proc = None
        if self._reader_thread is not None and self._reader_thread.is_alive():
            self._reader_thread.join(timeout=1.5)
        self._reader_thread = None
        if self._stderr_thread is not None and self._stderr_thread.is_alive():
            self._stderr_thread.join(timeout=1.0)
        self._stderr_thread = None
