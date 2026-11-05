#!/usr/bin/env python3
"""Multi-device USB audio stream manager and real-time acoustic field simulator.

Coordinates concurrent audio capture across multiple USB audio devices (UMA-8 and
UMA-16 arrays) and includes a high-fidelity synthetic wavefield simulator for
testing and verification without physical hardware.
"""

import threading
import time
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import sounddevice as sd

from geometry import RigGeometry, SOUND_SPEED


class MultiDeviceAudioManager:
    """Coordinates concurrent audio capture across multiple arrays or runs in simulation mode."""

    def __init__(
        self,
        geometry: RigGeometry,
        sample_rate: int = 44100,
        block_duration: float = 0.2,
        simulate: bool = False,
    ):
        self.geo = geometry
        self.sample_rate = sample_rate
        self.block_duration = block_duration
        self.blocksize = max(128, int(round(sample_rate * block_duration)))
        self.simulate = simulate

        self.streams: Dict[str, sd.InputStream] = {}
        self.buffers: Dict[str, np.ndarray] = {}
        self.lock = threading.Lock()
        self.is_running = False

        # Simulation state
        self.sim_target_pos = np.array([1.2, 2.5], dtype=float)  # x=1.2m, y=2.5m forward
        self.sim_freq_hz = 1000.0
        self.sim_snr_db = 15.0
        self.sim_t0 = time.time()
        self.sim_speed_rad_per_s = 0.3  # orbit speed for moving virtual drone

    def set_simulation_target(
        self,
        x: Optional[float] = None,
        y: Optional[float] = None,
        freq_hz: Optional[float] = None,
        snr_db: Optional[float] = None,
    ):
        with self.lock:
            if x is not None:
                self.sim_target_pos[0] = x
            if y is not None:
                self.sim_target_pos[1] = y
            if freq_hz is not None:
                self.sim_freq_hz = freq_hz
            if snr_db is not None:
                self.sim_snr_db = snr_db

    @staticmethod
    def list_available_devices() -> List[Dict]:
        """Return list of input-capable audio devices."""
        devices = []
        for i, d in enumerate(sd.query_devices()):
            if d["max_input_channels"] > 0:
                devices.append({
                    "index": i,
                    "name": d["name"],
                    "channels": d["max_input_channels"],
                    "samplerate": d["default_samplerate"],
                })
        return devices

    def start(self, device_map: Optional[Dict[str, int]] = None):
        """Start audio capture streams or simulation loop.

        device_map: {"front": dev_idx, "left": dev_idx, "right": dev_idx}
        """
        if self.is_running:
            return
        self.is_running = True

        if self.simulate:
            return

        # Hardware capture mode
        for arr_id, arr in self.geo.arrays.items():
            dev_idx = None
            if device_map and arr_id in device_map:
                dev_idx = device_map[arr_id]
            else:
                # Automatic hint discovery
                hint = arr_id.lower()
                for d in sd.query_devices():
                    if d["max_input_channels"] >= len(arr.mic_positions):
                        name = d["name"].lower()
                        if (arr.array_type == "uma8" and ("uma" in name or "micarray" in name)) or \
                           (arr.array_type == "uma16" and ("uma-16" in name or "uma16" in name)):
                            dev_idx = d["index"]
                            break

            if dev_idx is None:
                # Fallback to default system input if specific device isn't matched
                default_dev = sd.default.device[0]
                dev_idx = default_dev if default_dev is not None else 0

            info = sd.query_devices(dev_idx)
            n_ch = min(info["max_input_channels"], 16 if arr.array_type == "uma16" else 8)

            def make_callback(side_id):
                def callback(indata, frames, time_info, status):
                    with self.lock:
                        self.buffers[side_id] = indata.copy()
                return callback

            stream = sd.InputStream(
                device=dev_idx,
                channels=n_ch,
                samplerate=self.sample_rate,
                blocksize=self.blocksize,
                dtype="float32",
                callback=make_callback(arr_id),
            )
            stream.start()
            self.streams[arr_id] = stream

    def stop(self):
        self.is_running = False
        for s in self.streams.values():
            try:
                s.stop()
                s.close()
            except Exception:
                pass
        self.streams.clear()

    def get_latest_blocks(self) -> Dict[str, np.ndarray]:
        """Fetch synchronized audio blocks for all arrays."""
        if self.simulate:
            return self._generate_simulated_blocks()

        with self.lock:
            out = {}
            for arr_id in self.geo.arrays:
                if arr_id in self.buffers:
                    out[arr_id] = self.buffers[arr_id].copy()
            return out

    def _generate_simulated_blocks(self) -> Dict[str, np.ndarray]:
        """Synthesize wave propagation from virtual moving target to all 39 mics."""
        now = time.time()
        elapsed = now - self.sim_t0

        # Optional slow orbit to simulate moving drone
        orbit_radius = 2.5
        center_x, center_y = 0.0, 3.0
        angle = self.sim_speed_rad_per_s * elapsed
        target_pos = np.array([
            center_x + orbit_radius * np.sin(angle),
            center_y + orbit_radius * np.cos(angle) * 0.5,
        ])
        with self.lock:
            target_pos = self.sim_target_pos.copy() if self.sim_speed_rad_per_s == 0 else target_pos
            freq = self.sim_freq_hz
            snr = self.sim_snr_db

        t = np.arange(self.blocksize) / self.sample_rate
        out = {}

        for arr_id, arr in self.geo.arrays.items():
            n_channels = 8 if arr.array_type == "uma8" else 16
            block = np.zeros((self.blocksize, n_channels), dtype=np.float32)

            for ch, mic_pos in arr.mic_positions.items():
                if ch >= n_channels:
                    continue
                # Spherical propagation delay
                dist = np.linalg.norm(target_pos - mic_pos)
                delay_s = dist / SOUND_SPEED
                # Inverse-square amplitude scaling + phase shift
                amp = 0.5 / max(0.5, dist)
                tone = amp * np.sin(2.0 * np.pi * freq * (t - delay_s))

                # Additive sensor noise
                noise_amp = amp * (10.0 ** (-snr / 20.0))
                noise = np.random.normal(0, noise_amp, self.blocksize)
                block[:, ch] = tone + noise

            out[arr_id] = block

        return out
