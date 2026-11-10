#!/usr/bin/env python3
"""Multi-device USB audio stream manager and real-time acoustic field simulator.

Coordinates concurrent audio capture across multiple USB audio devices (UMA-8 and
UMA-16 / MCHStreamer arrays) and includes a high-fidelity synthetic wavefield simulator for
testing and verification without physical hardware.
"""

import threading
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import scipy.signal as signal
import sounddevice as sd

from geometry import RigGeometry, SOUND_SPEED


def resolve_audio_device(spec: Optional[Union[int, str]], min_channels: int = 1) -> Optional[int]:
    """Resolve an integer index or name substring to a valid sounddevice input index."""
    if spec is None:
        return None
    # 1. Direct integer index
    try:
        idx = int(spec)
        info = sd.query_devices(idx)
        if info["max_input_channels"] >= min_channels:
            return idx
    except (ValueError, TypeError):
        pass
    except Exception:
        pass

    # 2. Match by name substring
    query = str(spec).strip().lower()
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] >= min_channels:
            if query in d["name"].lower():
                return i
    return None


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
        self.active_devices: Dict[str, Tuple[int, str]] = {}
        self.enabled_arrays = set(self.geo.arrays)
        self.lock = threading.Lock()
        self.is_running = False

        # Simulation state
        self.sim_target_pos = np.array([-1.0, 2.5], dtype=float)  # Shared forward/left field of view
        self.sim_freq_hz = 1000.0
        self.sim_snr_db = 15.0
        self.sim_t0 = time.time()
        self.sim_speed_rad_per_s = 0.3  # orbit speed for moving virtual drone
        self.sim_drone_wave: Optional[np.ndarray] = None
        self._load_sim_drone_audio()

    def _load_sim_drone_audio(self):
        """Preload realistic drone sound from reference outputs for synthetic wavefield."""
        ref_path = Path(__file__).resolve().parent / "model" / "reference_outputs.npz"
        if not ref_path.exists():
            ref_path = Path("/home/dhia/Desktop/projects/MobileNetV2/model/reference_outputs.npz")
        if ref_path.exists():
            try:
                data = np.load(ref_path)
                labels = data.get("labels", [])
                drone_indices = np.where(labels == 1)[0]
                idx = int(drone_indices[0]) if len(drone_indices) > 0 else 0
                w16 = data["waveforms"][idx]
                # Resample 16 kHz to rig sample rate (44.1 kHz)
                self.sim_drone_wave = signal.resample_poly(w16, 441, 160).astype(np.float32)
            except Exception as e:
                self.sim_drone_wave = None

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

    def start(
        self,
        device_map: Optional[Dict[str, Union[int, str]]] = None,
        array_ids: Optional[List[str]] = None,
    ):
        """Start capture for the selected arrays or run them in simulation mode."""
        if self.is_running:
            return
        self.enabled_arrays = set(array_ids) if array_ids is not None else set(self.geo.arrays)
        self.is_running = True

        if self.simulate:
            return

        # Hardware capture mode
        for arr_id, arr in self.geo.arrays.items():
            if arr_id not in self.enabled_arrays:
                continue
            req_channels = 16 if arr.array_type == "uma16" else 7
            dev_idx = None

            # 1. Check explicit caller map
            if device_map and arr_id in device_map:
                dev_idx = resolve_audio_device(device_map[arr_id], min_channels=req_channels)

            # 2. Check rig_config.json
            if dev_idx is None:
                cfg_idx = self.geo.cfg.get("arrays", {}).get(arr_id, {}).get("device_index")
                if cfg_idx is not None:
                    dev_idx = resolve_audio_device(cfg_idx, min_channels=req_channels)

            # 3. Automatic name discovery (MCHStreamer / UMA16 / micArray)
            if dev_idx is None:
                hints = ["micarray", "uma8", "uma-8"] if arr.array_type == "uma8" else ["mchstreamer", "uma16", "uma-16"]
                for h in hints:
                    dev_idx = resolve_audio_device(h, min_channels=req_channels)
                    if dev_idx is not None:
                        break

            # 4. Fallback to default system input if specific device isn't matched
            if dev_idx is None:
                default_dev = sd.default.device[0]
                dev_idx = default_dev if default_dev is not None and default_dev >= 0 else 0

            info = sd.query_devices(dev_idx)
            n_ch = min(info["max_input_channels"], 16 if arr.array_type == "uma16" else 8)
            dev_name = info["name"]
            self.active_devices[arr_id] = (dev_idx, dev_name)
            print(f"[AUDIO] Array '{arr.name}' bound to device [{dev_idx}] {dev_name!r} ({n_ch} channels @ {self.sample_rate} Hz)")

            def make_callback(side_id):
                def callback(indata, frames, time_info, status):
                    with self.lock:
                        self.buffers[side_id] = indata.copy()
                return callback

            try:
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
            except Exception as e:
                print(f"[ERROR] Failed to start audio stream on device [{dev_idx}] for array '{arr_id}': {e}")

    def stop(self):
        self.is_running = False
        for s in self.streams.values():
            try:
                s.stop()
                s.close()
            except Exception:
                pass
        self.streams.clear()
        self.active_devices.clear()

    def get_latest_blocks(self) -> Dict[str, np.ndarray]:
        """Fetch the latest audio blocks from the selected arrays."""
        if self.simulate:
            return self._generate_simulated_blocks()

        with self.lock:
            out = {}
            for arr_id in self.enabled_arrays:
                if arr_id in self.buffers:
                    out[arr_id] = self.buffers[arr_id].copy()
            return out

    def _generate_simulated_blocks(self) -> Dict[str, np.ndarray]:
        """Synthesize wave propagation from the virtual target to all active microphones."""
        now = time.time()
        elapsed = now - self.sim_t0

        # Optional slow orbit to simulate moving drone
        orbit_radius = 0.5
        center_x, center_y = -1.0, 2.5
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
            if arr_id not in self.enabled_arrays:
                continue
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
                tone = 0.25 * amp * np.sin(2.0 * np.pi * freq * (t - delay_s))

                if self.sim_drone_wave is not None and len(self.sim_drone_wave) > 0:
                    delay_samples = int(round(delay_s * self.sample_rate))
                    idx = (int(round(elapsed * self.sample_rate)) + delay_samples) % len(self.sim_drone_wave)
                    drone_rolled = np.roll(self.sim_drone_wave, -idx)
                    drone_chunk = drone_rolled[:self.blocksize]
                    if len(drone_chunk) < self.blocksize:
                        drone_chunk = np.pad(drone_chunk, (0, self.blocksize - len(drone_chunk)))
                    sig = tone + 4.0 * amp * drone_chunk
                else:
                    sig = tone

                # Additive sensor noise
                noise_amp = amp * (10.0 ** (-snr / 20.0))
                noise = np.random.normal(0, noise_amp, self.blocksize)
                block[:, ch] = sig + noise

            out[arr_id] = block

        return out
