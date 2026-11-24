"""Frequency-gated SRP-PHAT direction-of-arrival estimation.

Elevation-capable vertical arrays use one joint local azimuth/elevation search.
The local direction is u = [cos(el) cos(az), cos(el) sin(az), sin(el)], where
local +X is the array normal and local +Y is the viewer's right.
"""

from collections import deque
from dataclasses import dataclass
import math
import time
from typing import Dict, Optional, Tuple

import numpy as np

SOUND_SPEED = 343.0


@dataclass
class DOAResult:
    array_id: str
    local_angle_deg: Optional[float]
    global_angle_deg: Optional[float]
    prominence: float
    confidence: float
    spectrum: Optional[np.ndarray]
    error: Optional[str] = None
    stable: bool = False
    elevation_deg: Optional[float] = None
    elevation_prominence: float = 0.0
    elevation_confidence: float = 0.0
    elevation_stable: bool = False
    direction_vector: Optional[np.ndarray] = None


class _TimestampedTracker:
    def __init__(self, window, min_prominence, decay, jump_deg, min_samples, max_age):
        if window < 1 or min_samples < 1 or max_age <= 0.0:
            raise ValueError("Tracker window, minimum samples, and maximum age must be positive")
        self.window = int(window)
        self.min_prominence = float(min_prominence)
        self.decay = float(decay)
        self.jump_deg = float(jump_deg)
        self.min_samples = int(min_samples)
        self.max_age = float(max_age)
        self.estimates = deque(maxlen=self.window)
        self.timestamps = deque(maxlen=self.window)
        self.current_accepted = False
        self.current_timestamp = None

    def _expire(self, timestamp):
        while self.timestamps and timestamp - self.timestamps[0] > self.max_age:
            self.timestamps.popleft()
            self.estimates.popleft()

    def _prepare(self, timestamp):
        if timestamp is None:
            timestamp = time.monotonic()
        timestamp = float(timestamp)
        if not math.isfinite(timestamp):
            self.current_accepted = False
            return None
        if self.timestamps and timestamp < self.timestamps[-1]:
            self.reset()
        self._expire(timestamp)
        self.current_timestamp = timestamp
        return timestamp

    def miss(self, timestamp=None):
        timestamp = self._prepare(timestamp)
        self.current_accepted = False
        if timestamp is not None:
            self.current_timestamp = timestamp

    def _append(self, angle, timestamp):
        self.estimates.append(angle)
        self.timestamps.append(timestamp)
        self.current_accepted = True
        self.current_timestamp = timestamp

    @property
    def ready(self):
        return len(self.estimates) >= min(self.window, max(1, self.min_samples))

    def current_fresh(self, now=None):
        if not self.current_accepted or self.current_timestamp is None:
            return False
        now = self.current_timestamp if now is None else float(now)
        return math.isfinite(now) and 0.0 <= now - self.current_timestamp <= self.max_age

    def reset(self):
        self.estimates.clear()
        self.timestamps.clear()
        self.current_accepted = False
        self.current_timestamp = None


class CircularTracker(_TimestampedTracker):
    """Wrap-safe temporal smoothing with observation freshness and jump detection."""

    def __init__(
        self,
        window: int = 5,
        min_prominence: float = 2.0,
        decay: float = 0.75,
        jump_deg: float = 40.0,
        min_samples: int = 2,
        max_age: float = 0.6,
    ):
        super().__init__(window, min_prominence, decay, jump_deg, min_samples, max_age)

    @staticmethod
    def _circular_diff(a_deg, b_deg):
        return abs(((a_deg - b_deg + 180.0) % 360.0) - 180.0)

    def add(self, angle_deg: Optional[float], prominence: float, timestamp=None):
        timestamp = self._prepare(timestamp)
        self.current_accepted = False
        if (
            timestamp is None
            or angle_deg is None
            or not math.isfinite(angle_deg)
            or not math.isfinite(prominence)
            or prominence < self.min_prominence
        ):
            return False
        current = self.angle
        if current is not None and self._circular_diff(angle_deg, current) > self.jump_deg:
            self.estimates.clear()
            self.timestamps.clear()
        self._append(float(np.deg2rad(angle_deg)), timestamp)
        return True

    def _weighted_vector(self):
        angles = np.asarray(self.estimates)
        weights = self.decay ** np.arange(len(angles) - 1, -1, -1)
        return (weights * np.exp(1j * angles)).sum() / weights.sum()

    @property
    def angle(self):
        if not self.estimates:
            return None
        return float(np.rad2deg(np.angle(self._weighted_vector())) % 360.0)

    @property
    def confidence(self):
        if not self.estimates:
            return 0.0
        return float(min(abs(self._weighted_vector()), 1.0))


class ElevationTracker(_TimestampedTracker):
    def __init__(
        self,
        window: int = 5,
        min_prominence: float = 0.45,
        decay: float = 0.75,
        jump_deg: float = 40.0,
        min_samples: int = 3,
        max_age: float = 0.6,
    ):
        super().__init__(window, min_prominence, decay, jump_deg, min_samples, max_age)

    def add(self, angle_deg: Optional[float], prominence: float, timestamp=None):
        timestamp = self._prepare(timestamp)
        self.current_accepted = False
        if (
            timestamp is None
            or angle_deg is None
            or not math.isfinite(angle_deg)
            or not math.isfinite(prominence)
            or prominence < self.min_prominence
        ):
            return False
        current = self.angle
        if current is not None and abs(angle_deg - current) > self.jump_deg:
            self.estimates.clear()
            self.timestamps.clear()
        self._append(float(angle_deg), timestamp)
        return True

    def _weighted_values(self):
        values = np.asarray(self.estimates)
        weights = self.decay ** np.arange(len(values) - 1, -1, -1)
        mean = float(np.sum(values * weights) / np.sum(weights))
        return values, weights, mean

    @property
    def angle(self):
        if not self.estimates:
            return None
        return self._weighted_values()[2]

    @property
    def confidence(self):
        if not self.estimates:
            return 0.0
        values, weights, mean = self._weighted_values()
        std = np.sqrt(np.sum(weights * (values - mean) ** 2) / np.sum(weights))
        return float(np.clip(1.0 - std / 45.0, 0.0, 1.0))


class ArrayDOA:
    """SRP-PHAT estimator with a joint 3D search for vertical arrays."""

    def __init__(
        self,
        array_id: str,
        mic_positions_local: Dict[int, np.ndarray],
        orientation_deg: float,
        sample_rate: int = 44100,
        grid_deg: float = 1.0,
        freq_lo: float = 200.0,
        freq_hi: float = 16000.0,
        channel_delays: Optional[Dict[int, float]] = None,
        mount: str = "flat",
        mic_heights_local: Optional[Dict[int, float]] = None,
        sound_speed: float = SOUND_SPEED,
        phase_response: Optional[Dict] = None,
        tracker_timeout_s: float = 0.6,
        steering_chunk: int = 256,
        frequency_chunk: int = 32,
    ):
        self.array_id = array_id
        self.mic_positions = mic_positions_local
        self.mic_heights = mic_heights_local or {}
        self.orientation_deg = float(orientation_deg)
        self.sample_rate = int(sample_rate)
        self.grid_deg = float(grid_deg)
        self.freq_lo = float(freq_lo)
        self.freq_hi = float(freq_hi)
        self.mount = mount
        self.sound_speed = float(sound_speed)
        self.steering_chunk = int(steering_chunk)
        self.frequency_chunk = int(frequency_chunk)
        if self.mount not in ("flat", "vertical"):
            raise ValueError(f"Unsupported array mount: {self.mount!r}")
        if self.sample_rate <= 0 or not 0.0 < self.grid_deg <= 90.0:
            raise ValueError("Sample rate and angular grid must be positive")
        if not math.isfinite(self.sound_speed) or self.sound_speed <= 0.0:
            raise ValueError("Sound speed must be finite and positive")
        if self.steering_chunk < 1 or self.frequency_chunk < 1:
            raise ValueError("Steering and frequency chunks must be positive")
        if not math.isfinite(tracker_timeout_s) or tracker_timeout_s <= 0.0:
            raise ValueError("Tracker timeout must be finite and positive")

        self.channels = sorted(self.mic_positions)
        if len(self.channels) < 2:
            raise ValueError("DOA requires at least two microphone channels")
        positions_2d = []
        positions_3d = []
        for ch in self.channels:
            position = np.asarray(self.mic_positions[ch], dtype=float).reshape(-1)
            if position.size not in (2, 3) or not np.all(np.isfinite(position)):
                raise ValueError(f"Microphone {ch} must have finite two- or three-dimensional coordinates")
            height = float(position[2] if position.size == 3 else self.mic_heights.get(ch, 0.0))
            if not math.isfinite(height):
                raise ValueError(f"Microphone {ch} height must be finite")
            positions_2d.append(position[:2])
            positions_3d.append(np.array([position[0], position[1], height], dtype=float))
        self._positions_2d = np.stack(positions_2d)
        self._positions_3d = np.stack(positions_3d)

        if self.mount == "vertical":
            horizontal_positions = np.unique(np.round(self._positions_2d[:, 1], decimals=9))
            if len(horizontal_positions) < 2:
                raise ValueError("A vertical array needs at least two horizontal positions")
            self.nominal_spacing_m = float(np.max(np.diff(horizontal_positions)))
        else:
            distances = np.linalg.norm(
                self._positions_2d[:, None, :] - self._positions_2d[None, :, :], axis=2
            )
            np.fill_diagonal(distances, np.inf)
            nearest_spacings = distances.min(axis=1)
            self.nominal_spacing_m = float(np.median(nearest_spacings[np.isfinite(nearest_spacings)]))
        if not math.isfinite(self.nominal_spacing_m) or self.nominal_spacing_m <= 0.0:
            raise ValueError("Microphone spacing must be finite and positive")
        self.max_direction_freq_hz = self.sound_speed / (2.0 * self.nominal_spacing_m)

        if self.mount == "vertical":
            self.angles = np.arange(-90.0, 90.0 + self.grid_deg, self.grid_deg) % 360.0
        else:
            self.angles = np.arange(0.0, 360.0, self.grid_deg)
        self.n_angles = len(self.angles)
        self.elevation_angles = np.arange(-90.0, 90.0 + self.grid_deg, self.grid_deg)
        heights = self._positions_3d[:, 2]
        self.elevation_enabled = self.mount == "vertical" and len(np.unique(np.round(heights, decimals=9))) >= 2

        self.delays = np.zeros(len(self.channels), dtype=float)
        for ch, delay in (channel_delays or {}).items():
            if ch not in self.channels:
                raise ValueError(f"Unknown calibration channel: {ch}")
            if not math.isfinite(delay):
                raise ValueError("Channel delays must be finite samples")
            self.delays[self.channels.index(ch)] = float(delay)
        self.phase_response = self._validate_phase_response(phase_response or {})

        self.tracker = CircularTracker(
            window=5,
            min_prominence=0.45 if self.mount == "vertical" else 2.0,
            min_samples=3 if self.mount == "vertical" else 2,
            max_age=tracker_timeout_s,
        )
        self.elevation_tracker = ElevationTracker(max_age=tracker_timeout_s)
        self._last_n = None
        self._band_idx = None
        self._steer = None
        self._align = None

    def _validate_phase_response(self, response):
        validated = {}
        for ch, calibration in response.items():
            if ch not in self.channels:
                raise ValueError(f"Unknown phase-response channel: {ch}")
            if not isinstance(calibration, dict):
                raise ValueError("Phase response must contain frequencies_hz and phase_deg")
            frequencies = np.asarray(calibration.get("frequencies_hz", []), dtype=float)
            phases = np.asarray(calibration.get("phase_deg", []), dtype=float)
            if frequencies.ndim != 1 or phases.ndim != 1 or frequencies.size != phases.size:
                raise ValueError("Phase-response frequencies and phases must be matching vectors")
            if frequencies.size == 0:
                continue
            if (
                not np.all(np.isfinite(frequencies))
                or not np.all(np.isfinite(phases))
                or np.any(frequencies < 0.0)
                or np.any(np.diff(frequencies) <= 0.0)
            ):
                raise ValueError("Phase-response frequencies must be finite, non-negative, and strictly increasing")
            validated[ch] = (frequencies, phases)
        return validated

    @staticmethod
    def _direction_grid(azimuths, elevations):
        azimuth = np.deg2rad(np.asarray(azimuths, dtype=float))[:, None]
        elevation = np.deg2rad(np.asarray(elevations, dtype=float))[None, :]
        cos_el = np.cos(elevation)
        directions = np.stack(
            [
                np.broadcast_to(cos_el * np.cos(azimuth), (len(azimuths), len(elevations))),
                np.broadcast_to(cos_el * np.sin(azimuth), (len(azimuths), len(elevations))),
                np.broadcast_to(np.sin(elevation), (len(azimuths), len(elevations))),
            ],
            axis=-1,
        )
        return directions.reshape(-1, 3)

    @staticmethod
    def _axis(start, stop, step, center=None):
        values = np.arange(start, stop + step * 0.5, step, dtype=float)
        values = values[(values >= start - 1e-9) & (values <= stop + 1e-9)]
        if values.size == 0 or values[-1] < stop - 1e-9:
            values = np.append(values, stop)
        if center is not None:
            values = np.append(values, center)
        return np.unique(np.clip(values, start, stop))

    def _power_for_directions(self, cross, frequencies, weights, directions):
        directions = np.asarray(directions, dtype=float)
        power = np.zeros(len(directions), dtype=float)
        for direction_start in range(0, len(directions), self.steering_chunk):
            direction_chunk = directions[direction_start:direction_start + self.steering_chunk]
            tau = -(self._positions_3d @ direction_chunk.T) / self.sound_speed
            chunk_power = np.zeros(len(direction_chunk), dtype=float)
            for freq_start in range(0, len(frequencies), self.frequency_chunk):
                freq_end = min(freq_start + self.frequency_chunk, len(frequencies))
                f = frequencies[freq_start:freq_end]
                c = cross[freq_start:freq_end]
                steer = np.exp(2j * np.pi * f[:, None, None] * tau[None, :, :]).astype(np.complex64)
                projected = np.einsum("kmn,kmt->knt", c, steer, optimize=True)
                response = np.einsum("knt,knt->t", projected, steer.conj(), optimize=True).real
                chunk_power += np.sum(weights[freq_start:freq_end, None] * response, axis=0)
            power[direction_start:direction_start + len(direction_chunk)] = chunk_power
        if not np.all(np.isfinite(power)):
            raise ValueError("Non-finite beamforming power")
        return np.maximum(power, 0.0)

    @staticmethod
    def _prominence(power):
        power = np.asarray(power, dtype=float)
        if power.size == 0:
            return 0.0
        value = (float(np.max(power)) - float(np.median(power))) / (float(np.std(power)) + 1e-12)
        return float(np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0))

    @staticmethod
    def _parabolic(values, coordinates, index, wrap=False):
        count = len(values)
        if wrap:
            left = (index - 1) % count
            right = (index + 1) % count
            step = float(np.median(np.diff(coordinates))) if count > 1 else 0.0
            y0, y1, y2 = values[left], values[index], values[right]
            denominator = y0 - 2.0 * y1 + y2
            if denominator < 0.0:
                fraction = float(np.clip(0.5 * (y0 - y2) / denominator, -0.5, 0.5))
                return float(coordinates[index] + fraction * step)
            return float(coordinates[index])
        if index <= 0 or index >= count - 1:
            return float(coordinates[index])
        y0, y1, y2 = values[index - 1:index + 2]
        denominator = y0 - 2.0 * y1 + y2
        if denominator >= 0.0:
            return float(coordinates[index])
        fraction = float(np.clip(0.5 * (y0 - y2) / denominator, -0.5, 0.5))
        step = 0.5 * (coordinates[index + 1] - coordinates[index - 1])
        return float(coordinates[index] + fraction * step)

    def _joint_estimate(self, cross, frequencies, weights):
        coarse_step = max(4.0, self.grid_deg)
        coarse_az = self._axis(-90.0, 90.0, coarse_step)
        coarse_el = self._axis(-90.0, 90.0, coarse_step)
        directions = self._direction_grid(coarse_az, coarse_el)
        coarse_power = self._power_for_directions(cross, frequencies, weights, directions).reshape(
            len(coarse_az), len(coarse_el)
        )
        coarse_peak = np.unravel_index(int(np.argmax(coarse_power)), coarse_power.shape)
        center_az = float(coarse_az[coarse_peak[0]])
        center_el = float(coarse_el[coarse_peak[1]])

        fine_az = self._axis(
            max(-90.0, center_az - coarse_step),
            min(90.0, center_az + coarse_step),
            self.grid_deg,
            center_az,
        )
        fine_el = self._axis(
            max(-90.0, center_el - coarse_step),
            min(90.0, center_el + coarse_step),
            self.grid_deg,
            center_el,
        )
        fine_directions = self._direction_grid(fine_az, fine_el)
        fine_power = self._power_for_directions(cross, frequencies, weights, fine_directions).reshape(
            len(fine_az), len(fine_el)
        )
        fine_peak = np.unravel_index(int(np.argmax(fine_power)), fine_power.shape)
        az_index, el_index = int(fine_peak[0]), int(fine_peak[1])
        azimuth = self._parabolic(fine_power[:, el_index], fine_az, az_index)
        elevation = self._parabolic(fine_power[az_index, :], fine_el, el_index)
        elevation = float(np.clip(elevation, -90.0, 90.0))
        if abs(elevation) >= 90.0 - 1e-9:
            azimuth = 0.0
        azimuth = float(np.clip(azimuth, -90.0, 90.0))

        az_projection = np.max(coarse_power, axis=1)
        el_projection = np.max(coarse_power, axis=0)
        signed_angles = (self.angles + 180.0) % 360.0 - 180.0
        spectrum = np.interp(signed_angles, coarse_az, az_projection)
        return (
            azimuth,
            elevation,
            self._prominence(az_projection),
            self._prominence(el_projection),
            spectrum,
        )

    def _miss(self, error, track, timestamp):
        if track:
            self.tracker.miss(timestamp)
            self.elevation_tracker.miss(timestamp)
        return DOAResult(self.array_id, None, None, 0.0, 0.0, None, error=error)

    def _validate_band(self, target_freq, target_bandwidth_hz, band):
        if target_freq is not None and (not math.isfinite(target_freq) or target_freq <= 0.0):
            return "invalid_band"
        if not math.isfinite(target_bandwidth_hz) or target_bandwidth_hz <= 0.0:
            return "invalid_band"
        if band is not None:
            if len(band) != 2 or not all(math.isfinite(float(value)) for value in band):
                return "invalid_band"
            low, high = float(band[0]), float(band[1])
            if low < 0.0 or high <= low or high > self.sample_rate / 2.0:
                return "invalid_band"
        return None

    def _spectra(self, x, target_freq, target_bandwidth_hz, band):
        n = len(x)
        windowed = x * np.hanning(n)[:, None]
        spectrum = np.fft.rfft(windowed, axis=0)
        frequencies = np.fft.rfftfreq(n, 1.0 / self.sample_rate)
        nyquist_limit = self.sample_rate / 2.0 - 50.0
        if band is not None:
            search_lo = max(20.0, float(band[0]) - 25.0)
            search_hi = min(nyquist_limit, float(band[1]) + 25.0)
        else:
            search_lo = self.freq_lo
            search_hi = self.freq_hi
        if target_freq is not None:
            search_lo = min(search_lo, max(20.0, target_freq - target_bandwidth_hz))
            search_hi = max(search_hi, min(nyquist_limit, target_freq + target_bandwidth_hz))
        search_lo = max(0.0, search_lo)
        search_hi = min(self.sample_rate / 2.0, search_hi)
        if search_hi <= search_lo:
            return None
        keep = (frequencies >= search_lo) & (frequencies <= search_hi)
        keep &= frequencies <= self.max_direction_freq_hz
        if target_freq is not None:
            keep &= frequencies >= target_freq - target_bandwidth_hz / 2.0
            keep &= frequencies <= target_freq + target_bandwidth_hz / 2.0
        elif band is not None:
            keep &= frequencies >= band[0]
            keep &= frequencies <= band[1]
        if not np.any(keep):
            return None
        energy = np.mean(np.abs(spectrum) ** 2, axis=1)
        selected_energy = energy[keep]
        if not np.all(np.isfinite(selected_energy)) or float(np.max(selected_energy)) <= 1e-20:
            return None
        keep &= energy >= float(np.max(selected_energy)) * 0.05
        if not np.any(keep):
            return None
        selected = spectrum[keep]
        f = frequencies[keep]
        if np.any(self.delays != 0.0) or self.phase_response:
            alignment = np.exp(2j * np.pi * f[:, None] * self.delays[None, :] / self.sample_rate)
            for index, ch in enumerate(self.channels):
                if ch in self.phase_response:
                    calibration_f, calibration_phase = self.phase_response[ch]
                    phase = np.interp(f, calibration_f, calibration_phase)
                    alignment[:, index] *= np.exp(-1j * np.deg2rad(phase))
            selected = selected * alignment
        normalized = selected / (np.abs(selected) + 1e-12)
        cross = normalized[:, :, None] * normalized[:, None, :].conj()
        weights = np.mean(np.abs(selected) ** 2, axis=1)
        weight_sum = float(np.sum(weights))
        if not math.isfinite(weight_sum) or weight_sum <= 0.0:
            raise ValueError("Invalid beamforming weights")
        return f, cross.astype(np.complex64), weights / weight_sum

    def estimate(
        self,
        block: np.ndarray,
        target_freq: Optional[float] = None,
        target_bandwidth_hz: float = 60.0,
        band: Optional[Tuple[float, float]] = None,
        track: bool = True,
        captured_at: Optional[float] = None,
        now: Optional[float] = None,
    ) -> DOAResult:
        """Estimate one block, failing closed with a documented error on invalid input."""
        now_value = time.monotonic() if now is None else float(now)
        timestamp = now_value if captured_at is None else float(captured_at)
        if not math.isfinite(now_value) or not math.isfinite(timestamp):
            return self._miss("invalid_timestamp", track, now_value if math.isfinite(now_value) else None)
        age = now_value - timestamp
        if age > self.tracker.max_age:
            return self._miss("stale_observation", track, timestamp)
        if age < -self.tracker.max_age:
            return self._miss("invalid_timestamp", track, timestamp)
        band_error = self._validate_band(target_freq, target_bandwidth_hz, band)
        if band_error is not None:
            return self._miss(band_error, track, timestamp)
        if target_freq is not None and target_freq > self.max_direction_freq_hz:
            return self._miss("spatial_aliasing", track, timestamp)
        if target_freq is None and band is not None and band[0] > self.max_direction_freq_hz:
            return self._miss("spatial_aliasing", track, timestamp)

        try:
            samples = np.asarray(block)
        except Exception:
            return self._miss("invalid_samples", track, timestamp)
        if samples.ndim != 2 or samples.shape[0] < 16:
            return self._miss("invalid_shape", track, timestamp)
        if samples.shape[1] <= max(self.channels):
            return self._miss("invalid_channels", track, timestamp)
        if np.iscomplexobj(samples):
            return self._miss("invalid_samples", track, timestamp)
        try:
            x = np.asarray(samples[:, self.channels], dtype=np.float64)
        except (TypeError, ValueError):
            return self._miss("invalid_samples", track, timestamp)
        if not np.all(np.isfinite(x)):
            return self._miss("invalid_samples", track, timestamp)
        if not np.any(np.abs(x) > 1e-12):
            return self._miss("no_signal", track, timestamp)

        try:
            spectra = self._spectra(x, target_freq, target_bandwidth_hz, band)
        except ValueError:
            return self._miss("invalid_weights", track, timestamp)
        if spectra is None:
            return self._miss("no_in_band_energy", track, timestamp)
        frequencies, cross, weights = spectra
        if not np.all(np.isfinite(weights)) or np.any(weights < 0.0) or float(np.sum(weights)) <= 0.0:
            return self._miss("invalid_weights", track, timestamp)

        try:
            if self.elevation_enabled:
                local_angle, elevation, prominence, elevation_prominence, spectrum = self._joint_estimate(
                    cross, frequencies, weights
                )
            else:
                directions = self._direction_grid(self.angles, [0.0])
                spectrum = self._power_for_directions(cross, frequencies, weights, directions)
                peak_index = int(np.argmax(spectrum))
                local_angle = self._parabolic(spectrum, self.angles, peak_index, wrap=self.mount == "flat")
                local_angle %= 360.0
                elevation = None
                prominence = self._prominence(spectrum)
                elevation_prominence = 0.0
        except ValueError:
            return self._miss("invalid_weights", track, timestamp)

        local_angle = float(local_angle % 360.0)
        if not math.isfinite(local_angle) or (elevation is not None and not math.isfinite(elevation)):
            return self._miss("invalid_weights", track, timestamp)
        spectrum = np.nan_to_num(np.asarray(spectrum, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)

        raw_local = local_angle
        raw_elevation = elevation
        confidence = 0.5
        elevation_confidence = 0.5 if elevation is not None else 0.0
        stable = False
        elevation_stable = False
        if track:
            self.tracker.add(raw_local, prominence, timestamp=timestamp)
            if raw_elevation is not None:
                self.elevation_tracker.add(raw_elevation, elevation_prominence, timestamp=timestamp)
            fresh = self.tracker.current_fresh(now_value)
            if fresh and self.tracker.ready:
                local_angle = self.tracker.angle
                confidence = self.tracker.confidence
                stable = self.mount == "vertical" and confidence >= 0.85
            if raw_elevation is not None:
                elevation_fresh = self.elevation_tracker.current_fresh(now_value)
                if elevation_fresh and self.elevation_tracker.ready:
                    elevation = self.elevation_tracker.angle
                    elevation_confidence = self.elevation_tracker.confidence
                    elevation_stable = elevation_confidence >= 0.85

        global_angle = float((local_angle + self.orientation_deg) % 360.0)
        direction_vector = None
        if raw_elevation is not None:
            if elevation_prominence < self.elevation_tracker.min_prominence and not elevation_stable:
                elevation = None
            if elevation is not None:
                azimuth_rad = np.deg2rad(global_angle)
                elevation_rad = np.deg2rad(elevation)
                direction_vector = np.array([
                    np.cos(elevation_rad) * np.sin(azimuth_rad),
                    np.cos(elevation_rad) * np.cos(azimuth_rad),
                    np.sin(elevation_rad),
                ])
                direction_vector /= np.linalg.norm(direction_vector)

        return DOAResult(
            array_id=self.array_id,
            local_angle_deg=float(local_angle),
            global_angle_deg=global_angle,
            prominence=float(prominence),
            confidence=float(confidence),
            spectrum=spectrum,
            stable=bool(stable),
            elevation_deg=None if elevation is None else float(elevation),
            elevation_prominence=float(elevation_prominence),
            elevation_confidence=float(elevation_confidence),
            elevation_stable=bool(elevation_stable),
            direction_vector=direction_vector,
        )

    def reset(self):
        self.tracker.reset()
        self.elevation_tracker.reset()
