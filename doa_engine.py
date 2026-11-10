#!/usr/bin/env python3
"""Narrowband SRP-PHAT Direction of Arrival (DOA) engine with frequency-gated steering.

Steers phase transforms exclusively on target frequency bins, rejecting all
out-of-band acoustic energy (speech, wind, ambient room noise). Includes circular
temporal tracking with jump detection.
"""

from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

SOUND_SPEED = 343.0  # m/s


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


class CircularTracker:
    """Wrap-around safe temporal smoothing using circular statistics."""

    def __init__(
        self,
        window: int = 5,
        min_prominence: float = 2.0,
        decay: float = 0.75,
        jump_deg: float = 40.0,
        min_samples: int = 2,
    ):
        self.window = window
        self.min_prominence = min_prominence
        self.decay = decay
        self.jump_deg = jump_deg
        self.min_samples = min_samples
        self.estimates = deque(maxlen=window)

    def _circular_diff(self, a_deg: float, b_deg: float) -> float:
        return abs(((a_deg - b_deg + 180.0) % 360.0) - 180.0)

    def add(self, angle_deg: Optional[float], prominence: float):
        if angle_deg is None or prominence < self.min_prominence:
            return
        cur = self.angle
        if cur is not None and self._circular_diff(angle_deg, cur) > self.jump_deg:
            self.estimates.clear()  # Jump detected: flush stale history
        self.estimates.append(np.deg2rad(angle_deg))

    def _weighted_vector(self) -> complex:
        th = np.array(self.estimates)
        n = len(th)
        weights = self.decay ** np.arange(n - 1, -1, -1)
        z = (weights * np.exp(1j * th)).sum() / weights.sum()
        return z

    @property
    def ready(self) -> bool:
        return len(self.estimates) >= min(self.window, max(1, self.min_samples))

    @property
    def angle(self) -> Optional[float]:
        if not self.estimates:
            return None
        return float(np.rad2deg(np.angle(self._weighted_vector())) % 360.0)

    @property
    def confidence(self) -> float:
        if not self.estimates:
            return 0.0
        return float(min(abs(self._weighted_vector()), 1.0))

    def reset(self):
        self.estimates.clear()


class ElevationTracker:
    def __init__(
        self,
        window: int = 5,
        min_prominence: float = 0.45,
        decay: float = 0.75,
        jump_deg: float = 40.0,
        min_samples: int = 3,
    ):
        self.window = window
        self.min_prominence = min_prominence
        self.decay = decay
        self.jump_deg = jump_deg
        self.min_samples = min_samples
        self.estimates = deque(maxlen=window)

    def add(self, angle_deg: Optional[float], prominence: float):
        if angle_deg is None or prominence < self.min_prominence:
            return
        current = self.angle
        if current is not None and abs(angle_deg - current) > self.jump_deg:
            self.estimates.clear()
        self.estimates.append(float(angle_deg))

    @property
    def ready(self) -> bool:
        return len(self.estimates) >= min(self.window, max(1, self.min_samples))

    @property
    def angle(self) -> Optional[float]:
        if not self.estimates:
            return None
        values = np.asarray(self.estimates)
        weights = self.decay ** np.arange(len(values) - 1, -1, -1)
        return float(np.sum(values * weights) / np.sum(weights))

    @property
    def confidence(self) -> float:
        if not self.estimates:
            return 0.0
        values = np.asarray(self.estimates)
        weights = self.decay ** np.arange(len(values) - 1, -1, -1)
        mean = np.sum(values * weights) / np.sum(weights)
        std = np.sqrt(np.sum(weights * (values - mean) ** 2) / np.sum(weights))
        return float(np.clip(1.0 - std / 45.0, 0.0, 1.0))

    def reset(self):
        self.estimates.clear()


class ArrayDOA:
    """SRP-PHAT DOA estimator for an individual array with frequency-bin gating."""

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
    ):
        self.array_id = array_id
        self.mic_positions = mic_positions_local
        self.mic_heights = mic_heights_local or {}
        self.orientation_deg = orientation_deg
        self.sample_rate = sample_rate
        self.grid_deg = grid_deg
        self.freq_lo = freq_lo
        self.freq_hi = freq_hi
        self.mount = mount
        if self.mount not in ("flat", "vertical"):
            raise ValueError(f"Unsupported array mount: {self.mount!r}")

        self.channels = sorted(self.mic_positions.keys())
        positions = np.stack([self.mic_positions[ch] for ch in self.channels])
        if self.mount == "vertical":
            horizontal_positions = np.unique(np.round(positions[:, 1], decimals=9))
            if len(horizontal_positions) < 2:
                raise ValueError("A vertical array needs at least two horizontal positions")
            self.nominal_spacing_m = float(np.max(np.diff(horizontal_positions)))
        else:
            distances = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=2)
            np.fill_diagonal(distances, np.inf)
            nearest_spacings = distances.min(axis=1)
            self.nominal_spacing_m = float(np.median(nearest_spacings[np.isfinite(nearest_spacings)]))
        self.max_direction_freq_hz = SOUND_SPEED / (2.0 * self.nominal_spacing_m)
        if self.mount == "vertical":
            self.angles = np.arange(-90.0, 90.0 + grid_deg, grid_deg) % 360.0
        else:
            self.angles = np.arange(0, 360, grid_deg)
        self.n_angles = len(self.angles)

        self.delays = None
        if channel_delays:
            self.delays = np.array([channel_delays.get(ch, 0.0) for ch in self.channels])

        self._steer = None  # (F, M, T)
        self._align = None
        self._last_n = None
        self._band_idx = None

        if self.mount == "vertical":
            self.tracker = CircularTracker(window=5, min_prominence=0.45, min_samples=3)
        else:
            self.tracker = CircularTracker(window=5, min_prominence=2.0)
        self.elevation_tracker = ElevationTracker()
        heights = np.array([self.mic_heights.get(ch, 0.0) for ch in self.channels])
        self.elevation_enabled = (
            self.mount == "vertical"
            and all(ch in self.mic_heights for ch in self.channels)
            and len(np.unique(np.round(heights, decimals=9))) >= 2
        )
        self.elevation_angles = np.arange(-90.0, 90.0 + grid_deg, grid_deg)

    def _build_steering(self, n_fft: int):
        freqs = np.fft.rfftfreq(n_fft, 1.0 / self.sample_rate)
        band_mask = (freqs >= self.freq_lo) & (freqs <= self.freq_hi)
        f = freqs[band_mask]
        self._band_idx = np.flatnonzero(band_mask)

        if self.delays is not None:
            self._align = np.exp(
                +2j * np.pi * f[:, None] * self.delays[None, :] / self.sample_rate
            ).astype(np.complex64)

        # Local mic positions (M, 2): pos[:, 0] = lx (long axis), pos[:, 1] = ly (short axis)
        pos = np.stack([self.mic_positions[ch] for ch in self.channels])  # (M, 2)
        ang_rad = np.deg2rad(self.angles)
        # Unit vectors in local frame: 0 deg = along long axis, 90 deg = across short axis
        u_long = np.cos(ang_rad)   # (T,)
        u_short = np.sin(ang_rad)  # (T,)
        # Delay tau = -(lx * u_long + ly * u_short) / c -> (M, T)
        tau = -(pos[:, 0:1] * u_long[None, :] + pos[:, 1:2] * u_short[None, :]) / SOUND_SPEED

        # Steering matrix (F, M, T)
        self._steer = np.exp(
            +2j * np.pi * f[:, None, None] * tau[None, :, :]
        ).astype(np.complex64)
        return self._steer

    def _estimate_elevation(self, A, frequencies, weights, steer_sel, peak_idx):
        aligned = A * steer_sel[:, :, peak_idx]
        heights = np.array([self.mic_heights[ch] for ch in self.channels])
        row_heights = np.unique(np.round(heights, decimals=9))
        row_spectra = []
        for height in row_heights:
            members = np.flatnonzero(np.isclose(heights, height))
            row_spectra.append(np.mean(aligned[:, members], axis=1))
        rows = np.stack(row_spectra, axis=1)
        rows = rows / (np.abs(rows) + 1e-12)

        elevation_rad = np.deg2rad(self.elevation_angles)
        tau = -(row_heights[:, None] * np.sin(elevation_rad)[None, :]) / SOUND_SPEED
        steer = np.exp(
            2j * np.pi * frequencies[:, None, None] * tau[None, :, :]
        ).astype(np.complex64)
        response = np.einsum("kr,kre->ke", rows, steer)
        power = (weights[:, None] * np.abs(response) ** 2).sum(axis=0)
        peak_idx = int(np.argmax(power))
        frac = 0.0
        if 0 < peak_idx < len(self.elevation_angles) - 1:
            y0, y1, y2 = power[peak_idx - 1:peak_idx + 2]
            denom = y0 - 2 * y1 + y2
            if denom < 0:
                frac = float(np.clip(0.5 * (y0 - y2) / denom, -0.5, 0.5))
        elevation = float(self.elevation_angles[peak_idx] + frac * self.grid_deg)
        prominence = float((power.max() - np.median(power)) / (np.std(power) + 1e-12))
        return elevation, prominence

    def estimate(
        self,
        block: np.ndarray,
        target_freq: Optional[float] = None,
        target_bandwidth_hz: float = 60.0,
        band: Optional[Tuple[float, float]] = None,
    ) -> DOAResult:
        """Estimate DOA. Fully controlled by GUI band [low, high] or specific target_freq."""
        if target_freq is not None and target_freq > self.max_direction_freq_hz:
            self.tracker.reset()
            self.elevation_tracker.reset()
            return DOAResult(self.array_id, None, None, 0.0, 0.0, None, "spatial_aliasing")
        if target_freq is None and band is not None and band[0] > self.max_direction_freq_hz:
            self.tracker.reset()
            self.elevation_tracker.reset()
            return DOAResult(self.array_id, None, None, 0.0, 0.0, None, "spatial_aliasing")

        # Dynamically adapt frequency band from GUI band or detected target_freq
        if band is not None:
            low_hz, high_hz = float(band[0]), float(band[1])
            needed_lo = max(20.0, low_hz - 25.0)
            needed_hi = min(self.sample_rate / 2.0 - 50.0, high_hz + 25.0)
            if abs(self.freq_lo - needed_lo) > 5.0 or abs(self.freq_hi - needed_hi) > 5.0:
                self.freq_lo = needed_lo
                self.freq_hi = needed_hi
                self._last_n = None  # Rebuild steering grid for exact user band
        elif target_freq is not None:
            needed_hi = min(self.sample_rate / 2.0 - 50.0, target_freq + target_bandwidth_hz)
            needed_lo = max(20.0, target_freq - target_bandwidth_hz)
            if needed_hi > self.freq_hi or needed_lo < self.freq_lo:
                self.freq_hi = max(self.freq_hi, needed_hi)
                self.freq_lo = min(self.freq_lo, needed_lo)
                self._last_n = None  # Rebuild steering grid

        n = block.shape[0]
        if n != self._last_n:
            self._build_steering(n)
            self._last_n = n

        # Extract only relevant channels (N, M)
        x = np.stack([block[:, ch] for ch in self.channels], axis=1)
        X = np.fft.rfft(x * np.hanning(n)[:, None], axis=0)  # (N_freq, M)
        freqs = np.fft.rfftfreq(n, 1.0 / self.sample_rate)

        band_idx = self._band_idx
        Xb = X[band_idx, :]
        if self._align is not None:
            Xb = Xb * self._align

        fb = freqs[band_idx]
        energy = np.mean(np.abs(Xb) ** 2, axis=1)

        # Frequency-gating: isolate target tone bins
        keep = fb <= self.max_direction_freq_hz
        if target_freq is not None:
            f_lo = target_freq - target_bandwidth_hz / 2.0
            f_hi = target_freq + target_bandwidth_hz / 2.0
            keep &= (fb >= f_lo) & (fb <= f_hi)
        elif band is not None:
            keep &= (fb >= band[0]) & (fb <= band[1])

        if not np.any(keep):
            return DOAResult(self.array_id, None, None, 0.0, 0.0, None)

        # Reject bins below energy floor within the target band
        max_energy = energy[keep].max()
        keep &= energy >= max_energy * 0.05  # -13 dB from band peak

        if not np.any(keep):
            return DOAResult(self.array_id, None, None, 0.0, 0.0, None)

        X_sel = Xb[keep]
        w = energy[keep]
        w = w / w.sum()

        # Phase Transform (PHAT): normalize magnitude
        A = X_sel / (np.abs(X_sel) + 1e-12)
        steer_sel = self._steer[keep]  # (K, M, T)

        # Beamforming dot product
        s = np.einsum("km,kmt->kt", A, steer_sel)
        p = (w[:, None] * (np.abs(s) ** 2)).sum(axis=0)  # (T,)

        peak_idx = int(np.argmax(p))

        # Flat grids wrap; a vertical half-plane grid must not interpolate across its endpoints.
        frac = 0.0
        can_interpolate = self.mount != "vertical" or 0 < peak_idx < self.n_angles - 1
        if can_interpolate:
            y0 = p[(peak_idx - 1) % self.n_angles]
            y1 = p[peak_idx]
            y2 = p[(peak_idx + 1) % self.n_angles]
            denom = y0 - 2 * y1 + y2
            if denom < 0:
                frac = float(np.clip(0.5 * (y0 - y2) / denom, -0.5, 0.5))

        local_angle = (self.angles[peak_idx] + frac * self.grid_deg) % 360.0
        prominence = float((p.max() - np.median(p)) / (np.std(p) + 1e-12))

        # Add to circular tracker
        self.tracker.add(local_angle, prominence)
        smooth_local = self.tracker.angle if self.tracker.ready else local_angle
        confidence = self.tracker.confidence if self.tracker.ready else 0.5
        stable = self.mount == "vertical" and self.tracker.ready and confidence >= 0.85

        # Global bearing (relative to rig nose / +Y)
        global_angle = (smooth_local + self.orientation_deg) % 360.0

        elevation = None
        elevation_prominence = 0.0
        elevation_confidence = 0.0
        elevation_stable = False
        direction_vector = None
        if self.elevation_enabled:
            raw_elevation, elevation_prominence = self._estimate_elevation(
                A,
                fb[keep],
                w,
                steer_sel,
                peak_idx,
            )
            self.elevation_tracker.add(raw_elevation, elevation_prominence)
            smooth_elevation = (
                self.elevation_tracker.angle
                if self.elevation_tracker.ready
                else raw_elevation
            )
            elevation_confidence = (
                self.elevation_tracker.confidence if self.elevation_tracker.ready else 0.5
            )
            elevation_stable = self.elevation_tracker.ready and elevation_confidence >= 0.85
            if elevation_prominence >= 1.4 or elevation_stable:
                elevation = float(smooth_elevation)
                azimuth_rad = np.deg2rad(global_angle)
                elevation_rad = np.deg2rad(elevation)
                horizontal = np.cos(elevation_rad)
                direction_vector = np.array([
                    horizontal * np.sin(azimuth_rad),
                    horizontal * np.cos(azimuth_rad),
                    np.sin(elevation_rad),
                ])

        return DOAResult(
            array_id=self.array_id,
            local_angle_deg=float(smooth_local),
            global_angle_deg=float(global_angle),
            prominence=prominence,
            confidence=confidence,
            spectrum=p,
            stable=stable,
            elevation_deg=elevation,
            elevation_prominence=elevation_prominence,
            elevation_confidence=elevation_confidence,
            elevation_stable=elevation_stable,
            direction_vector=direction_vector,
        )

    def reset(self):
        self.tracker.reset()
        self.elevation_tracker.reset()
