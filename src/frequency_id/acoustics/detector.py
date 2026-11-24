#!/usr/bin/env python3
"""Frequency detection engine with sub-bin parabolic interpolation and threshold gating.

Monitors user-specified frequency bands, calculates real FFT spectra, performs
multi-block averaging to depress random acoustic noise, and extracts the exact
peak frequency with sub-bin precision.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class DetectionResult:
    detected: bool
    peak_freq: Optional[float]
    peak_db: Optional[float]
    threshold_db: float
    band: Tuple[float, float]
    freqs: np.ndarray
    mags: np.ndarray
    peak_candidates: Tuple[Tuple[float, float], ...] = ()


class FrequencyDetector:
    """Real-time spectral analysis and target band peak extractor."""

    def __init__(
        self,
        sample_rate: int = 44100,
        block_duration: float = 0.2,
        average_blocks: int = 4,
        threshold_dbfs: float = -65.0,
    ):
        self.sample_rate = sample_rate
        self.block_duration = block_duration
        self.average_blocks = max(1, average_blocks)
        self.threshold_dbfs = threshold_dbfs

        self.blocksize = max(64, int(round(sample_rate * block_duration)))
        self.freqs = np.fft.rfftfreq(self.blocksize, 1.0 / sample_rate)
        self.bin_width = sample_rate / self.blocksize
        self.window = np.hanning(self.blocksize)
        self.window_sum = float(self.window.sum())

        self.mag_history = []

    def set_parameters(
        self,
        threshold_dbfs: Optional[float] = None,
        average_blocks: Optional[int] = None,
    ):
        if threshold_dbfs is not None:
            self.threshold_dbfs = float(threshold_dbfs)
        if average_blocks is not None:
            self.average_blocks = max(1, int(average_blocks))

    def _refine_peak(self, mags: np.ndarray, best_idx: int) -> Tuple[float, float]:
        offset = 0.0
        if 0 < best_idx < len(mags) - 1:
            a, b, c = mags[best_idx - 1], mags[best_idx], mags[best_idx + 1]
            denom = a - 2.0 * b + c
            if denom != 0:
                offset = 0.5 * (a - c) / denom
                offset = max(-1.0, min(1.0, offset))
        peak_freq = self.freqs[best_idx] + offset * self.bin_width
        return float(peak_freq), float(mags[best_idx])

    def _parabolic_peak(self, mags: np.ndarray, low_hz: float, high_hz: float) -> Tuple[Optional[float], Optional[float]]:
        """Refines peak frequency using 3-bin parabolic interpolation."""
        mask = (self.freqs >= low_hz) & (self.freqs <= high_hz)
        if not np.any(mask):
            return None, None
        indices = np.flatnonzero(mask)
        best_idx = indices[np.argmax(mags[indices])]
        return self._refine_peak(mags, best_idx)

    def _candidate_peaks(self, mags: np.ndarray, low_hz: float, high_hz: float):
        mask = (self.freqs >= low_hz) & (self.freqs <= high_hz)
        indices = np.flatnonzero(mask)
        if not len(indices):
            return ()
        best_idx = indices[np.argmax(mags[indices])]
        peak_indices = [best_idx]
        for position, index in enumerate(indices):
            left = mags[indices[position - 1]] if position > 0 else -np.inf
            right = mags[indices[position + 1]] if position + 1 < len(indices) else -np.inf
            if mags[index] > left and mags[index] >= right and index != best_idx:
                peak_indices.append(index)
        peak_indices.sort(key=lambda index: mags[index], reverse=True)
        return tuple(self._refine_peak(mags, index) for index in peak_indices)

    def process(
        self,
        audio_channel_data: np.ndarray,
        low_hz: float,
        high_hz: float,
    ) -> DetectionResult:
        """Process 1D audio segment (length >= blocksize) and detect peak in [low_hz, high_hz]."""
        x = audio_channel_data[-self.blocksize:]
        if len(x) < self.blocksize:
            x = np.pad(x, (self.blocksize - len(x), 0))

        windowed = x * self.window
        spec = np.abs(np.fft.rfft(windowed))
        amps = 2.0 * spec / (self.window_sum + 1e-12)
        amps[amps == 0] = 1e-12
        mags = 20.0 * np.log10(amps)

        if self.average_blocks > 1:
            self.mag_history.append(mags)
            self.mag_history = self.mag_history[-self.average_blocks:]
            mags = np.mean(self.mag_history, axis=0)

        peak_freq, peak_db = self._parabolic_peak(mags, low_hz, high_hz)
        detected = (peak_db is not None) and (peak_db >= self.threshold_dbfs)
        candidates = self._candidate_peaks(mags, low_hz, high_hz)

        return DetectionResult(
            detected=detected,
            peak_freq=peak_freq,
            peak_db=peak_db,
            threshold_db=self.threshold_dbfs,
            band=(low_hz, high_hz),
            freqs=self.freqs,
            mags=mags,
            peak_candidates=candidates,
        )

    def reset(self):
        self.mag_history.clear()
