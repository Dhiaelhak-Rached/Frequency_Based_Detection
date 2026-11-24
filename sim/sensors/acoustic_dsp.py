"""16-channel UMA-16 acoustic wavefield synthesizer + real FrequencyDetector/ArrayDOA DSP."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Dict, Optional, Tuple

import numpy as np

from frequency_id.acoustics.audio_manager import AudioBlock
from frequency_id.acoustics.detector import FrequencyDetector
from frequency_id.acoustics.doa_factory import create_doa_engine
from frequency_id.acoustics.geometry import RigGeometry
from frequency_id.telemetry import build_guidance_payload
from sim.sensors.virtual_uma16 import ideal_body_angles, ned_to_body_rig


@dataclass(frozen=True)
class WavefieldDSPConfig:
    sample_rate: int = 44100
    block_duration: float = 0.2
    tone_freq_hz: float = 1000.0
    band_low_hz: float = 900.0
    band_high_hz: float = 1100.0
    snr_db: float = 18.0
    threshold_dbfs: float = -65.0
    mount_calibration_id: str = "uma16-front-body-v1"
    seed: int = 123


class AcousticWavefieldDOASensor:
    """Computes real 3D DOA (azimuth & elevation) from a synthesized 16-ch UMA-16 wavefield."""

    def __init__(self, config: Optional[WavefieldDSPConfig] = None):
        self.cfg = config or WavefieldDSPConfig()
        self.geo = RigGeometry()
        self.geo.set_uma16_front_only(True)
        self.arr = self.geo.arrays["left"]

        self.rng = np.random.default_rng(self.cfg.seed)
        self.blocksize = max(128, int(round(self.cfg.sample_rate * self.cfg.block_duration)))
        self.sequence = 0
        self._t_offset = 0.0

        self.detector = FrequencyDetector(
            sample_rate=self.cfg.sample_rate,
            block_duration=self.cfg.block_duration,
            average_blocks=2,
            threshold_dbfs=self.cfg.threshold_dbfs,
        )
        self.doa = create_doa_engine(
            self.geo,
            "left",
            sample_rate=self.cfg.sample_rate,
        )
        # Body-rig 3D capsule coordinates: (right_m, forward_m=0, up_m)
        self._mic_body = {
            ch: np.array(
                [
                    float(self.arr.local_mics[ch][1]),
                    0.0,
                    float(self.arr.mic_heights_local.get(ch, 0.0)),
                ],
                dtype=float,
            )
            for ch in sorted(self.arr.local_mics)
        }

    def sample(
        self,
        now: Optional[float],
        drone_ned: np.ndarray,
        drone_rpy_deg: Tuple[float, float, float],
        target_ned: np.ndarray,
        target_active: bool = True,
    ) -> Tuple[Dict, Dict]:
        now_val = time.monotonic() if now is None else float(now)
        roll_deg, pitch_deg, yaw_deg = drone_rpy_deg
        rel_body = ned_to_body_rig(
            np.asarray(target_ned, dtype=float) - np.asarray(drone_ned, dtype=float),
            roll_deg,
            pitch_deg,
            yaw_deg,
        )
        ideal_az, ideal_el, true_range = ideal_body_angles(rel_body)

        t = self._t_offset + np.arange(self.blocksize, dtype=float) / self.cfg.sample_rate
        self._t_offset += self.blocksize / self.cfg.sample_rate

        block = np.zeros((self.blocksize, 16), dtype=np.float32)
        sig_amp = 0.25 / max(0.5, min(true_range, 10.0) * 0.25)
        noise_amp = sig_amp * (10.0 ** (-self.cfg.snr_db / 20.0))

        for ch, mic_pos in self._mic_body.items():
            if target_active:
                dist = float(np.linalg.norm(rel_body - mic_pos))
                delay_s = dist / self.geo.sound_speed
                tone = sig_amp * np.sin(2.0 * np.pi * self.cfg.tone_freq_hz * (t - delay_s))
            else:
                tone = 0.0
            noise = self.rng.normal(0.0, noise_amp, self.blocksize)
            block[:, ch] = (tone + noise).astype(np.float32)

        det_res = self.detector.process(
            block[:, 0],
            low_hz=self.cfg.band_low_hz,
            high_hz=self.cfg.band_high_hz,
        )
        doa_res = None
        if det_res.detected:
            doa_res = self.doa.estimate(
                block,
                target_freq=det_res.peak_freq,
                band=(self.cfg.band_low_hz, self.cfg.band_high_hz),
                track=True,
                captured_at=now_val,
                now=now_val,
            )

        self.sequence += 1
        audio_block = AudioBlock(samples=block, captured_at=now_val, sequence=self.sequence)
        payload = build_guidance_payload(
            array_mode="uma16_front_only",
            audio_block=audio_block,
            doa=doa_res,
            suppressed=False,
            mount_calibration_id=self.cfg.mount_calibration_id,
        )
        diag = {
            "ideal_azimuth_deg": ideal_az,
            "ideal_elevation_deg": ideal_el,
            "true_range_m": true_range,
            "detected": bool(det_res.detected),
        }
        return payload, diag
