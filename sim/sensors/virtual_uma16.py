"""Virtual 3D UMA-16 V2 acoustic sensor feeding the real UMA-16 front-only DOA pipeline."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from frequency_id.acoustics.audio_manager import AudioBlock
from frequency_id.acoustics.detector import DetectionResult, FrequencyDetector
from frequency_id.acoustics.direction_suppression import (
    DirectionSuppression,
    select_front_only_target,
)
from frequency_id.acoustics.doa_engine import ArrayDOA, DOAResult
from frequency_id.acoustics.geometry import RigGeometry, SOUND_SPEED
from frequency_id.telemetry import build_guidance_payload


@dataclass
class AcousticSensorConfig:
    """Configurable impairments and physics parameters for the virtual UMA-16 V2."""

    target_freq_hz: float = 1010.0
    band_low_hz: float = 800.0
    band_high_hz: float = 1300.0
    threshold_dbfs: float = -42.0
    average_blocks: int = 2
    snr_db: float = 20.0
    delay_s: float = 0.04
    dropout_prob: float = 0.0
    force_dropout: bool = False
    force_low_elev_prominence: bool = False
    unstable_elevation_jitter_deg: float = 0.0
    propeller_noise_db: Optional[float] = None  # relative dB above target tone; None disables
    propeller_freq_hz: float = 1180.0
    propeller_bearing_deg: float = 89.0
    propeller_elevation_deg: float = -15.0
    suppress_propeller_bearing: bool = False
    suppress_tolerance_deg: float = 10.0
    repeat_last_observation: bool = False
    use_body_attitude: bool = True
    mount_calibration_id: str = "uma16-front-body-v1"
    seed: int = 42


def ned_to_body_rig(
    delta_ned: np.ndarray,
    roll_deg: float,
    pitch_deg: float,
    yaw_deg: float,
    use_attitude: bool = True,
) -> np.ndarray:
    """Convert relative vector [dNorth, dEast, dDown] to rig body frame [x_right, y_forward, z_up]."""
    dn, de, dd = float(delta_ned[0]), float(delta_ned[1]), float(delta_ned[2])
    psi = math.radians(yaw_deg)
    theta = math.radians(pitch_deg) if use_attitude else 0.0
    phi = math.radians(roll_deg) if use_attitude else 0.0

    # 1. Yaw (N, E, D) -> (fwd_h, right_h, D)
    c_psi, s_psi = math.cos(psi), math.sin(psi)
    x1 = c_psi * dn + s_psi * de
    y1 = -s_psi * dn + c_psi * de
    z1 = dd

    # 2. Pitch Ry(-theta)
    c_th, s_th = math.cos(theta), math.sin(theta)
    x2 = c_th * x1 - s_th * z1
    y2 = y1
    z2 = s_th * x1 + c_th * z1

    # 3. Roll Rx(-phi)
    c_ph, s_ph = math.cos(phi), math.sin(phi)
    fwd = x2
    right = c_ph * y2 + s_ph * z2
    down = -s_ph * y2 + c_ph * z2

    return np.array([right, fwd, -down], dtype=float)


def ideal_body_angles(rel_rig: np.ndarray) -> Tuple[float, float, float]:
    """Return ground-truth (azimuth_deg in [-180, 180], elevation_deg, range_m) in body frame."""
    x_right, y_fwd, z_up = float(rel_rig[0]), float(rel_rig[1]), float(rel_rig[2])
    horiz = math.hypot(x_right, y_fwd)
    dist = math.hypot(horiz, z_up)
    az = math.degrees(math.atan2(x_right, y_fwd))
    el = math.degrees(math.atan2(z_up, max(1e-6, horiz)))
    return az, el, dist


class VirtualUMA16Sensor:
    """Synthesizes 16-ch audio from 3D drone/target geometry and runs the real UMA-16 pipeline."""

    def __init__(
        self,
        config_path: Optional[str] = None,
        sensor_config: Optional[AcousticSensorConfig] = None,
    ):
        self.sensor = sensor_config or AcousticSensorConfig()
        self.geo = RigGeometry(config_path)
        self.geo.set_uma16_front_only(True)
        self.arr = self.geo.arrays["left"]

        self.detector = FrequencyDetector(
            sample_rate=self.geo.sample_rate,
            block_duration=0.2,
            average_blocks=self.sensor.average_blocks,
            threshold_dbfs=self.sensor.threshold_dbfs,
        )
        self.doa_engine = ArrayDOA(
            array_id=self.arr.array_id,
            mic_positions_local=self.arr.local_mics,
            orientation_deg=self.arr.orientation_deg,
            sample_rate=self.geo.sample_rate,
            mount=self.arr.mount,
            mic_heights_local=self.arr.mic_heights_local,
            freq_lo=self.sensor.band_low_hz,
            freq_hi=self.sensor.band_high_hz,
        )
        self.suppression = DirectionSuppression(
            enabled=self.sensor.suppress_propeller_bearing,
            bearing_deg=self.sensor.propeller_bearing_deg,
            tolerance_deg=self.sensor.suppress_tolerance_deg,
            max_peak_candidates=4,
        )

        self.rng = np.random.default_rng(self.sensor.seed)
        self.n_samples = self.detector.blocksize
        self.t_vec = np.arange(self.n_samples, dtype=np.float64) / self.geo.sample_rate
        self.sequence = 0
        self._last_guidance_payload: Optional[Dict] = None
        self._last_doa: Optional[DOAResult] = None
        self._last_det: Optional[DetectionResult] = None

        # Precompute 3D mic coordinates in rig body frame: [x_right, y_forward, z_up]
        self.mic_coords_3d: Dict[int, np.ndarray] = {
            ch: np.array(
                [
                    float(self.arr.mic_positions[ch][0]),
                    float(self.arr.mic_positions[ch][1]),
                    float(self.arr.mic_heights_local[ch]),
                ],
                dtype=np.float64,
            )
            for ch in sorted(self.arr.local_mics.keys())
        }

    def reset(self):
        self.detector.reset()
        self.doa_engine.reset()
        self._last_guidance_payload = None
        self._last_doa = None
        self._last_det = None

    def _synthesize_block(
        self,
        rel_rig: np.ndarray,
        now: float,
        target_active: bool,
    ) -> np.ndarray:
        block = np.zeros((self.n_samples, 16), dtype=np.float32)
        dist_center = max(0.5, float(np.linalg.norm(rel_rig)))
        # Keep amplitude strong enough to stay above threshold_dbfs at realistic 3-25m ranges
        amp = min(0.5, 1.5 / dist_center)

        # Target signal (unless dropped out)
        dropout = self.sensor.force_dropout or (
            self.sensor.dropout_prob > 0.0 and self.rng.random() < self.sensor.dropout_prob
        )
        if target_active and not dropout:
            # Optional vertical multipath / elevation jitter perturbation
            perturbed_rel = rel_rig.copy()
            if self.sensor.unstable_elevation_jitter_deg > 0.0:
                jitter = math.radians(self.rng.normal(0.0, self.sensor.unstable_elevation_jitter_deg))
                horiz = math.hypot(perturbed_rel[0], perturbed_rel[1])
                cur_el = math.atan2(perturbed_rel[2], max(1e-6, horiz))
                r = max(0.5, float(np.linalg.norm(perturbed_rel)))
                new_el = max(-1.3, min(1.3, cur_el + jitter))
                az_rad = math.atan2(perturbed_rel[0], perturbed_rel[1])
                perturbed_rel = np.array(
                    [
                        r * math.cos(new_el) * math.sin(az_rad),
                        r * math.cos(new_el) * math.cos(az_rad),
                        r * math.sin(new_el),
                    ],
                    dtype=np.float64,
                )

            phase_offset = (now * 2.0 * math.pi * self.sensor.target_freq_hz) % (2.0 * math.pi)
            for ch, mic_pos in self.mic_coords_3d.items():
                d_ch = float(np.linalg.norm(perturbed_rel - mic_pos))
                delay_ch = d_ch / SOUND_SPEED
                sig = amp * np.sin(
                    2.0 * math.pi * self.sensor.target_freq_hz * (self.t_vec - delay_ch) + phase_offset
                )
                if self.sensor.force_low_elev_prominence:
                    # Scramble vertical phase across rows while preserving horizontal columns
                    row_phase = float(self.rng.uniform(-math.pi, math.pi)) * (mic_pos[2] / 0.063)
                    sig = amp * np.sin(
                        2.0 * math.pi * self.sensor.target_freq_hz * (self.t_vec - delay_ch)
                        + phase_offset
                        + row_phase
                    )
                block[:, ch] += sig.astype(np.float32)

        # Optional self propeller-noise interference
        if self.sensor.propeller_noise_db is not None:
            prop_amp = amp * (10.0 ** (self.sensor.propeller_noise_db / 20.0))
            p_az = math.radians(self.sensor.propeller_bearing_deg)
            p_el = math.radians(self.sensor.propeller_elevation_deg)
            p_dir = np.array(
                [
                    math.cos(p_el) * math.sin(p_az),
                    math.cos(p_el) * math.cos(p_az),
                    math.sin(p_el),
                ],
                dtype=np.float64,
            )
            for ch, mic_pos in self.mic_coords_3d.items():
                delay_p = -float(np.dot(mic_pos, p_dir)) / SOUND_SPEED
                prop_tone = prop_amp * np.sin(
                    2.0 * math.pi * self.sensor.propeller_freq_hz * (self.t_vec - delay_p)
                )
                # Broadband motor/blade turbulence
                broadband = self.rng.normal(0.0, prop_amp * 0.25, self.n_samples)
                block[:, ch] += (prop_tone + broadband).astype(np.float32)

        # Additive per-channel white sensor noise
        noise_amp = amp * (10.0 ** (-self.sensor.snr_db / 20.0))
        block += self.rng.normal(0.0, noise_amp, block.shape).astype(np.float32)
        return block

    def sample(
        self,
        now: float,
        drone_ned: np.ndarray,
        drone_rpy_deg: Tuple[float, float, float],
        target_ned: np.ndarray,
        target_active: bool = True,
    ) -> Tuple[Optional[Dict], Dict]:
        """Run synthetic 3D wavefield -> FrequencyDetector -> ArrayDOA -> build_guidance_payload.

        Returns (guidance_payload, diagnostics_dict).
        The controller MUST only consume guidance_payload.
        """
        rel_rig = ned_to_body_rig(
            target_ned - drone_ned,
            roll_deg=drone_rpy_deg[0],
            pitch_deg=drone_rpy_deg[1],
            yaw_deg=drone_rpy_deg[2],
            use_attitude=self.sensor.use_body_attitude,
        )
        ideal_az, ideal_el, true_range = ideal_body_angles(rel_rig)

        if self.sensor.repeat_last_observation and self._last_guidance_payload is not None:
            diag = {
                "ideal_azimuth_deg": ideal_az,
                "ideal_elevation_deg": ideal_el,
                "true_range_m": true_range,
                "repeated": True,
            }
            return dict(self._last_guidance_payload), diag

        samples = self._synthesize_block(rel_rig, now, target_active=target_active)
        self.sequence += 1
        captured_at = now - max(0.0, float(self.sensor.delay_s))
        audio_block = AudioBlock(samples=samples, captured_at=captured_at, sequence=self.sequence)

        band = (self.sensor.band_low_hz, self.sensor.band_high_hz)
        det_res = self.detector.process(samples[:, 0], band[0], band[1])

        self.suppression = DirectionSuppression(
            enabled=self.sensor.suppress_propeller_bearing,
            bearing_deg=self.sensor.propeller_bearing_deg,
            tolerance_deg=self.sensor.suppress_tolerance_deg,
            max_peak_candidates=4,
        )

        selection = select_front_only_target(
            det_res,
            samples,
            self.doa_engine,
            band,
            self.suppression,
        )
        doa_res = selection.doa
        if self.sensor.force_low_elev_prominence and doa_res is not None:
            doa_res.elevation_prominence = 0.1
            doa_res.elevation_stable = False

        payload = build_guidance_payload(
            array_mode="uma16_front_only",
            audio_block=audio_block,
            doa=doa_res,
            suppressed=selection.suppressed,
            mount_calibration_id=self.sensor.mount_calibration_id,
        )
        # Note: if signal was not detected by FrequencyDetector, DOA returns None -> payload has None angles
        # GuidanceController._observation_valid() marks this invalid -> TARGET_LOST_HOLD.
        self._last_guidance_payload = payload
        self._last_doa = doa_res
        self._last_det = selection.detection

        diag = {
            "ideal_azimuth_deg": ideal_az,
            "ideal_elevation_deg": ideal_el,
            "true_range_m": true_range,
            "peak_freq_hz": selection.detection.peak_freq,
            "peak_db": selection.detection.peak_db,
            "detected": selection.detection.detected,
            "suppressed": selection.suppressed,
            "repeated": False,
        }
        return payload, diag
