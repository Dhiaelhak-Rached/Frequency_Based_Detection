"""Vision-priority / acoustic-fallback target state machine.

Arbitrates between camera-based visual tracking (highest priority) and
acoustic direction-of-arrival / triangulation (fallback when vision has no
calibrated target direction).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Optional, Tuple

from classifier import ClassificationResult
from detector import DetectionResult
from doa_engine import DOAResult
from triangulation import TriangulationResult
from vision_detector import VisionResult


class TrackingMode(str, Enum):
    SEARCHING = "SEARCHING"
    ACOUSTIC_TRACKING = "ACOUSTIC_TRACKING"
    VISION_HOLD = "VISION_HOLD"
    VISION_TRACKING = "VISION_TRACKING"


@dataclass(frozen=True)
class StateMachineConfig:
    vision_hold_s: float = 0.0
    require_drone_for_audio: bool = False
    min_acoustic_prominence: float = 1.4

    def __post_init__(self):
        if self.vision_hold_s < 0.0:
            raise ValueError("vision_hold_s must be non-negative")
        if self.min_acoustic_prominence < 0.0:
            raise ValueError("min_acoustic_prominence must be non-negative")


@dataclass(frozen=True)
class FusedTargetState:
    mode: TrackingMode
    active_source: str
    bearing_deg: Optional[float]
    elevation_deg: Optional[float]
    direction_vector: Optional[Tuple[float, float, float]]
    vision_uncalibrated: bool = False
    audio_drone_confirmed: bool = False
    reason: str = ""


def bearing_elevation_to_vector(
    bearing_deg: Optional[float],
    elevation_deg: Optional[float],
) -> Optional[Tuple[float, float, float]]:
    """Convert global bearing (deg CW from +Y) and elevation (deg above horizon) to (x, y, z)."""
    if bearing_deg is None or elevation_deg is None:
        return None
    az_rad = math.radians(float(bearing_deg))
    el_rad = math.radians(float(elevation_deg))
    horizontal = math.cos(el_rad)
    return (
        horizontal * math.sin(az_rad),
        horizontal * math.cos(az_rad),
        math.sin(el_rad),
    )


def _normalize_vector(vec) -> Optional[Tuple[float, float, float]]:
    if vec is None:
        return None
    x, y, z = float(vec[0]), float(vec[1]), float(vec[2])
    norm = math.sqrt(x * x + y * y + z * z)
    if norm <= 0.0 or not math.isfinite(norm):
        return None
    return (x / norm, y / norm, z / norm)


class TargetStateMachine:
    """State machine prioritizing vision over acoustic direction with sticky vision tracking."""

    def __init__(self, config: Optional[StateMachineConfig] = None):
        self.config = config or StateMachineConfig()
        self.mode = TrackingMode.SEARCHING
        self._vision_lost_at: Optional[float] = None
        self._last_vision_state: Optional[FusedTargetState] = None

    def reset(self):
        """Reset state machine back to SEARCHING."""
        self.mode = TrackingMode.SEARCHING
        self._vision_lost_at = None
        self._last_vision_state = None

    def update(
        self,
        now: float,
        vision_has_target: bool,
        vision_bearing_deg: Optional[float] = None,
        vision_elevation_deg: Optional[float] = None,
        vision_direction_vector: Optional[Tuple[float, float, float]] = None,
        acoustic_valid: bool = False,
        acoustic_bearing_deg: Optional[float] = None,
        acoustic_elevation_deg: Optional[float] = None,
        acoustic_direction_vector: Optional[Tuple[float, float, float]] = None,
        audio_drone_confirmed: bool = False,
    ) -> FusedTargetState:
        """Advance the state machine for one step using explicit sensor inputs."""
        vision_uncalibrated = bool(vision_has_target and vision_bearing_deg is None)
        vision_ready = bool(vision_has_target and vision_bearing_deg is not None)

        if vision_ready:
            vec = _normalize_vector(vision_direction_vector) or bearing_elevation_to_vector(
                vision_bearing_deg, vision_elevation_deg
            )
            reason = "vision_kept" if self.mode == TrackingMode.VISION_TRACKING else "vision_acquired"
            self.mode = TrackingMode.VISION_TRACKING
            self._vision_lost_at = None
            state = FusedTargetState(
                mode=self.mode,
                active_source="VISION",
                bearing_deg=float(vision_bearing_deg) % 360.0,
                elevation_deg=None if vision_elevation_deg is None else float(vision_elevation_deg),
                direction_vector=vec,
                vision_uncalibrated=False,
                audio_drone_confirmed=audio_drone_confirmed,
                reason=reason,
            )
            self._last_vision_state = state
            return state

        # Vision does not have a calibrated target direction; check optional hold window.
        if (
            self.config.vision_hold_s > 0.0
            and self.mode in (TrackingMode.VISION_TRACKING, TrackingMode.VISION_HOLD)
            and self._last_vision_state is not None
        ):
            if self._vision_lost_at is None:
                self._vision_lost_at = now
            if (now - self._vision_lost_at) < self.config.vision_hold_s:
                self.mode = TrackingMode.VISION_HOLD
                held = self._last_vision_state
                return FusedTargetState(
                    mode=self.mode,
                    active_source="VISION",
                    bearing_deg=held.bearing_deg,
                    elevation_deg=held.elevation_deg,
                    direction_vector=held.direction_vector,
                    vision_uncalibrated=vision_uncalibrated,
                    audio_drone_confirmed=audio_drone_confirmed,
                    reason="vision_hold",
                )

        self._vision_lost_at = None
        self._last_vision_state = None

        audio_allowed = acoustic_valid and acoustic_bearing_deg is not None
        if self.config.require_drone_for_audio and not audio_drone_confirmed:
            audio_allowed = False

        if audio_allowed:
            vec = _normalize_vector(acoustic_direction_vector) or bearing_elevation_to_vector(
                acoustic_bearing_deg, acoustic_elevation_deg
            )
            reason = (
                "fallback_to_audio"
                if self.mode in (TrackingMode.VISION_TRACKING, TrackingMode.VISION_HOLD)
                else "acoustic_tracking"
            )
            self.mode = TrackingMode.ACOUSTIC_TRACKING
            return FusedTargetState(
                mode=self.mode,
                active_source="SOUND",
                bearing_deg=float(acoustic_bearing_deg) % 360.0,
                elevation_deg=None if acoustic_elevation_deg is None else float(acoustic_elevation_deg),
                direction_vector=vec,
                vision_uncalibrated=vision_uncalibrated,
                audio_drone_confirmed=audio_drone_confirmed,
                reason=reason,
            )

        self.mode = TrackingMode.SEARCHING
        return FusedTargetState(
            mode=self.mode,
            active_source="NONE",
            bearing_deg=None,
            elevation_deg=None,
            direction_vector=None,
            vision_uncalibrated=vision_uncalibrated,
            audio_drone_confirmed=audio_drone_confirmed,
            reason="vision_uncalibrated" if vision_uncalibrated else "searching",
        )

    def from_pipeline_outputs(
        self,
        now: float,
        vision: Optional[VisionResult] = None,
        detection: Optional[DetectionResult] = None,
        single_doa: Optional[DOAResult] = None,
        triangulation: Optional[TriangulationResult] = None,
        classification: Optional[ClassificationResult] = None,
        suppressed: bool = False,
    ) -> FusedTargetState:
        """Evaluate one state-machine step directly from pipeline result objects."""
        vision_has_target = bool(
            vision is not None and vision.error is None and vision.target is not None
        )
        v_bearing = None
        v_elevation = None
        v_vector = None
        if vision_has_target and vision is not None and vision.direction is not None:
            v_bearing = vision.direction.global_bearing_deg
            v_elevation = vision.direction.elevation_deg
            v_vector = bearing_elevation_to_vector(v_bearing, v_elevation)

        audio_drone_confirmed = bool(
            detection is not None
            and detection.detected
            and classification is not None
            and classification.is_drone
        )

        acoustic_valid = False
        a_bearing = None
        a_elevation = None
        a_vector = None

        if not suppressed and detection is not None and detection.detected:
            if single_doa is not None:
                if (
                    single_doa.error is None
                    and single_doa.global_angle_deg is not None
                    and (
                        single_doa.prominence >= self.config.min_acoustic_prominence
                        or single_doa.stable
                    )
                ):
                    acoustic_valid = True
                    a_bearing = single_doa.global_angle_deg
                    a_elevation = single_doa.elevation_deg
                    if single_doa.direction_vector is not None:
                        a_vector = _normalize_vector(single_doa.direction_vector)
            elif triangulation is not None and triangulation.success and triangulation.bearing_deg is not None:
                acoustic_valid = True
                a_bearing = triangulation.bearing_deg

        return self.update(
            now=now,
            vision_has_target=vision_has_target,
            vision_bearing_deg=v_bearing,
            vision_elevation_deg=v_elevation,
            vision_direction_vector=v_vector,
            acoustic_valid=acoustic_valid,
            acoustic_bearing_deg=a_bearing,
            acoustic_elevation_deg=a_elevation,
            acoustic_direction_vector=a_vector,
            audio_drone_confirmed=audio_drone_confirmed,
        )


def format_fused_state(state: FusedTargetState) -> str:
    """Format a FusedTargetState into the concise 5-line CLI output."""
    bearing_txt = f"{state.bearing_deg:.1f}°" if state.bearing_deg is not None else "---.-°"
    elevation_txt = f"{state.elevation_deg:.1f}°" if state.elevation_deg is not None else "---.-°"
    if state.direction_vector is not None:
        x, y, z = state.direction_vector
        vector_txt = f"({x:+.3f}, {y:+.3f}, {z:+.3f})"
    else:
        vector_txt = "N/A"
    return (
        f"MODE: {state.mode.value}\n"
        f"SOURCE: {state.active_source}\n"
        f"BEARING: {bearing_txt}\n"
        f"ELEVATION: {elevation_txt}\n"
        f"DIRECTION_VECTOR: {vector_txt}"
    )
