from dataclasses import dataclass, replace
import math
from typing import Optional, Tuple

from detector import DetectionResult
from doa_engine import DOAResult


@dataclass(frozen=True)
class DirectionSuppression:
    enabled: bool = False
    bearing_deg: float = 0.0
    tolerance_deg: float = 10.0
    max_peak_candidates: int = 4

    def __post_init__(self):
        if not math.isfinite(self.bearing_deg):
            raise ValueError("Suppression bearing must be finite")
        if not math.isfinite(self.tolerance_deg) or not 0.0 < self.tolerance_deg <= 90.0:
            raise ValueError("Suppression tolerance must be greater than 0 and at most 90 degrees")
        if not 1 <= self.max_peak_candidates <= 16:
            raise ValueError("Maximum peak candidates must be between 1 and 16")
        object.__setattr__(self, "bearing_deg", self.bearing_deg % 360.0)

    @classmethod
    def from_config(
        cls,
        config,
        bearing_override: Optional[float] = None,
        tolerance_override: Optional[float] = None,
    ):
        return cls(
            enabled=bool(config.get("enabled", False)) or bearing_override is not None,
            bearing_deg=float(
                bearing_override if bearing_override is not None else config.get("bearing_deg", 0.0)
            ),
            tolerance_deg=float(
                tolerance_override if tolerance_override is not None else config.get("tolerance_deg", 10.0)
            ),
            max_peak_candidates=int(config.get("max_peak_candidates", 4)),
        )

    def contains(self, bearing_deg: Optional[float]) -> bool:
        if not self.enabled or bearing_deg is None:
            return False
        difference = abs((bearing_deg - self.bearing_deg + 180.0) % 360.0 - 180.0)
        return difference <= self.tolerance_deg


@dataclass
class TargetSelection:
    detection: DetectionResult
    doa: Optional[DOAResult]
    suppressed: bool = False


def reliable_doa(result: DOAResult) -> bool:
    return result.global_angle_deg is not None and (result.prominence >= 1.4 or result.stable)


def displayable_doa(result: Optional[DOAResult], suppression: DirectionSuppression) -> bool:
    return result is not None and not suppression.contains(result.global_angle_deg)


def _suppressed_selection(detection: DetectionResult) -> TargetSelection:
    return TargetSelection(
        replace(
            detection,
            detected=False,
            peak_freq=None,
            peak_db=None,
            peak_candidates=(),
        ),
        None,
        True,
    )


def select_front_only_target(
    detection: DetectionResult,
    block,
    doa_engine,
    band: Tuple[float, float],
    suppression: DirectionSuppression,
) -> TargetSelection:
    if not detection.detected:
        return TargetSelection(detection, None)
    if not suppression.enabled:
        result = doa_engine.estimate(block, target_freq=detection.peak_freq, band=band)
        return TargetSelection(detection, result)

    candidates = detection.peak_candidates
    if not candidates and detection.peak_freq is not None and detection.peak_db is not None:
        candidates = ((detection.peak_freq, detection.peak_db),)

    fallback = None
    for frequency, level in candidates[:suppression.max_peak_candidates]:
        if level < detection.threshold_db:
            continue
        candidate = replace(
            detection,
            detected=True,
            peak_freq=frequency,
            peak_db=level,
        )
        raw = doa_engine.estimate(
            block,
            target_freq=frequency,
            band=band,
            track=False,
        )
        if suppression.contains(raw.global_angle_deg):
            continue
        if fallback is None:
            fallback = candidate
        if not reliable_doa(raw):
            continue
        tracked = doa_engine.estimate(block, target_freq=frequency, band=band)
        if not displayable_doa(tracked, suppression):
            return _suppressed_selection(detection)
        return TargetSelection(candidate, tracked)

    if fallback is not None:
        tracked = doa_engine.estimate(
            block,
            target_freq=fallback.peak_freq,
            band=band,
        )
        if displayable_doa(tracked, suppression):
            return TargetSelection(fallback, tracked)

    return _suppressed_selection(detection)
