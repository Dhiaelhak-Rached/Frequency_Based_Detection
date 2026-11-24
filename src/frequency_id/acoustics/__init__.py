"""Acoustic array geometry, capture, detection, DOA, triangulation, and audio classification."""

from .audio_manager import AudioBlock, MultiDeviceAudioManager
from .classifier import ClassificationResult, DroneClassifier
from .detector import DetectionResult, FrequencyDetector
from .direction_suppression import (
    DirectionSuppression,
    displayable_doa,
    select_front_only_target,
)
from .doa_engine import ArrayDOA, DOAResult, SOUND_SPEED
from .geometry import ArrayGeometry, RigGeometry, bearing_to_vector, vector_to_bearing
from .triangulation import TriangulationResult, Triangulator

__all__ = [
    "ArrayDOA",
    "ArrayGeometry",
    "AudioBlock",
    "ClassificationResult",
    "DOAResult",
    "DetectionResult",
    "DirectionSuppression",
    "DroneClassifier",
    "FrequencyDetector",
    "MultiDeviceAudioManager",
    "RigGeometry",
    "SOUND_SPEED",
    "TriangulationResult",
    "Triangulator",
    "bearing_to_vector",
    "displayable_doa",
    "select_front_only_target",
    "vector_to_bearing",
]
