"""Sensor fusion state machine, fail-closed acoustic guidance, and PX4 MAVSDK controller."""

from .guidance import (
    AcousticObservation,
    FlightState,
    GuidanceCommand,
    GuidanceConfig,
    GuidanceController,
    GuidanceState,
    wrap_angle_deg,
)
from .px4 import PX4FlightMonitor, parse_telemetry_line, validate_observation_metadata
from .state_machine import (
    FusedTargetState,
    StateMachineConfig,
    TargetStateMachine,
    TrackingMode,
    format_fused_state,
)

__all__ = [
    "AcousticObservation",
    "FlightState",
    "FusedTargetState",
    "GuidanceCommand",
    "GuidanceConfig",
    "GuidanceController",
    "GuidanceState",
    "PX4FlightMonitor",
    "StateMachineConfig",
    "TargetStateMachine",
    "TrackingMode",
    "format_fused_state",
    "parse_telemetry_line",
    "validate_observation_metadata",
    "wrap_angle_deg",
]
