"""Fail-closed angle guidance from timestamped UMA-16 front-only observations."""

from __future__ import annotations

from dataclasses import dataclass, fields
from enum import Enum
import math
from typing import Optional


def wrap_angle_deg(angle: float) -> float:
    return (float(angle) + 180.0) % 360.0 - 180.0


def _clip(value: float, limit: float) -> float:
    return max(-limit, min(limit, value))


class GuidanceState(str, Enum):
    DISABLED = "DISABLED"
    HOVER_SEARCH = "HOVER_SEARCH"
    ALIGN_YAW = "ALIGN_YAW"
    ALIGN_VERTICAL = "ALIGN_VERTICAL"
    FOLLOW = "FOLLOW"
    TARGET_LOST_HOLD = "TARGET_LOST_HOLD"
    ABORTED = "ABORTED"


@dataclass(frozen=True)
class GuidanceConfig:
    observation_timeout_s: float = 0.6
    required_stable_samples: int = 5
    filter_alpha: float = 0.35
    min_elevation_prominence: float = 0.45
    yaw_deadband_deg: float = 5.0
    yaw_exit_deg: float = 8.0
    elevation_deadband_deg: float = 4.0
    elevation_exit_deg: float = 7.0
    yaw_rate_gain: float = 0.8
    max_yaw_rate_deg_s: float = 20.0
    vertical_speed_gain: float = 0.02
    max_vertical_speed_m_s: float = 0.3
    max_forward_speed_m_s: float = 0.5
    max_vertical_accel_m_s2: float = 0.4
    max_horizontal_accel_m_s2: float = 0.5
    min_altitude_m: float = 2.0
    max_altitude_m: float = 20.0
    max_vertical_travel_m: float = 3.0
    max_follow_distance_m: float = 15.0
    max_follow_duration_s: float = 45.0
    geofence_radius_m: float = 25.0
    min_battery_percent: float = 30.0

    def __post_init__(self):
        positive = (
            "observation_timeout_s", "filter_alpha", "yaw_rate_gain",
            "vertical_speed_gain", "max_yaw_rate_deg_s",
            "max_vertical_speed_m_s", "max_forward_speed_m_s",
            "max_vertical_accel_m_s2", "max_horizontal_accel_m_s2",
            "max_vertical_travel_m", "max_follow_distance_m",
            "max_follow_duration_s", "geofence_radius_m",
        )
        for name in positive:
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        nonnegative = (
            "min_elevation_prominence", "yaw_deadband_deg", "yaw_exit_deg",
            "elevation_deadband_deg", "elevation_exit_deg",
        )
        for name in nonnegative:
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if not math.isfinite(self.min_altitude_m) or not math.isfinite(self.max_altitude_m):
            raise ValueError("Altitude limits must be finite")
        if not 0.0 < self.filter_alpha <= 1.0:
            raise ValueError("filter_alpha must be in (0, 1]")
        if not isinstance(self.required_stable_samples, int) or self.required_stable_samples < 1:
            raise ValueError("required_stable_samples must be an integer of at least 1")
        if self.yaw_exit_deg < self.yaw_deadband_deg:
            raise ValueError("yaw_exit_deg must not be smaller than yaw_deadband_deg")
        if self.elevation_exit_deg < self.elevation_deadband_deg:
            raise ValueError("elevation_exit_deg must not be smaller than elevation_deadband_deg")
        if self.max_altitude_m <= self.min_altitude_m:
            raise ValueError("max_altitude_m must exceed min_altitude_m")
        if not 0.0 <= self.min_battery_percent <= 100.0:
            raise ValueError("min_battery_percent must be in [0, 100]")

    @classmethod
    def from_mapping(cls, values):
        names = {field.name for field in fields(cls)}
        return cls(**{key: value for key, value in values.items() if key in names})


@dataclass(frozen=True)
class AcousticObservation:
    captured_at: float
    sequence: int
    azimuth_deg: Optional[float] = None
    elevation_deg: Optional[float] = None
    azimuth_prominence: float = 0.0
    elevation_prominence: float = 0.0
    azimuth_stable: bool = False
    elevation_stable: bool = False
    error: Optional[str] = None
    suppressed: bool = False

    @classmethod
    def from_mapping(cls, values):
        names = {field.name for field in fields(cls)}
        return cls(**{key: values[key] for key in names if key in values})


@dataclass(frozen=True)
class FlightState:
    altitude_m: float
    distance_from_start_m: float
    battery_percent: float
    offboard_active: bool
    telemetry_fresh: bool
    geofence_ok: bool


@dataclass(frozen=True)
class GuidanceCommand:
    forward_m_s: float = 0.0
    right_m_s: float = 0.0
    down_m_s: float = 0.0
    yaw_rate_deg_s: float = 0.0
    state: GuidanceState = GuidanceState.DISABLED
    reason: str = "disabled"


class GuidanceController:
    def __init__(self, config: Optional[GuidanceConfig] = None):
        self.config = config or GuidanceConfig()
        self.state = GuidanceState.DISABLED
        self._last_observation = None
        self._last_sequence = None
        self._filtered_azimuth = None
        self._filtered_elevation = None
        self._stable_samples = 0
        self._aligned_samples = 0
        self._vertical_anchor_m = None
        self._follow_started_at = None
        self._last_update_at = None
        self._limit_latched = False
        self._last_command = GuidanceCommand()

    def _reset_tracking(self):
        self._last_observation = None
        self._last_sequence = None
        self._filtered_azimuth = None
        self._filtered_elevation = None
        self._stable_samples = 0
        self._aligned_samples = 0
        self._vertical_anchor_m = None
        self._follow_started_at = None
        self._limit_latched = False
        self._last_command = GuidanceCommand(state=self.state)

    def enable(self):
        self._reset_tracking()
        self.state = GuidanceState.HOVER_SEARCH
        return self._hover("enabled")

    def disable(self):
        self.state = GuidanceState.DISABLED
        self._reset_tracking()
        self._last_command = GuidanceCommand(state=self.state)
        return self._last_command

    def abort(self, reason="aborted"):
        self.state = GuidanceState.ABORTED
        self._last_command = GuidanceCommand(state=self.state, reason=reason)
        return self._last_command

    def _hover(self, reason):
        return GuidanceCommand(state=self.state, reason=reason)

    def _target_lost(self, reason, latched=False):
        self.state = GuidanceState.TARGET_LOST_HOLD
        self._stable_samples = 0
        self._aligned_samples = 0
        self._filtered_azimuth = None
        self._filtered_elevation = None
        self._vertical_anchor_m = None
        self._follow_started_at = None
        self._limit_latched = latched
        self._last_command = self._hover(reason)
        return self._last_command

    def _observation_valid(self, observation):
        numeric = (
            observation.azimuth_deg, observation.elevation_deg,
            observation.azimuth_prominence, observation.elevation_prominence,
        )
        if any(value is None or not math.isfinite(float(value)) for value in numeric):
            return False
        if observation.error or observation.suppressed:
            return False
        azimuth_quality = (
            observation.azimuth_stable
            or observation.azimuth_prominence >= 1.4
        )
        elevation_quality = (
            observation.elevation_stable
            and observation.elevation_prominence >= self.config.min_elevation_prominence
        )
        return azimuth_quality and elevation_quality

    def _accept_observation(self, observation):
        azimuth = wrap_angle_deg(observation.azimuth_deg)
        elevation = float(observation.elevation_deg)
        alpha = self.config.filter_alpha
        if self._filtered_azimuth is None:
            self._filtered_azimuth = azimuth
            self._filtered_elevation = elevation
        else:
            difference = wrap_angle_deg(azimuth - self._filtered_azimuth)
            self._filtered_azimuth = wrap_angle_deg(self._filtered_azimuth + alpha * difference)
            self._filtered_elevation += alpha * (elevation - self._filtered_elevation)
        self._stable_samples += 1

    def _yaw_rate(self):
        error = self._filtered_azimuth
        if abs(error) <= self.config.yaw_deadband_deg:
            return 0.0
        return _clip(
            self.config.yaw_rate_gain * error,
            self.config.max_yaw_rate_deg_s,
        )

    def _vertical_speed(self, altitude_m):
        elevation = self._filtered_elevation
        if abs(elevation) <= self.config.elevation_deadband_deg:
            return 0.0
        down = _clip(
            -self.config.vertical_speed_gain * elevation,
            self.config.max_vertical_speed_m_s,
        )
        if altitude_m <= self.config.min_altitude_m and down > 0.0:
            return 0.0
        if altitude_m >= self.config.max_altitude_m and down < 0.0:
            return 0.0
        travel = altitude_m - self._vertical_anchor_m
        if travel >= self.config.max_vertical_travel_m and down < 0.0:
            return 0.0
        if -travel >= self.config.max_vertical_travel_m and down > 0.0:
            return 0.0
        return down

    def _command(self, forward, down, yaw_rate, reason, now, immediate_stop=False):
        target = GuidanceCommand(
            forward_m_s=forward,
            right_m_s=0.0,
            down_m_s=down,
            yaw_rate_deg_s=yaw_rate,
            state=self.state,
            reason=reason,
        )
        if immediate_stop or self._last_update_at is None:
            self._last_command = target
            return target
        dt = max(0.0, min(0.25, now - self._last_update_at))
        forward_step = self.config.max_horizontal_accel_m_s2 * dt
        vertical_step = self.config.max_vertical_accel_m_s2 * dt
        previous = self._last_command
        self._last_command = GuidanceCommand(
            forward_m_s=previous.forward_m_s + _clip(forward - previous.forward_m_s, forward_step),
            right_m_s=0.0,
            down_m_s=previous.down_m_s + _clip(down - previous.down_m_s, vertical_step),
            yaw_rate_deg_s=target.yaw_rate_deg_s,
            state=self.state,
            reason=reason,
        )
        return self._last_command

    def update(self, now, observation, flight):
        if self.state == GuidanceState.DISABLED:
            return self._hover("disabled")
        if self.state == GuidanceState.ABORTED:
            return self._hover(self._last_command.reason)
        if not flight.telemetry_fresh:
            return self.abort("flight_telemetry_stale")
        if not flight.offboard_active:
            return self.abort("offboard_lost_or_manual_override")
        if not flight.geofence_ok:
            return self.abort("geofence_violation")
        if flight.battery_percent < self.config.min_battery_percent:
            return self.abort("battery_low")
        if not all(math.isfinite(value) for value in (
            flight.altitude_m,
            flight.distance_from_start_m,
            flight.battery_percent,
        )):
            return self.abort("invalid_flight_telemetry")
        if not self.config.min_altitude_m <= flight.altitude_m <= self.config.max_altitude_m:
            return self.abort("altitude_envelope_violation")

        accepted_new = False
        if observation is not None:
            is_new = self._last_sequence is None or observation.sequence > self._last_sequence
            stale = (
                not math.isfinite(observation.captured_at)
                or now < observation.captured_at - 0.1
                or now - observation.captured_at > self.config.observation_timeout_s
            )
            if is_new and stale:
                return self._target_lost("acoustic_observation_stale")
            if is_new:
                self._last_observation = observation
                self._last_sequence = observation.sequence
                if not self._observation_valid(observation):
                    return self._target_lost("acoustic_observation_invalid")
                self._accept_observation(observation)
                accepted_new = True
        if self._last_observation is not None and (
            now - self._last_observation.captured_at > self.config.observation_timeout_s
        ):
            return self._target_lost("acoustic_observation_timeout")

        if self._filtered_azimuth is None or self._filtered_elevation is None:
            self._last_update_at = now
            return self._target_lost("waiting_for_acoustic_observation")
        if self._limit_latched:
            self._last_update_at = now
            return self._hover("motion_limit_latched")
        if self.state == GuidanceState.TARGET_LOST_HOLD:
            self.state = GuidanceState.HOVER_SEARCH
        if self.state == GuidanceState.HOVER_SEARCH:
            self._last_update_at = now
            if self._stable_samples < self.config.required_stable_samples:
                return self._hover("waiting_for_stable_observations")
            self.state = GuidanceState.ALIGN_YAW
            return self._hover("stable_observation_acquired")

        yaw_aligned = abs(self._filtered_azimuth) <= self.config.yaw_deadband_deg
        elevation_aligned = abs(self._filtered_elevation) <= self.config.elevation_deadband_deg
        if self.state == GuidanceState.ALIGN_YAW and yaw_aligned:
            self.state = GuidanceState.ALIGN_VERTICAL
            self._vertical_anchor_m = flight.altitude_m
            self._aligned_samples = 0

        if self.state == GuidanceState.FOLLOW:
            if abs(self._filtered_azimuth) > self.config.yaw_exit_deg:
                self.state = GuidanceState.ALIGN_YAW
                self._last_update_at = now
                return self._command(0.0, 0.0, self._yaw_rate(), "yaw_alignment_lost", now, True)
            if abs(self._filtered_elevation) > self.config.elevation_exit_deg:
                self.state = GuidanceState.ALIGN_VERTICAL
                self._vertical_anchor_m = flight.altitude_m
                self._aligned_samples = 0
                self._last_update_at = now
                return self._command(0.0, 0.0, self._yaw_rate(), "elevation_alignment_lost", now, True)
            if flight.distance_from_start_m >= self.config.max_follow_distance_m:
                return self._target_lost("follow_distance_limit", latched=True)
            if now - self._follow_started_at >= self.config.max_follow_duration_s:
                return self._target_lost("follow_duration_limit", latched=True)
            command = self._command(
                self.config.max_forward_speed_m_s,
                0.0,
                self._yaw_rate(),
                "following_direction_without_range",
                now,
            )
        elif self.state == GuidanceState.ALIGN_VERTICAL:
            if abs(self._filtered_azimuth) > self.config.yaw_exit_deg:
                self.state = GuidanceState.ALIGN_YAW
                self._aligned_samples = 0
            if self.state == GuidanceState.ALIGN_VERTICAL:
                if yaw_aligned and elevation_aligned:
                    if accepted_new:
                        self._aligned_samples += 1
                    if self._aligned_samples >= self.config.required_stable_samples:
                        self.state = GuidanceState.FOLLOW
                        self._follow_started_at = now
                        self._last_update_at = now
                        return self._hover("alignment_stable")
                elif accepted_new:
                    self._aligned_samples = 0
            down = self._vertical_speed(flight.altitude_m) if self.state == GuidanceState.ALIGN_VERTICAL else 0.0
            command = self._command(0.0, down, self._yaw_rate(), "aligning_vertical", now)
        else:
            command = self._command(0.0, 0.0, self._yaw_rate(), "aligning_yaw", now)

        self._last_update_at = now
        return command
