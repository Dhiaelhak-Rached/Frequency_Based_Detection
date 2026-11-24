#!/usr/bin/env python3
"""Supervised PX4 Offboard adapter for UMA-16 angle guidance and takeoff diagnostics."""

import argparse
import asyncio
from dataclasses import dataclass
import json
import math
from pathlib import Path
import queue
import signal
import sys
import time
from typing import Any, Callable, Dict, Optional

from frequency_id.control.guidance import (
    AcousticObservation,
    FlightState,
    GuidanceCommand,
    GuidanceConfig,
    GuidanceController,
    GuidanceState,
)

DEFAULT_PX4_CONFIG: Dict[str, Any] = {
    "connect": "udp://:14540",
    "takeoff_altitude_m": 2.5,
    "hover_duration_s": 5.0,
    "connect_timeout_s": 10.0,
    "takeoff_timeout_s": 20.0,
    "auto_land": True,
    "min_battery_percent": 30.0,
}

MANUAL_OVERRIDE_MODES = frozenset(
    {"MANUAL", "STABILIZED", "ALTCTL", "POSCTL", "RETURN_TO_LAUNCH"}
)

ARM_CONFIRM_TIMEOUT_S = 5.0
ARM_ONLY_HOLD_S = 3.0
LAND_CONFIRM_TIMEOUT_S = 30.0
TAKEOFF_POLL_INTERVAL_S = 0.05
TAKEOFF_EMIT_INTERVAL_S = 0.2


@dataclass(frozen=True)
class TakeoffTestConfig:
    takeoff_altitude_m: float = 2.5
    hover_duration_s: float = 5.0
    connect_timeout_s: float = 10.0
    takeoff_timeout_s: float = 20.0
    min_battery_percent: float = 30.0
    arm_only: bool = False
    auto_land: bool = True
    guidance_max_altitude_m: float = 20.0

    def __post_init__(self):
        if not math.isfinite(self.guidance_max_altitude_m) or self.guidance_max_altitude_m <= 0.0:
            raise ValueError("guidance_max_altitude_m must be finite and positive")
        max_alt = min(5.0, float(self.guidance_max_altitude_m))
        if not math.isfinite(self.takeoff_altitude_m) or not (0.5 <= self.takeoff_altitude_m <= max_alt):
            raise ValueError(
                f"takeoff_altitude_m must be finite and in [0.5, {max_alt:g}]"
            )
        if not math.isfinite(self.hover_duration_s) or not (0.0 <= self.hover_duration_s <= 30.0):
            raise ValueError("hover_duration_s must be finite and in [0.0, 30.0]")
        if not math.isfinite(self.connect_timeout_s) or self.connect_timeout_s <= 0.0:
            raise ValueError("connect_timeout_s must be finite and positive")
        if not math.isfinite(self.takeoff_timeout_s) or self.takeoff_timeout_s <= 0.0:
            raise ValueError("takeoff_timeout_s must be finite and positive")
        if not math.isfinite(self.min_battery_percent) or not (0.0 <= self.min_battery_percent <= 100.0):
            raise ValueError("min_battery_percent must be finite and in [0.0, 100.0]")
        if not isinstance(self.arm_only, bool):
            raise ValueError("arm_only must be a bool")
        if not isinstance(self.auto_land, bool):
            raise ValueError("auto_land must be a bool")


def emit_test_event(phase: str, message: str, **extra) -> None:
    sanitized = {}
    for key, value in extra.items():
        if isinstance(value, float) and not math.isfinite(value):
            sanitized[key] = None
        else:
            sanitized[key] = value
    payload = {"type": "px4_test", "phase": phase, "message": message, **sanitized}
    print("@PX4_TEST:" + json.dumps(payload), flush=True)


def parse_telemetry_line(line):
    if not line.startswith("@JSON:"):
        return None
    try:
        payload = json.loads(line[6:])
    except json.JSONDecodeError:
        return None
    if payload.get("type") != "telemetry":
        return None
    observation = payload.get("guidance")
    return observation if isinstance(observation, dict) else None


def validate_observation_metadata(observation, mount_calibration_id):
    if observation.get("array_mode") != "uma16_front_only" or observation.get("array_id") != "left":
        raise ValueError("guidance requires the UMA-16 front-only acoustic observation")
    if observation.get("mount_calibration_id") != mount_calibration_id:
        raise ValueError("acoustic mounting calibration does not match controller configuration")


class PX4FlightMonitor:
    def __init__(self, drone, config):
        self.drone = drone
        self.config = config
        self.flight_mode = None
        self.position = None
        self.local_position = None
        self.battery_percent = float("nan")
        self.received_at = {}
        self.started_at = None
        self.start_position = None

    def tasks(self):
        return (
            self._watch_flight_mode(),
            self._watch_position(),
            self._watch_local_position(),
            self._watch_battery(),
        )

    def _mark(self, name):
        self.received_at[name] = time.monotonic()

    async def _watch_flight_mode(self):
        async for value in self.drone.telemetry.flight_mode():
            self.flight_mode = value
            self._mark("flight_mode")

    async def _watch_position(self):
        async for value in self.drone.telemetry.position():
            self.position = value
            self._mark("position")

    async def _watch_local_position(self):
        async for value in self.drone.telemetry.position_velocity_ned():
            self.local_position = value.position
            if self.start_position is None:
                self.start_position = value.position
                self.started_at = time.monotonic()
            self._mark("local_position")

    async def _watch_battery(self):
        async for value in self.drone.telemetry.battery():
            self.battery_percent = float(value.remaining_percent)
            self._mark("battery")

    def ready(self, now):
        # PX4 streams position/local_position at high rate (~30-50 Hz), HEARTBEAT (flight_mode)
        # at 1 Hz, and BATTERY_STATUS at 0.5 Hz (every 2.0 s), so allow 5.0 s for battery.
        timeouts = {
            "flight_mode": 2.5,
            "position": 2.0,
            "local_position": 2.0,
            "battery": 5.0,
        }
        return all(
            name in self.received_at and now - self.received_at[name] <= max_age
            for name, max_age in timeouts.items()
        )

    def flight_state(self, now):
        fresh = self.ready(now)
        if self.local_position is not None and hasattr(self.local_position, "down_m"):
            altitude = -float(self.local_position.down_m)
        elif self.position is not None:
            altitude = float(self.position.relative_altitude_m)
        else:
            altitude = float("nan")
        distance = float("inf")
        if self.local_position is not None and self.start_position is not None:
            north = float(self.local_position.north_m) - float(self.start_position.north_m)
            east = float(self.local_position.east_m) - float(self.start_position.east_m)
            distance = math.hypot(north, east)
        mode_name = getattr(self.flight_mode, "name", str(self.flight_mode or ""))
        return FlightState(
            altitude_m=altitude,
            distance_from_start_m=distance,
            battery_percent=self.battery_percent,
            offboard_active=mode_name == "OFFBOARD",
            telemetry_fresh=fresh and math.isfinite(distance),
            geofence_ok=distance <= self.config.geofence_radius_m,
        )


class _PX4TakeoffTelemetry:
    """Concurrent MAVSDK telemetry subscriber for the standalone arm/takeoff diagnostic."""

    def __init__(self, drone):
        self.drone = drone
        self.armed: Optional[bool] = None
        self.in_air: bool = False
        self.landed_state: Optional[str] = None
        self.altitude_m: float = float("nan")
        self.vz_m_s: float = float("nan")
        self.flight_mode: str = ""
        self.battery_percent: float = float("nan")
        self.battery_received: bool = False
        self.health: Any = None
        self.px4_status_text: str = ""

    def start_tasks(self):
        coros = [
            self._watch_armed(),
            self._watch_in_air(),
            self._watch_position_velocity_ned(),
            self._watch_flight_mode(),
            self._watch_battery(),
            self._watch_health(),
        ]
        if hasattr(self.drone.telemetry, "landed_state"):
            coros.append(self._watch_landed_state())
        if hasattr(self.drone.telemetry, "status_text"):
            coros.append(self._watch_status_text())
        return [asyncio.create_task(coro) for coro in coros]

    async def _watch_armed(self):
        async for value in self.drone.telemetry.armed():
            self.armed = bool(value)

    async def _watch_in_air(self):
        async for value in self.drone.telemetry.in_air():
            self.in_air = bool(value)

    async def _watch_landed_state(self):
        async for value in self.drone.telemetry.landed_state():
            self.landed_state = getattr(value, "name", str(value))

    async def _watch_position_velocity_ned(self):
        async for value in self.drone.telemetry.position_velocity_ned():
            pos = getattr(value, "position", None)
            vel = getattr(value, "velocity", None)
            if pos is not None and hasattr(pos, "down_m"):
                self.altitude_m = -float(pos.down_m)
            if vel is not None and hasattr(vel, "down_m_s"):
                self.vz_m_s = float(vel.down_m_s)

    async def _watch_flight_mode(self):
        async for value in self.drone.telemetry.flight_mode():
            self.flight_mode = getattr(value, "name", str(value or ""))

    async def _watch_battery(self):
        async for value in self.drone.telemetry.battery():
            self.battery_percent = float(value.remaining_percent)
            self.battery_received = True

    async def _watch_health(self):
        async for value in self.drone.telemetry.health():
            self.health = value

    async def _watch_status_text(self):
        async for value in self.drone.telemetry.status_text():
            text = getattr(value, "text", str(value))
            if text:
                self.px4_status_text = str(text)

    def snapshot(self) -> Dict[str, Any]:
        return {
            "armed": self.armed,
            "in_air": self.in_air,
            "landed_state": self.landed_state,
            "altitude_m": round(self.altitude_m, 3) if math.isfinite(self.altitude_m) else None,
            "vz_m_s": round(self.vz_m_s, 3) if math.isfinite(self.vz_m_s) else None,
            "flight_mode": self.flight_mode,
            "battery_percent": (
                round(self.battery_percent, 1) if math.isfinite(self.battery_percent) else None
            ),
            "px4_status_text": self.px4_status_text,
        }


def _load_takeoff_test_config(args) -> tuple[TakeoffTestConfig, str]:
    raw_cfg: Dict[str, Any] = {}
    config_path = getattr(args, "config", None)
    if config_path:
        path = Path(config_path)
        if path.is_file():
            with open(path, "r") as config_file:
                loaded = json.load(config_file)
                if isinstance(loaded, dict):
                    raw_cfg = loaded

    px4_cfg = raw_cfg.get("px4") if isinstance(raw_cfg.get("px4"), dict) else {}
    guidance_cfg = raw_cfg.get("guidance") if isinstance(raw_cfg.get("guidance"), dict) else {}

    connect_uri = getattr(args, "connect", None) or px4_cfg.get(
        "connect", DEFAULT_PX4_CONFIG["connect"]
    )
    takeoff_alt = (
        args.takeoff_alt
        if getattr(args, "takeoff_alt", None) is not None
        else px4_cfg.get("takeoff_altitude_m", DEFAULT_PX4_CONFIG["takeoff_altitude_m"])
    )
    hover_s = (
        args.hover_s
        if getattr(args, "hover_s", None) is not None
        else px4_cfg.get("hover_duration_s", DEFAULT_PX4_CONFIG["hover_duration_s"])
    )
    connect_timeout_s = (
        args.connect_timeout_s
        if getattr(args, "connect_timeout_s", None) is not None
        else px4_cfg.get("connect_timeout_s", DEFAULT_PX4_CONFIG["connect_timeout_s"])
    )
    takeoff_timeout_s = (
        args.takeoff_timeout_s
        if getattr(args, "takeoff_timeout_s", None) is not None
        else px4_cfg.get("takeoff_timeout_s", DEFAULT_PX4_CONFIG["takeoff_timeout_s"])
    )
    min_battery = px4_cfg.get(
        "min_battery_percent",
        guidance_cfg.get("min_battery_percent", DEFAULT_PX4_CONFIG["min_battery_percent"]),
    )
    auto_land = (
        args.auto_land
        if getattr(args, "auto_land", None) is not None
        else px4_cfg.get("auto_land", DEFAULT_PX4_CONFIG["auto_land"])
    )
    guidance_max_alt = float(guidance_cfg.get("max_altitude_m", 20.0))

    cfg = TakeoffTestConfig(
        takeoff_altitude_m=float(takeoff_alt),
        hover_duration_s=float(hover_s),
        connect_timeout_s=float(connect_timeout_s),
        takeoff_timeout_s=float(takeoff_timeout_s),
        min_battery_percent=float(min_battery),
        arm_only=bool(getattr(args, "arm_only", False)),
        auto_land=bool(auto_land),
        guidance_max_altitude_m=guidance_max_alt,
    )
    return cfg, str(connect_uri)


def _is_low_battery(battery_percent: float, min_battery_percent: float) -> bool:
    return math.isfinite(battery_percent) and 0.0 <= battery_percent < min_battery_percent


def _check_preflight(telemetry: _PX4TakeoffTelemetry, cfg: TakeoffTestConfig) -> list[str]:
    health = telemetry.health
    low_battery = _is_low_battery(telemetry.battery_percent, cfg.min_battery_percent)
    is_armable = bool(getattr(health, "is_armable", False))
    local_ok = bool(getattr(health, "is_local_position_ok", False))
    home_ok = bool(getattr(health, "is_home_position_ok", False))

    required_failed = low_battery or not is_armable or (
        not cfg.arm_only and (not local_ok or not home_ok)
    )
    if not required_failed:
        return []

    failing: list[str] = []
    if low_battery:
        failing.append(
            f"low_battery({telemetry.battery_percent:.1f}%<{cfg.min_battery_percent:.1f}%)"
        )
    flag_map = (
        ("gyrometer", "is_gyrometer_calibration_ok"),
        ("accelerometer", "is_accelerometer_calibration_ok"),
        ("magnetometer", "is_magnetometer_calibration_ok"),
        ("local_position", "is_local_position_ok"),
        ("global_position", "is_global_position_ok"),
        ("home_position", "is_home_position_ok"),
        ("armable", "is_armable"),
    )
    for label, attr in flag_map:
        if hasattr(health, attr):
            if not bool(getattr(health, attr)):
                failing.append(label)
        elif label in ("armable",) or (not cfg.arm_only and label in ("local_position", "home_position")):
            failing.append(label)
    return failing


async def _safe_land(drone, telemetry: _PX4TakeoffTelemetry, timeout_s: float) -> tuple[bool, str]:
    try:
        await drone.action.land()
    except Exception as exc:
        return False, f"land command failed: {exc}"
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if telemetry.flight_mode in MANUAL_OVERRIDE_MODES:
            return False, f"manual RC takeover ({telemetry.flight_mode})"
        if not telemetry.in_air and not telemetry.armed:
            return True, ""
        await asyncio.sleep(TAKEOFF_POLL_INTERVAL_S)
    return False, "timed out waiting for landing and disarm"


async def _safe_disarm(drone, telemetry: _PX4TakeoffTelemetry, timeout_s: float) -> tuple[bool, str]:
    try:
        await drone.action.disarm()
    except Exception as exc:
        return False, f"disarm command failed: {exc}"
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if telemetry.armed is False:
            return True, ""
        await asyncio.sleep(TAKEOFF_POLL_INTERVAL_S)
    return False, "timed out waiting for disarm confirmation"


async def run_takeoff_test(
    args,
    drone_factory: Optional[Callable[[], Any]] = None,
    cancel_event: Optional[asyncio.Event] = None,
) -> int:
    if not getattr(args, "authorize_arm", False):
        raise ValueError("--authorize-arm is required; arming PX4 motors is disabled by default")

    cfg, connect_uri = _load_takeoff_test_config(args)

    if drone_factory is None:
        try:
            from mavsdk import System
        except ImportError as exc:
            raise RuntimeError(
                "MAVSDK-Python is required: install the project requirements"
            ) from exc
        drone_factory = System

    emit_test_event("CONNECTING", f"Connecting to PX4 at {connect_uri}", connect=connect_uri)
    drone = drone_factory()
    try:
        await drone.connect(system_address=connect_uri)
        await _wait_for_connection(drone, cfg.connect_timeout_s)
    except Exception as exc:
        emit_test_event(
            "FAILED",
            f"Failed to connect to PX4 at {connect_uri}: {exc}",
            engines_started=False,
            takeoff_succeeded=False,
        )
        return 1

    telemetry = _PX4TakeoffTelemetry(drone)
    watcher_tasks = telemetry.start_tasks()
    stop_event = cancel_event if cancel_event is not None else asyncio.Event()
    loop = asyncio.get_running_loop()
    registered_signals = []
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop_event.set)
            registered_signals.append(sig)
        except (NotImplementedError, RuntimeError):
            pass

    takeoff_succeeded = False

    try:
        emit_test_event(
            "PREFLIGHT_CHECK",
            "Waiting for PX4 health, battery, and arming telemetry",
            **telemetry.snapshot(),
        )
        preflight_deadline = time.monotonic() + cfg.connect_timeout_s
        while (
            telemetry.health is None
            or not telemetry.battery_received
            or telemetry.armed is None
        ):
            if stop_event.is_set():
                emit_test_event(
                    "ABORTED",
                    "Test cancelled during preflight telemetry wait",
                    engines_started=False,
                    takeoff_succeeded=False,
                    **telemetry.snapshot(),
                )
                return 130
            if time.monotonic() >= preflight_deadline:
                missing = []
                if telemetry.health is None:
                    missing.append("health")
                if not telemetry.battery_received:
                    missing.append("battery")
                if telemetry.armed is None:
                    missing.append("armed")
                status_suffix = (
                    f" | PX4: {telemetry.px4_status_text}" if telemetry.px4_status_text else ""
                )
                emit_test_event(
                    "FAILED",
                    f"Timed out waiting for preflight telemetry ({', '.join(missing)}){status_suffix}",
                    engines_started=False,
                    takeoff_succeeded=False,
                    **telemetry.snapshot(),
                )
                return 1
            await asyncio.sleep(TAKEOFF_POLL_INTERVAL_S)

        failing_checks = _check_preflight(telemetry, cfg)
        if failing_checks:
            status_suffix = (
                f" | PX4: {telemetry.px4_status_text}" if telemetry.px4_status_text else ""
            )
            emit_test_event(
                "FAILED",
                f"Preflight check failed: {', '.join(failing_checks)}{status_suffix}",
                failing_checks=failing_checks,
                engines_started=False,
                takeoff_succeeded=False,
                **telemetry.snapshot(),
            )
            return 1

        if stop_event.is_set():
            emit_test_event(
                "ABORTED",
                "Test cancelled before arming",
                engines_started=False,
                takeoff_succeeded=False,
                **telemetry.snapshot(),
            )
            return 130

        emit_test_event("ARMING", "Sending arm command to PX4", **telemetry.snapshot())
        try:
            await drone.action.arm()
        except Exception as exc:
            status_suffix = (
                f" | PX4: {telemetry.px4_status_text}" if telemetry.px4_status_text else ""
            )
            emit_test_event(
                "FAILED",
                f"Arm command rejected: {exc}{status_suffix}",
                engines_started=False,
                takeoff_succeeded=False,
                **telemetry.snapshot(),
            )
            return 1

        arm_deadline = time.monotonic() + ARM_CONFIRM_TIMEOUT_S
        while telemetry.armed is not True:
            if stop_event.is_set():
                if telemetry.armed:
                    emit_test_event("DISARMING", "Disarming after cancel", **telemetry.snapshot())
                    await _safe_disarm(drone, telemetry, ARM_CONFIRM_TIMEOUT_S)
                emit_test_event(
                    "ABORTED",
                    "Test cancelled while waiting for arm confirmation",
                    engines_started=bool(telemetry.armed),
                    takeoff_succeeded=False,
                    **telemetry.snapshot(),
                )
                return 130
            if time.monotonic() >= arm_deadline:
                status_suffix = (
                    f" | PX4: {telemetry.px4_status_text}" if telemetry.px4_status_text else ""
                )
                emit_test_event(
                    "FAILED",
                    f"Timed out waiting for PX4 arm confirmation{status_suffix}",
                    engines_started=False,
                    takeoff_succeeded=False,
                    **telemetry.snapshot(),
                )
                return 1
            await asyncio.sleep(TAKEOFF_POLL_INTERVAL_S)

        emit_test_event(
            "ENGINES_RUNNING",
            "PX4 armed; engines running",
            engines_started=True,
            takeoff_succeeded=False,
            **telemetry.snapshot(),
        )

        if cfg.arm_only:
            hold_deadline = time.monotonic() + ARM_ONLY_HOLD_S
            while time.monotonic() < hold_deadline:
                if stop_event.is_set():
                    break
                if telemetry.armed is False:
                    emit_test_event(
                        "FAILED",
                        "PX4 disarmed unexpectedly during arm-only bench test",
                        engines_started=True,
                        takeoff_succeeded=False,
                        arm_only=True,
                        **telemetry.snapshot(),
                    )
                    return 1
                await asyncio.sleep(TAKEOFF_POLL_INTERVAL_S)

            emit_test_event(
                "DISARMING",
                "Disarming PX4 after bench arm test",
                engines_started=True,
                takeoff_succeeded=False,
                arm_only=True,
                **telemetry.snapshot(),
            )
            disarmed_ok, disarm_err = await _safe_disarm(drone, telemetry, ARM_CONFIRM_TIMEOUT_S)
            if stop_event.is_set():
                emit_test_event(
                    "ABORTED",
                    "Arm-only bench test cancelled; PX4 disarmed",
                    engines_started=True,
                    takeoff_succeeded=False,
                    arm_only=True,
                    **telemetry.snapshot(),
                )
                return 130
            if not disarmed_ok:
                emit_test_event(
                    "FAILED",
                    f"Failed to disarm after bench arm test: {disarm_err}",
                    engines_started=True,
                    takeoff_succeeded=False,
                    arm_only=True,
                    **telemetry.snapshot(),
                )
                return 1
            emit_test_event(
                "PASSED",
                "Arm-only bench test passed",
                engines_started=True,
                takeoff_succeeded=False,
                arm_only=True,
                **telemetry.snapshot(),
            )
            return 0

        # Takeoff mode
        try:
            await drone.action.set_takeoff_altitude(cfg.takeoff_altitude_m)
            await drone.action.takeoff()
        except Exception as exc:
            status_suffix = (
                f" | PX4: {telemetry.px4_status_text}" if telemetry.px4_status_text else ""
            )
            if telemetry.in_air:
                emit_test_event("LANDING", "Landing after takeoff command error", **telemetry.snapshot())
                await _safe_land(drone, telemetry, LAND_CONFIRM_TIMEOUT_S)
            elif telemetry.armed:
                emit_test_event("DISARMING", "Disarming after takeoff command error", **telemetry.snapshot())
                await _safe_disarm(drone, telemetry, ARM_CONFIRM_TIMEOUT_S)
            emit_test_event(
                "FAILED",
                f"Takeoff command failed: {exc}{status_suffix}",
                engines_started=True,
                takeoff_succeeded=False,
                **telemetry.snapshot(),
            )
            return 1

        takeoff_deadline = time.monotonic() + cfg.takeoff_timeout_s
        last_emit = 0.0
        min_confirm_alt = max(0.4, cfg.takeoff_altitude_m - 0.35, 0.8 * cfg.takeoff_altitude_m)
        while time.monotonic() < takeoff_deadline:
            if telemetry.flight_mode in MANUAL_OVERRIDE_MODES:
                emit_test_event(
                    "ABORTED",
                    f"Manual RC takeover detected ({telemetry.flight_mode}); stopping commands",
                    engines_started=True,
                    takeoff_succeeded=False,
                    **telemetry.snapshot(),
                )
                return 130
            if stop_event.is_set():
                break

            now = time.monotonic()
            if now - last_emit >= TAKEOFF_EMIT_INTERVAL_S:
                emit_test_event(
                    "TAKING_OFF",
                    f"Climbing toward {cfg.takeoff_altitude_m:.2f} m",
                    target_altitude_m=cfg.takeoff_altitude_m,
                    engines_started=True,
                    takeoff_succeeded=False,
                    **telemetry.snapshot(),
                )
                last_emit = now

            alt = telemetry.altitude_m
            vz = telemetry.vz_m_s
            if (
                telemetry.in_air
                and math.isfinite(alt)
                and alt >= min_confirm_alt
                and math.isfinite(vz)
                and abs(vz) < 0.35
            ):
                takeoff_succeeded = True
                break
            await asyncio.sleep(TAKEOFF_POLL_INTERVAL_S)

        if stop_event.is_set() and not takeoff_succeeded:
            if telemetry.in_air:
                emit_test_event("LANDING", "Landing after cancel during takeoff", **telemetry.snapshot())
                await _safe_land(drone, telemetry, LAND_CONFIRM_TIMEOUT_S)
            elif telemetry.armed:
                emit_test_event("DISARMING", "Disarming after cancel during takeoff", **telemetry.snapshot())
                await _safe_disarm(drone, telemetry, ARM_CONFIRM_TIMEOUT_S)
            emit_test_event(
                "ABORTED",
                "Takeoff test cancelled during climb",
                engines_started=True,
                takeoff_succeeded=False,
                **telemetry.snapshot(),
            )
            return 130

        if not takeoff_succeeded:
            if telemetry.in_air:
                emit_test_event("LANDING", "Landing after takeoff timeout", **telemetry.snapshot())
                await _safe_land(drone, telemetry, LAND_CONFIRM_TIMEOUT_S)
            elif telemetry.armed:
                emit_test_event("DISARMING", "Disarming after takeoff timeout", **telemetry.snapshot())
                await _safe_disarm(drone, telemetry, ARM_CONFIRM_TIMEOUT_S)
            status_suffix = (
                f" | PX4: {telemetry.px4_status_text}" if telemetry.px4_status_text else ""
            )
            emit_test_event(
                "FAILED",
                f"Timed out confirming takeoff to {cfg.takeoff_altitude_m:.2f} m{status_suffix}",
                engines_started=True,
                takeoff_succeeded=False,
                **telemetry.snapshot(),
            )
            return 1

        emit_test_event(
            "AIRBORNE_HOVER",
            f"Hovering at {telemetry.altitude_m:.2f} m for {cfg.hover_duration_s:.1f} s",
            engines_started=True,
            takeoff_succeeded=True,
            **telemetry.snapshot(),
        )

        hover_deadline = time.monotonic() + cfg.hover_duration_s
        low_battery_abort = False
        while time.monotonic() < hover_deadline:
            if telemetry.flight_mode in MANUAL_OVERRIDE_MODES:
                emit_test_event(
                    "ABORTED",
                    f"Manual RC takeover detected ({telemetry.flight_mode}); stopping commands",
                    engines_started=True,
                    takeoff_succeeded=True,
                    **telemetry.snapshot(),
                )
                return 130
            if stop_event.is_set():
                break
            if _is_low_battery(telemetry.battery_percent, cfg.min_battery_percent):
                low_battery_abort = True
                break
            await asyncio.sleep(TAKEOFF_POLL_INTERVAL_S)

        if low_battery_abort:
            emit_test_event(
                "LANDING",
                f"Low battery ({telemetry.battery_percent:.1f}%); landing immediately",
                engines_started=True,
                takeoff_succeeded=True,
                **telemetry.snapshot(),
            )
            await _safe_land(drone, telemetry, LAND_CONFIRM_TIMEOUT_S)
            emit_test_event(
                "FAILED",
                f"Aborted hover due to low battery ({telemetry.battery_percent:.1f}%)",
                engines_started=True,
                takeoff_succeeded=True,
                **telemetry.snapshot(),
            )
            return 1

        if stop_event.is_set():
            if telemetry.in_air:
                emit_test_event(
                    "LANDING",
                    "Cancel received; landing safely",
                    engines_started=True,
                    takeoff_succeeded=True,
                    **telemetry.snapshot(),
                )
                await _safe_land(drone, telemetry, LAND_CONFIRM_TIMEOUT_S)
            elif telemetry.armed:
                emit_test_event(
                    "DISARMING",
                    "Cancel received; disarming safely",
                    engines_started=True,
                    takeoff_succeeded=True,
                    **telemetry.snapshot(),
                )
                await _safe_disarm(drone, telemetry, ARM_CONFIRM_TIMEOUT_S)
            emit_test_event(
                "ABORTED",
                "Takeoff test cancelled by operator",
                engines_started=True,
                takeoff_succeeded=True,
                **telemetry.snapshot(),
            )
            return 130

        if cfg.auto_land:
            emit_test_event(
                "LANDING",
                "Auto-landing after hover",
                engines_started=True,
                takeoff_succeeded=True,
                **telemetry.snapshot(),
            )
            landed_ok, land_err = await _safe_land(drone, telemetry, LAND_CONFIRM_TIMEOUT_S)
            if not landed_ok:
                if telemetry.flight_mode in MANUAL_OVERRIDE_MODES:
                    emit_test_event(
                        "ABORTED",
                        f"Manual RC takeover during landing ({telemetry.flight_mode})",
                        engines_started=True,
                        takeoff_succeeded=True,
                        **telemetry.snapshot(),
                    )
                    return 130
                emit_test_event(
                    "FAILED",
                    f"Auto-land failed: {land_err}",
                    engines_started=True,
                    takeoff_succeeded=True,
                    **telemetry.snapshot(),
                )
                return 1
            emit_test_event(
                "PASSED",
                "PX4 arm, takeoff, hover, and auto-land test passed",
                engines_started=True,
                takeoff_succeeded=True,
                **telemetry.snapshot(),
            )
            return 0

        emit_test_event(
            "PASSED",
            "PX4 arm and takeoff test passed; vehicle hovering in HOLD",
            hovering=True,
            engines_started=True,
            takeoff_succeeded=True,
            **telemetry.snapshot(),
        )
        return 0
    finally:
        for sig in registered_signals:
            try:
                loop.remove_signal_handler(sig)
            except (NotImplementedError, RuntimeError):
                pass
        for task in watcher_tasks:
            task.cancel()
        await asyncio.gather(*watcher_tasks, return_exceptions=True)


async def _observation_reader(output):
    while True:
        line = await asyncio.to_thread(sys.stdin.readline)
        if line == "":
            while True:
                try:
                    output.get_nowait()
                except queue.Empty:
                    break
            output.put(None)
            return
        observation = parse_telemetry_line(line)
        if observation is not None:
            while True:
                try:
                    output.get_nowait()
                except queue.Empty:
                    break
            output.put(observation)


async def _wait_for_connection(drone, timeout_s):
    async def wait():
        async for state in drone.core.connection_state():
            if state.is_connected:
                return
    await asyncio.wait_for(wait(), timeout=timeout_s)


async def run_controller(args):
    if not args.authorize_offboard:
        raise ValueError("--authorize-offboard is required; automatic flight is disabled by default")
    if sys.stdin.isatty():
        raise ValueError("Pipe timestamped main.py --stream-json telemetry into this controller")
    with open(args.config, "r") as config_file:
        settings = json.load(config_file).get("guidance", {})
    config = GuidanceConfig.from_mapping(settings)
    motion_enabled = bool(settings.get("enabled", False)) and args.enable_motion
    if args.enable_motion and not settings.get("enabled", False):
        raise ValueError("Motion requires both guidance.enabled=true and --enable-motion")
    control_rate_hz = float(settings.get("control_rate_hz", 20.0))
    if not math.isfinite(control_rate_hz) or not 1.0 <= control_rate_hz <= 100.0:
        raise ValueError("guidance.control_rate_hz must be between 1 and 100")
    mount_calibration_id = settings.get("mount_calibration_id")
    if motion_enabled and not mount_calibration_id:
        raise ValueError("Motion requires guidance.mount_calibration_id")

    try:
        from mavsdk import System
        from mavsdk.offboard import VelocityBodyYawspeed
    except ImportError as exc:
        raise RuntimeError("MAVSDK-Python is required: install the project requirements") from exc

    drone = System()
    await drone.connect(system_address=args.connect)
    await _wait_for_connection(drone, args.connect_timeout_s)
    monitor = PX4FlightMonitor(drone, config)
    monitor_tasks = [asyncio.create_task(task) for task in monitor.tasks()]
    observations = queue.Queue(maxsize=1)
    reader_task = asyncio.create_task(_observation_reader(observations))
    guidance = GuidanceController(config)
    offboard_started = False

    try:
        deadline = time.monotonic() + args.connect_timeout_s
        while not monitor.ready(time.monotonic()):
            if time.monotonic() >= deadline:
                raise TimeoutError("Timed out waiting for fresh PX4 flight telemetry")
            await asyncio.sleep(0.05)

        await drone.offboard.set_velocity_body(VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0))
        await drone.offboard.start()
        offboard_started = True
        # Wait for the 1 Hz MAVLink HEARTBEAT to confirm OFFBOARD while keeping setpoints alive
        offboard_deadline = time.monotonic() + 2.5
        while getattr(monitor.flight_mode, "name", str(monitor.flight_mode or "")) != "OFFBOARD":
            if time.monotonic() >= offboard_deadline:
                raise RuntimeError("PX4 did not confirm OFFBOARD flight mode")
            await drone.offboard.set_velocity_body(VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0))
            await asyncio.sleep(1.0 / control_rate_hz)
        if motion_enabled:
            guidance.enable()

        latest_observation = None
        stream_closed = False
        last_report = 0.0
        control_period = 1.0 / control_rate_hz
        while True:
            now = time.monotonic()
            try:
                while True:
                    incoming = observations.get_nowait()
                    if incoming is None:
                        stream_closed = True
                    else:
                        latest_observation = incoming
            except queue.Empty:
                pass

            flight = monitor.flight_state(now)
            if not flight.offboard_active:
                command = guidance.abort("offboard_lost_or_manual_override")
            elif not flight.telemetry_fresh:
                command = guidance.abort("flight_telemetry_stale")
            elif stream_closed:
                command = guidance.abort("acoustic_stream_closed")
            elif motion_enabled:
                try:
                    if latest_observation is not None:
                        validate_observation_metadata(latest_observation, mount_calibration_id)
                    command = guidance.update(
                        now,
                        (
                            AcousticObservation.from_mapping(latest_observation)
                            if latest_observation is not None
                            else None
                        ),
                        flight,
                    )
                except (TypeError, ValueError) as exc:
                    command = guidance.abort(f"invalid_acoustic_telemetry:{exc}")
            else:
                command = GuidanceCommand()

            if command.state != GuidanceState.ABORTED and flight.offboard_active:
                await drone.offboard.set_velocity_body(
                    VelocityBodyYawspeed(
                        command.forward_m_s,
                        command.right_m_s,
                        command.down_m_s,
                        command.yaw_rate_deg_s,
                    )
                )

            if command.state == GuidanceState.ABORTED or now - last_report >= 1.0:
                print(
                    f"[GUIDANCE] {command.state.value} {command.reason} "
                    f"forward={command.forward_m_s:.2f} down={command.down_m_s:.2f} "
                    f"yaw_rate={command.yaw_rate_deg_s:.2f}",
                    file=sys.stderr,
                    flush=True,
                )
                last_report = now
            if command.state == GuidanceState.ABORTED:
                break
            await asyncio.sleep(control_period)
    finally:
        if offboard_started and monitor.flight_mode is not None:
            mode_name = getattr(monitor.flight_mode, "name", str(monitor.flight_mode))
            if mode_name == "OFFBOARD":
                try:
                    await drone.offboard.set_velocity_body(VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0))
                    await drone.offboard.stop()
                except Exception as exc:
                    print(f"[WARNING] Failed to stop Offboard cleanly: {exc}", file=sys.stderr)
        reader_task.cancel()
        for task in monitor_tasks:
            task.cancel()
        await asyncio.gather(reader_task, *monitor_tasks, return_exceptions=True)


def build_parser():
    from frequency_id.paths import get_default_rig_config_path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--connect", default="udp://:14540", help="MAVSDK PX4 connection address")
    parser.add_argument("--config", default=str(get_default_rig_config_path()))
    parser.add_argument("--connect-timeout-s", type=float, default=10.0)
    parser.add_argument("--authorize-offboard", action="store_true", help="authorize hover-only Offboard entry")
    parser.add_argument("--enable-motion", action="store_true", help="allow guidance motion when also enabled in config")
    parser.add_argument(
        "--test-takeoff",
        action="store_true",
        help="run standalone PX4 engine-arm and takeoff diagnostic test",
    )
    parser.add_argument(
        "--authorize-arm",
        action="store_true",
        help="explicitly authorize arming PX4 motors during takeoff test",
    )
    parser.add_argument(
        "--arm-only",
        action="store_true",
        help="bench test: arm engines briefly and disarm without taking off",
    )
    parser.add_argument(
        "--takeoff-alt",
        type=float,
        default=None,
        help="target takeoff altitude in meters (defaults to px4.takeoff_altitude_m)",
    )
    parser.add_argument(
        "--hover-s",
        type=float,
        default=None,
        help="hover duration in seconds after takeoff before landing/finishing",
    )
    parser.add_argument(
        "--auto-land",
        dest="auto_land",
        action="store_true",
        default=None,
        help="automatically land after hover completes",
    )
    parser.add_argument(
        "--no-auto-land",
        dest="auto_land",
        action="store_false",
        help="hold hover after takeoff instead of auto-landing",
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        if args.test_takeoff:
            return int(asyncio.run(run_takeoff_test(args)))
        asyncio.run(run_controller(args))
    except (ValueError, RuntimeError, TimeoutError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
