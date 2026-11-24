#!/usr/bin/env python3
"""Supervised PX4 Offboard adapter for UMA-16 angle guidance."""

import argparse
import asyncio
import json
import math
from pathlib import Path
import queue
import sys
import time

from frequency_id.control.guidance import (
    AcousticObservation,
    FlightState,
    GuidanceCommand,
    GuidanceConfig,
    GuidanceController,
    GuidanceState,
)


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
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        asyncio.run(run_controller(args))
    except (ValueError, RuntimeError, TimeoutError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
