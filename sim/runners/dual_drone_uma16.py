#!/usr/bin/env python3
"""Two-drone PX4 SITL + Gazebo Garden simulation driven by a real UMA-16 V2 (no dashboard)."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
import math
from typing import Optional, Tuple

from frequency_id.control.guidance import (
    AcousticObservation,
    GuidanceCommand,
    GuidanceConfig,
    GuidanceController,
    GuidanceState,
    wrap_angle_deg,
)
from frequency_id.control.px4 import PX4FlightMonitor, validate_observation_metadata
from sim.sensors.acoustic_dsp import AcousticWavefieldDOASensor
from sim.sensors.real_uma16 import RealUMA16Config, RealUMA16Source
from sim.harness.sitl_harness import (
    AttitudeWatcher,
    MultiPX4SitlCluster,
    PX4InstanceConfig,
    apply_gui_environment,
    arm_and_hover_sitl,
    assert_sitl_endpoint,
    connect_sitl_drone,
    load_sim_guidance_config,
    set_gazebo_camera_pose,
)


@dataclass(frozen=True)
class TargetVelocityCommand:
    forward_m_s: float = 0.0
    right_m_s: float = 0.0
    down_m_s: float = 0.0
    yaw_rate_deg_s: float = 0.0
    active: bool = False


def compute_direction_command(
    azimuth_deg: float,
    elevation_deg: float,
    north_m: float,
    east_m: float,
    alt_m: float,
    yaw_deg: float,
    max_radius_m: float = 15.0,
    min_alt_m: float = 2.0,
    max_alt_m: float = 20.0,
    speed: float = 1.0,
) -> TargetVelocityCommand:
    """Convert body-relative (azimuth_deg, elevation_deg) into clamped body-velocity commands."""
    az = wrap_angle_deg(azimuth_deg)
    el = float(elevation_deg)
    az_rad = math.radians(az)
    el_rad = math.radians(el)
    cos_el = math.cos(el_rad)

    forward = speed * cos_el * math.cos(az_rad)
    right = speed * cos_el * math.sin(az_rad)
    down = -speed * math.sin(el_rad)
    if alt_m <= min_alt_m and down > 0.0:
        down = 0.0
    if alt_m >= max_alt_m and down < 0.0:
        down = 0.0

    # Keep drone within max_radius_m of its local origin so it stays in Gazebo view
    dist = math.hypot(north_m, east_m)
    if dist >= max_radius_m:
        heading_rad = math.radians(yaw_deg)
        world_vx = forward * math.cos(heading_rad) - right * math.sin(heading_rad)
        world_vy = forward * math.sin(heading_rad) + right * math.cos(heading_rad)
        if (north_m * world_vx + east_m * world_vy) > 0.0:
            forward = 0.0
            right = 0.0

    return TargetVelocityCommand(
        forward_m_s=forward,
        right_m_s=right,
        down_m_s=down,
        yaw_rate_deg_s=0.0,
        active=True,
    )


def compute_target_drone_command(
    obs: Optional[AcousticObservation],
    now: float,
    north_m: float,
    east_m: float,
    alt_m: float,
    yaw_deg: float,
    timeout_s: float = 0.8,
    max_radius_m: float = 15.0,
    min_alt_m: float = 2.0,
    max_alt_m: float = 20.0,
) -> TargetVelocityCommand:
    """Map live UMA-16 angles to Drone 2 body-velocity commands while keeping it in Gazebo view."""
    if obs is None or obs.error or obs.suppressed:
        return TargetVelocityCommand()
    if not math.isfinite(obs.captured_at) or (now - obs.captured_at) > timeout_s:
        return TargetVelocityCommand()
    if obs.azimuth_deg is None or obs.elevation_deg is None:
        return TargetVelocityCommand()
    if not (obs.azimuth_stable or obs.azimuth_prominence >= 1.4):
        return TargetVelocityCommand()

    return compute_direction_command(
        azimuth_deg=obs.azimuth_deg,
        elevation_deg=obs.elevation_deg,
        north_m=north_m,
        east_m=east_m,
        alt_m=alt_m,
        yaw_deg=yaw_deg,
        max_radius_m=max_radius_m,
        min_alt_m=min_alt_m,
        max_alt_m=max_alt_m,
    )


def apply_fixed_follow_distance(
    cmd: GuidanceCommand,
    horiz_dist_m: float,
    follow_distance_m: float = 5.0,
) -> GuidanceCommand:
    """Keep Drone 1 at a fixed horizontal separation behind Drone 2."""
    if cmd.state == GuidanceState.ABORTED:
        return cmd
    if cmd.state != GuidanceState.FOLLOW:
        return cmd
    err = horiz_dist_m - follow_distance_m
    if err <= 0.0:
        forward = -0.35 if err < -0.8 else 0.0
    else:
        forward = min(1.0, max(cmd.forward_m_s, 0.45 * err))
    return GuidanceCommand(
        forward_m_s=forward,
        right_m_s=cmd.right_m_s,
        down_m_s=cmd.down_m_s,
        yaw_rate_deg_s=cmd.yaw_rate_deg_s,
        state=cmd.state,
        reason=cmd.reason,
    )


def compute_auto_target_command(elapsed_s: float, alt_m: float, base_alt_m: float = 5.0) -> TargetVelocityCommand:
    """Gentle autonomous figure-8 / orbit velocity pattern for Drone 2 in auto-target mode."""
    omega_deg_s = 18.0
    alt_err = base_alt_m + 1.0 * math.sin(0.25 * elapsed_s) - alt_m
    down = max(-0.3, min(0.3, -0.4 * alt_err))
    return TargetVelocityCommand(
        forward_m_s=0.6,
        right_m_s=0.0,
        down_m_s=down,
        yaw_rate_deg_s=omega_deg_s * math.cos(0.2 * elapsed_s),
        active=True,
    )


async def _run_dual_drone(args, primary_uma16: RealUMA16Source, target_uma16: Optional[RealUMA16Source]):
    from mavsdk.offboard import VelocityBodyYawspeed

    inst0 = PX4InstanceConfig(instance_id=0, pose_enu="0,0,0,0,0,0", udp_port=14540, mavsdk_port=50051)
    inst1 = PX4InstanceConfig(instance_id=1, pose_enu="0,8,0,0,0,0", udp_port=14541, mavsdk_port=50052)
    assert_sitl_endpoint(inst0.connect_address)
    assert_sitl_endpoint(inst1.connect_address)

    if not args.headless:
        apply_gui_environment()

    cluster = MultiPX4SitlCluster(instances=(inst0, inst1), headless=args.headless)
    att1: Optional[AttitudeWatcher] = None
    att2: Optional[AttitudeWatcher] = None
    drone1 = None
    drone2 = None
    monitor1_tasks = []

    try:
        print("[DUAL-HIL] Starting 2 PX4 SITL + gz_x500 drones in Gazebo Garden...", flush=True)
        cluster.start(primary_wait_s=9.0, secondary_wait_s=6.0)
        if not args.headless:
            set_gazebo_camera_pose(pos_enu=(-7.5, -4.5, 6.5), look_at_enu=(0.0, 4.0, args.hover_alt))

        drone1, drone2 = await asyncio.gather(
            connect_sitl_drone(inst0.connect_address, timeout_s=35.0, mavsdk_port=inst0.mavsdk_port),
            connect_sitl_drone(inst1.connect_address, timeout_s=35.0, mavsdk_port=inst1.mavsdk_port),
        )
        att1 = AttitudeWatcher(drone1)
        att2 = AttitudeWatcher(drone2)
        att1.start()
        att2.start()
        await asyncio.sleep(0.3)

        print(f"[DUAL-HIL] Quick takeoff of both drones to {args.hover_alt:.1f} m...", flush=True)
        await asyncio.gather(
            arm_and_hover_sitl(drone1, att1, hover_alt_m=args.hover_alt, quick=True),
            arm_and_hover_sitl(drone2, att2, hover_alt_m=args.hover_alt, quick=True),
        )
        if not args.headless:
            set_gazebo_camera_pose(pos_enu=(-7.5, -4.5, 6.5), look_at_enu=(0.0, 4.0, args.hover_alt))

        config, mount_id, control_rate_hz = load_sim_guidance_config(
            {"geofence_radius_m": float(args.geofence_m)} if args.geofence_m is not None else None
        )
        monitor1 = PX4FlightMonitor(drone1, config)
        monitor1_tasks = [asyncio.create_task(t) for t in monitor1.tasks()]
        deadline = asyncio.get_running_loop().time() + 5.0
        while not monitor1.ready(asyncio.get_running_loop().time() * 0.0 + __import__("time").monotonic()):
            if asyncio.get_running_loop().time() >= deadline:
                raise TimeoutError("Timed out waiting for Drone 1 PX4FlightMonitor telemetry")
            await drone1.offboard.set_velocity_body(VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0))
            await drone2.offboard.set_velocity_body(VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0))
            await asyncio.sleep(0.05)

        monitor1.start_position = monitor1.local_position
        guidance1 = GuidanceController(config)
        guidance1.enable()

        import time
        import numpy as np
        t0 = time.monotonic()
        last_log = -1.0
        last_dsp_t = -1.0
        period_s = 1.0 / control_rate_hz
        dsp_sensor = AcousticWavefieldDOASensor() if args.follow_mode == "acoustic-dsp" else None
        payload1 = None
        diag1 = {}

        print(
            f"[DUAL-HIL] Active! follow={args.follow_mode}, target={args.target_mode}, "
            f"duration={args.duration:.0f}s (no dashboard; watch Gazebo 3D view).",
            flush=True,
        )

        while True:
            now = time.monotonic()
            elapsed = now - t0
            if elapsed >= args.duration:
                break

            desk_payload, _ = primary_uma16.sample(now)
            desk_obs = AcousticObservation.from_mapping(desk_payload) if desk_payload is not None else None

            d1_ned = np.array([att1.north_m, att1.east_m, att1.down_m], dtype=float)
            d2_ned_in_d1 = np.array([att2.north_m + 8.0, att2.east_m, att2.down_m], dtype=float)
            horiz_sep = float(math.hypot(d2_ned_in_d1[0] - d1_ned[0], d2_ned_in_d1[1] - d1_ned[1]))

            if dsp_sensor is not None:
                if last_dsp_t < 0.0 or (elapsed - last_dsp_t) >= 0.2:
                    d1_rpy = (att1.roll_deg, att1.pitch_deg, att1.yaw_deg)
                    payload1, diag1 = dsp_sensor.sample(now, d1_ned, d1_rpy, d2_ned_in_d1, target_active=True)
                    last_dsp_t = elapsed
            else:
                payload1 = desk_payload

            obs1 = None
            if payload1 is not None:
                validate_observation_metadata(payload1, mount_id)
                obs1 = AcousticObservation.from_mapping(payload1)

            flight1 = monitor1.flight_state(now)
            if not flight1.offboard_active:
                cmd1 = guidance1.abort("offboard_lost_or_manual_override")
            elif not flight1.telemetry_fresh:
                cmd1 = guidance1.abort("flight_telemetry_stale")
            else:
                cmd1 = guidance1.update(now, obs1, flight1)
                cmd1 = apply_fixed_follow_distance(cmd1, horiz_sep, follow_distance_m=args.follow_distance)

            if cmd1.state != GuidanceState.ABORTED and flight1.offboard_active:
                await drone1.offboard.set_velocity_body(
                    VelocityBodyYawspeed(
                        cmd1.forward_m_s,
                        cmd1.right_m_s,
                        cmd1.down_m_s,
                        cmd1.yaw_rate_deg_s,
                    )
                )

            # Drone 2 target command
            if args.target_mode == "auto-target":
                cmd2 = compute_auto_target_command(elapsed, -att2.down_m, base_alt_m=args.hover_alt)
            elif args.target_mode == "dual-mic" and target_uma16 is not None:
                payload2, _ = target_uma16.sample(now)
                obs2 = AcousticObservation.from_mapping(payload2) if payload2 is not None else None
                cmd2 = compute_target_drone_command(
                    obs2,
                    now,
                    att2.north_m,
                    att2.east_m,
                    -att2.down_m,
                    att2.yaw_deg,
                    max_radius_m=args.target_radius_m,
                )
            else:  # shared-mic / real UMA-16 steering Drone 2
                cmd2 = compute_target_drone_command(
                    desk_obs,
                    now,
                    att2.north_m,
                    att2.east_m,
                    -att2.down_m,
                    att2.yaw_deg,
                    max_radius_m=args.target_radius_m,
                )

            await drone2.offboard.set_velocity_body(
                VelocityBodyYawspeed(
                    cmd2.forward_m_s,
                    cmd2.right_m_s,
                    cmd2.down_m_s,
                    cmd2.yaw_rate_deg_s,
                )
            )

            if last_log < 0.0 or (elapsed - last_log) >= 1.0:
                az_str = "None" if obs1 is None or obs1.azimuth_deg is None else f"{wrap_angle_deg(obs1.azimuth_deg):+.1f}°"
                el_str = "None" if obs1 is None or obs1.elevation_deg is None else f"{obs1.elevation_deg:+.1f}°"
                print(
                    f"[t={elapsed:5.1f}s] Mic(az={az_str}, el={el_str}) | "
                    f"D1={cmd1.state.value} (fwd={cmd1.forward_m_s:.2f}, yaw={cmd1.yaw_rate_deg_s:+.1f}, alt={-att1.down_m:.2f}m) | "
                    f"D2=(N={att2.north_m:+.1f}, E={att2.east_m:+.1f}, alt={-att2.down_m:.2f}m, fwd={cmd2.forward_m_s:.2f})",
                    flush=True,
                )
                last_log = elapsed

            if cmd1.state == GuidanceState.ABORTED:
                print(f"[DUAL-HIL] Drone 1 aborted: {cmd1.reason}", flush=True)
                break

            sleep_for = period_s - (time.monotonic() - now)
            if sleep_for > 0.0:
                await asyncio.sleep(sleep_for)
    finally:
        for t in monitor1_tasks:
            t.cancel()
        if monitor1_tasks:
            await asyncio.gather(*monitor1_tasks, return_exceptions=True)
        for att in (att1, att2):
            if att is not None:
                await att.stop()
        for drone in (drone1, drone2):
            if drone is not None:
                try:
                    await asyncio.wait_for(drone.offboard.stop(), timeout=2.0)
                except Exception:
                    pass
        cluster.stop()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--follow-mode",
        choices=("acoustic-dsp", "shared-mic"),
        default="acoustic-dsp",
        help="How Drone 1 follows: acoustic-dsp (16-ch UMA-16 SRP-PHAT chasing Drone 2) or shared-mic",
    )
    parser.add_argument(
        "--target-mode",
        choices=("shared-mic", "auto-target", "dual-mic"),
        default="shared-mic",
        help="How Drone 2 moves: shared-mic (steered by real UMA16v2), auto-target, or dual-mic",
    )
    parser.add_argument("--device", default=None, help="UMA-16 device index or name substring")
    parser.add_argument("--target-device", default=None, help="Second mic device for dual-mic mode")
    parser.add_argument("--with-classifier", action="store_true", help="Enable DroneClassifier gating")
    parser.add_argument("--low", type=float, default=None, help="Lower target band (Hz)")
    parser.add_argument("--high", type=float, default=None, help="Upper target band (Hz)")
    parser.add_argument("--threshold", type=float, default=None, help="Detection threshold (dBFS)")
    parser.add_argument("--hover-alt", type=float, default=3.0, help="Initial hover altitude (m)")
    parser.add_argument("--duration", type=float, default=120.0, help="Session duration (s)")
    parser.add_argument("--geofence-m", type=float, default=None, help="Override Drone 1 geofence radius (m)")
    parser.add_argument("--target-radius-m", type=float, default=15.0, help="Keep Drone 2 within radius (m)")
    parser.add_argument("--follow-distance", type=float, default=5.0, help="Fixed distance Drone 1 keeps from Drone 2 (m)")
    parser.add_argument("--headless", action="store_true", help="Disable Gazebo 3D GUI")
    parser.add_argument("--startup-timeout", type=float, default=12.0, help="UMA-16 preflight timeout (s)")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    primary_cfg = RealUMA16Config(
        source_mode="local",
        device=args.device,
        no_classifier=not args.with_classifier,
        low_hz=args.low,
        high_hz=args.high,
        threshold_dbfs=args.threshold,
        startup_timeout_s=args.startup_timeout,
    )
    primary_uma16 = RealUMA16Source(config=primary_cfg)
    target_uma16 = None
    if args.target_mode == "dual-mic":
        target_cfg = RealUMA16Config(
            source_mode="local",
            device=args.target_device,
            no_classifier=not args.with_classifier,
            low_hz=args.low,
            high_hz=args.high,
            threshold_dbfs=args.threshold,
            startup_timeout_s=args.startup_timeout,
        )
        target_uma16 = RealUMA16Source(config=target_cfg)

    try:
        print("[DUAL-HIL] Checking live UMA-16 telemetry...", flush=True)
        primary_uma16.start()
        primary_uma16.wait_ready()
        if target_uma16 is not None:
            target_uma16.start()
            target_uma16.wait_ready()
        print("[DUAL-HIL] Live UMA-16 telemetry verified.", flush=True)
        asyncio.run(_run_dual_drone(args, primary_uma16, target_uma16))
        return 0
    except KeyboardInterrupt:
        print("\n[DUAL-HIL] Interrupted by user, shutting down cleanly.", flush=True)
        return 130
    finally:
        primary_uma16.close()
        if target_uma16 is not None:
            target_uma16.close()


if __name__ == "__main__":
    raise SystemExit(main())
