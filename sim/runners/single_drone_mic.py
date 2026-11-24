#!/usr/bin/env python3
"""Single-drone PX4 SITL + Gazebo Garden simulation driven by YOLO/DyHiT vision with UMA-16 mic fallback."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
import math
import signal
import subprocess
import time
from typing import Dict, Optional, Tuple

from frequency_id.control.guidance import AcousticObservation, wrap_angle_deg
from sim.sensors.real_uma16 import RealUMA16Config, RealUMA16Source
from sim.harness.sitl_harness import (
    AttitudeWatcher,
    DEFAULT_FOLLOW_OFFSET_XYZ,
    PX4SitlProcess,
    apply_gui_environment,
    arm_and_hover_sitl,
    assert_sitl_endpoint,
    connect_sitl_drone,
    set_gazebo_follow_offset,
    set_gazebo_follow_target,
    spawn_gazebo_motion_reference_arena,
)


@dataclass(frozen=True)
class SingleDroneCommand:
    """World NED velocity + target yaw command (plus body projections for logging/testing)."""

    forward_m_s: float = 0.0
    right_m_s: float = 0.0
    down_m_s: float = 0.0
    yaw_rate_deg_s: float = 0.0
    north_m_s: float = 0.0
    east_m_s: float = 0.0
    target_yaw_deg: float = 0.0
    active: bool = False


class DirectionSmoother:
    """Circular/linear EMA smoother for azimuth and elevation with source/timeout reset."""

    def __init__(self, alpha: float = 0.4, reset_after_s: float = 0.6) -> None:
        self.alpha = float(alpha)
        self.reset_after_s = float(reset_after_s)
        self._source: Optional[str] = None
        self._last_t: Optional[float] = None
        self._az: Optional[float] = None
        self._el: Optional[float] = None

    def reset(self) -> None:
        self._source = None
        self._last_t = None
        self._az = None
        self._el = None

    def update(self, now: float, source: str, az_deg: float, el_deg: float) -> Tuple[float, float]:
        az = wrap_angle_deg(az_deg)
        el = float(el_deg)
        if (
            self._source != source
            or self._last_t is None
            or (now - self._last_t) > self.reset_after_s
            or self._az is None
            or self._el is None
        ):
            self._az = az
            self._el = el
        else:
            diff = wrap_angle_deg(az - self._az)
            self._az = wrap_angle_deg(self._az + self.alpha * diff)
            self._el = self._el + self.alpha * (el - self._el)
        self._source = source
        self._last_t = float(now)
        return self._az, self._el


def compute_world_direction_command(
    azimuth_deg: float,
    elevation_deg: float,
    north_m: float,
    east_m: float,
    alt_m: float,
    yaw_deg: float,
    speed: float = 2.0,
    max_radius_m: float = 25.0,
    min_alt_m: float = 2.0,
    max_alt_m: float = 20.0,
) -> SingleDroneCommand:
    """Map world-referenced (azimuth_deg, elevation_deg) to NED velocity and target yaw."""
    az = wrap_angle_deg(azimuth_deg)
    el = max(-75.0, min(75.0, float(elevation_deg)))
    az_rad = math.radians(az)
    el_rad = math.radians(el)
    cos_el = math.cos(el_rad)

    v_north = speed * cos_el * math.cos(az_rad)
    v_east = speed * cos_el * math.sin(az_rad)
    v_down = -speed * math.sin(el_rad)

    if alt_m <= min_alt_m and v_down > 0.0:
        v_down = 0.0
    if alt_m >= max_alt_m and v_down < 0.0:
        v_down = 0.0

    dist = math.hypot(north_m, east_m)
    if dist >= max_radius_m and (north_m * v_north + east_m * v_east) > 0.0:
        v_north = 0.0
        v_east = 0.0

    target_yaw = az if cos_el >= 0.15 else wrap_angle_deg(yaw_deg)
    heading_rad = math.radians(yaw_deg)
    forward = v_north * math.cos(heading_rad) + v_east * math.sin(heading_rad)
    right = -v_north * math.sin(heading_rad) + v_east * math.cos(heading_rad)

    return SingleDroneCommand(
        forward_m_s=forward,
        right_m_s=right,
        down_m_s=v_down,
        yaw_rate_deg_s=0.0,
        north_m_s=v_north,
        east_m_s=v_east,
        target_yaw_deg=target_yaw,
        active=True,
    )


def _is_gazebo_gui_running() -> bool:
    try:
        return subprocess.run(["pgrep", "-f", "[g]z sim -g"], capture_output=True).returncode == 0
    except Exception:
        return False


def select_single_drone_command(
    guidance_payload: Optional[Dict],
    fused_payload: Optional[Dict],
    camera_payload: Optional[Dict],
    now: float,
    last_fused_received_at: Optional[float],
    north_m: float,
    east_m: float,
    alt_m: float,
    yaw_deg: float,
    vision_timeout_s: float = 0.8,
    mic_timeout_s: float = 0.8,
    max_radius_m: float = 25.0,
    min_alt_m: float = 2.0,
    max_alt_m: float = 20.0,
    speed: float = 2.0,
    smoother: Optional[DirectionSmoother] = None,
) -> Tuple[SingleDroneCommand, str, Optional[float], Optional[float], str]:
    """Select vision-priority direction when camera sees/tracks a drone, else UMA-16 mic direction."""
    fused_fresh = (
        last_fused_received_at is not None
        and (now - float(last_fused_received_at)) <= vision_timeout_s
    )
    if fused_fresh and isinstance(fused_payload, dict) and fused_payload.get("source") == "VISION":
        raw_bearing = fused_payload.get("bearing_deg")
        raw_elevation = fused_payload.get("elevation_deg")
        if raw_bearing is not None and math.isfinite(float(raw_bearing)):
            az = wrap_angle_deg(float(raw_bearing))
            el = 0.0 if raw_elevation is None else float(raw_elevation)
            if math.isfinite(el):
                if smoother is not None:
                    az, el = smoother.update(now, "VISION", az, el)
                cmd = compute_world_direction_command(
                    azimuth_deg=az,
                    elevation_deg=el,
                    north_m=north_m,
                    east_m=east_m,
                    alt_m=alt_m,
                    yaw_deg=yaw_deg,
                    speed=speed,
                    max_radius_m=max_radius_m,
                    min_alt_m=min_alt_m,
                    max_alt_m=max_alt_m,
                )
                cam_status = str((camera_payload or {}).get("status") or "VISION")
                return cmd, "VISION", az, el, cam_status

    obs = AcousticObservation.from_mapping(guidance_payload) if guidance_payload is not None else None
    if (
        obs is not None
        and not obs.error
        and not obs.suppressed
        and math.isfinite(obs.captured_at)
        and (now - obs.captured_at) <= mic_timeout_s
        and obs.azimuth_deg is not None
        and math.isfinite(float(obs.azimuth_deg))
        and (obs.azimuth_stable or obs.azimuth_prominence >= 1.4)
    ):
        az = wrap_angle_deg(float(obs.azimuth_deg))
        el_raw = obs.elevation_deg
        has_el = el_raw is not None and math.isfinite(float(el_raw))
        el = float(el_raw) if has_el else 0.0
        if smoother is not None:
            az, el = smoother.update(now, "MIC", az, el)
        cmd = compute_world_direction_command(
            azimuth_deg=az,
            elevation_deg=el,
            north_m=north_m,
            east_m=east_m,
            alt_m=alt_m,
            yaw_deg=yaw_deg,
            speed=speed,
            max_radius_m=max_radius_m,
            min_alt_m=min_alt_m,
            max_alt_m=max_alt_m,
        )
        detail = "acoustic" if has_el else "acoustic-az-only"
        return cmd, "MIC", az, el, detail

    az = None if obs is None or obs.azimuth_deg is None else wrap_angle_deg(obs.azimuth_deg)
    el = None if obs is None or obs.elevation_deg is None else float(obs.elevation_deg)
    cam_status = str((camera_payload or {}).get("status") or "searching")
    if obs is not None and obs.suppressed:
        detail = f"cam={cam_status},mic=suppressed"
    elif obs is not None and obs.error:
        detail = f"cam={cam_status},mic={obs.error}"
    else:
        detail = cam_status
    return SingleDroneCommand(), "NONE", az, el, detail


async def _run_single_drone(args, uma16: RealUMA16Source):
    from mavsdk.offboard import PositionNedYaw, VelocityNedYaw

    assert_sitl_endpoint(args.connect)
    if not args.headless:
        apply_gui_environment()

    sitl = PX4SitlProcess(headless=args.headless)
    att: Optional[AttitudeWatcher] = None
    drone = None
    try:
        print("[MIC-DRONE] Starting PX4 SITL + Gazebo Garden (1x gz_x500)...", flush=True)
        sitl.start(startup_wait_s=9.0)

        drone = await connect_sitl_drone(address=args.connect, timeout_s=35.0)
        att = AttitudeWatcher(drone)
        att.start()
        await asyncio.sleep(0.3)

        if not args.headless:
            await asyncio.to_thread(spawn_gazebo_motion_reference_arena, "default", args.hover_alt)
            await asyncio.to_thread(set_gazebo_follow_target, "x500_0", DEFAULT_FOLLOW_OFFSET_XYZ)

        print(f"[MIC-DRONE] Quick takeoff to {args.hover_alt:.1f} m...", flush=True)
        await arm_and_hover_sitl(drone, att, hover_alt_m=args.hover_alt, quick=True)
        if not args.headless:
            await asyncio.to_thread(set_gazebo_follow_target, "x500_0", DEFAULT_FOLLOW_OFFSET_XYZ)

        t0 = time.monotonic()
        last_log = -1.0
        last_gz_check = 0.0
        gz_gui_seen = False
        offset_reasserted = False
        hold_pose: Optional[Tuple[float, float, float, float]] = (
            att.north_m,
            att.east_m,
            max(-20.0, min(-2.0, -args.hover_alt)),
            att.yaw_deg,
        )
        smoother = DirectionSmoother(alpha=0.4)
        period_s = 0.05  # 20 Hz

        mode_desc = "UMA-16 mic only" if args.no_vision else "YOLO+DyHiT vision priority -> UMA-16 mic fallback"
        print(
            f"[MIC-DRONE] Active ({mode_desc}, speed={args.speed:.1f}m/s) for {args.duration:.0f}s.",
            flush=True,
        )

        while True:
            now = time.monotonic()
            elapsed = now - t0
            if elapsed >= args.duration:
                break

            if sitl.proc is not None and sitl.proc.poll() is not None:
                print("[MIC-DRONE] PX4 SITL exited; stopping.", flush=True)
                break

            if not args.headless and (elapsed - last_gz_check) >= 2.0:
                last_gz_check = elapsed
                gz_running = await asyncio.to_thread(_is_gazebo_gui_running)
                if gz_running:
                    gz_gui_seen = True
                elif gz_gui_seen:
                    print(
                        "[MIC-DRONE] Gazebo GUI closed; stopping simulation and releasing camera/UMA-16.",
                        flush=True,
                    )
                    break

            if not args.headless and not offset_reasserted and elapsed >= 2.0:
                offset_reasserted = True
                asyncio.create_task(
                    asyncio.to_thread(set_gazebo_follow_offset, DEFAULT_FOLLOW_OFFSET_XYZ)
                )

            guidance_payload, fused_payload, camera_payload, diag = uma16.sample_fused()
            cmd, active_src, active_az, active_el, src_detail = select_single_drone_command(
                guidance_payload=guidance_payload,
                fused_payload=fused_payload,
                camera_payload=camera_payload,
                now=now,
                last_fused_received_at=diag.get("last_fused_received_at"),
                north_m=att.north_m,
                east_m=att.east_m,
                alt_m=-att.down_m,
                yaw_deg=att.yaw_deg,
                vision_timeout_s=args.vision_timeout,
                max_radius_m=args.radius_m,
                speed=args.speed,
                smoother=smoother,
            )

            if cmd.active:
                hold_pose = None
                await drone.offboard.set_velocity_ned(
                    VelocityNedYaw(
                        cmd.north_m_s,
                        cmd.east_m_s,
                        cmd.down_m_s,
                        cmd.target_yaw_deg,
                    )
                )
            else:
                if hold_pose is None:
                    hold_d = max(-20.0, min(-2.0, att.down_m))
                    hold_pose = (att.north_m, att.east_m, hold_d, att.yaw_deg)
                await drone.offboard.set_position_ned(PositionNedYaw(*hold_pose))

            if last_log < 0.0 or (elapsed - last_log) >= 1.0:
                az_str = "None" if active_az is None else f"{active_az:+.1f}°"
                el_str = "None" if active_el is None else f"{active_el:+.1f}°"
                print(
                    f"[t={elapsed:5.1f}s] Src={active_src}({src_detail}) dir(az={az_str}, el={el_str}) | "
                    f"Drone(N={att.north_m:+.2f}, E={att.east_m:+.2f}, alt={-att.down_m:.2f}m, yaw={att.yaw_deg:+.1f}°) | "
                    f"VelNED(N={cmd.north_m_s:+.2f}, E={cmd.east_m_s:+.2f}, D={cmd.down_m_s:+.2f})",
                    flush=True,
                )
                last_log = elapsed

            sleep_for = period_s - (time.monotonic() - now)
            if sleep_for > 0.0:
                await asyncio.sleep(sleep_for)
    finally:
        if att is not None:
            await att.stop()
        if drone is not None:
            try:
                await asyncio.wait_for(drone.offboard.stop(), timeout=2.0)
            except Exception:
                pass
        sitl.stop()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default=None, help="UMA-16 device index or name substring")
    parser.add_argument("--with-classifier", action="store_true", help="Enable DroneClassifier gating")
    parser.add_argument("--no-classifier", action="store_true", help="Disable DroneClassifier gating (default)")
    parser.add_argument("--low", type=float, default=300.0, help="Lower target band (Hz)")
    parser.add_argument("--high", type=float, default=3000.0, help="Upper target band (Hz)")
    parser.add_argument("--threshold", type=float, default=-55.0, help="Detection threshold (dBFS)")
    parser.add_argument("--connect", default="udp://:14540", help="PX4 SITL MAVSDK endpoint")
    parser.add_argument("--hover-alt", type=float, default=10.0, help="Initial hover altitude (m)")
    parser.add_argument("--speed", type=float, default=2.0, help="Flight speed (m/s)")
    parser.add_argument("--duration", type=float, default=120.0, help="Session duration (s)")
    parser.add_argument("--radius-m", type=float, default=25.0, help="Horizontal flight radius limit (m)")
    parser.add_argument("--headless", action="store_true", help="Disable Gazebo 3D GUI")
    parser.add_argument("--startup-timeout", type=float, default=12.0, help="UMA-16 preflight timeout (s)")
    parser.add_argument("--no-vision", action="store_true", help="Disable YOLO + tracker vision priority")
    parser.add_argument("--camera", default=None, help="Camera index or /dev/video device path")
    parser.add_argument("--camera-fov", type=float, default=None, help="Horizontal camera FOV override (deg)")
    parser.add_argument("--no-tracker", action="store_true", help="Disable DyHiT visual tracker (YOLO-only)")
    parser.add_argument("--vision-timeout", type=float, default=0.8, help="Vision freshness timeout before mic fallback (s)")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    signal.signal(signal.SIGTERM, lambda *_a: (_ for _ in ()).throw(KeyboardInterrupt()))
    cfg = RealUMA16Config(
        source_mode="local",
        device=args.device,
        no_classifier=not args.with_classifier,
        low_hz=args.low,
        high_hz=args.high,
        threshold_dbfs=args.threshold,
        startup_timeout_s=args.startup_timeout,
        enable_vision=not args.no_vision,
        camera=args.camera,
        camera_fov=args.camera_fov,
        no_tracker=args.no_tracker,
    )
    uma16 = RealUMA16Source(config=cfg)
    try:
        label = "UMA-16" if args.no_vision else "YOLO/DyHiT + UMA-16"
        print(f"[MIC-DRONE] Checking live {label} telemetry...", flush=True)
        uma16.start()
        uma16.wait_ready()
        print(f"[MIC-DRONE] Live {label} telemetry verified.", flush=True)
        asyncio.run(_run_single_drone(args, uma16))
        return 0
    except KeyboardInterrupt:
        print("\n[MIC-DRONE] Interrupted by user, shutting down cleanly.", flush=True)
        return 130
    finally:
        uma16.close()


if __name__ == "__main__":
    raise SystemExit(main())
