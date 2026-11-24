#!/usr/bin/env python3
"""Run PX4 SITL + Gazebo Garden driven by a live physical UMA-16 V2 array."""

from __future__ import annotations

import argparse
import asyncio
import csv
from pathlib import Path
from typing import List, Optional

from sim.sensors.real_uma16 import RealUMA16Config, RealUMA16Source
from sim.harness.sitl_harness import (
    AttitudeWatcher,
    PX4SitlProcess,
    REPO_ROOT,
    apply_gui_environment,
    assert_sitl_endpoint,
    connect_sitl_drone,
    run_sitl_guidance_scenario,
)
from sim.harness.visualizer import SimRecord


DEFAULT_PLOT_PATH = REPO_ROOT / "sim" / "reports" / "plots" / "real_uma16_hil.png"


def write_records_csv(records: List[SimRecord], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "t_s",
        "drone_n",
        "drone_e",
        "drone_alt",
        "drone_yaw_deg",
        "drone_vz_m_s",
        "meas_az_deg",
        "meas_el_deg",
        "az_prominence",
        "el_prominence",
        "state",
        "reason",
        "cmd_forward_m_s",
        "cmd_down_m_s",
        "cmd_yaw_rate_deg_s",
        "offboard_active",
        "telemetry_fresh",
        "geofence_ok",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow({k: getattr(r, k) for k in fieldnames})


async def _run_hil(args, uma16: RealUMA16Source):
    assert_sitl_endpoint(args.connect)
    if not args.headless:
        apply_gui_environment()

    sitl = PX4SitlProcess(headless=args.headless)
    att: Optional[AttitudeWatcher] = None
    drone = None
    try:
        print("[HIL] Starting PX4 SITL + Gazebo Garden (gz_x500)...", flush=True)
        sitl.start(startup_wait_s=12.0)
        drone = await connect_sitl_drone(address=args.connect, timeout_s=35.0)
        att = AttitudeWatcher(drone)
        att.start()
        await asyncio.sleep(0.5)

        overrides = {}
        if args.geofence_m is not None:
            overrides["geofence_radius_m"] = float(args.geofence_m)

        print(
            f"[HIL] Ready! Hovering at {args.hover_alt:.1f} m for {args.duration:.1f} s "
            f"(source={args.source}, classifier={'ON' if args.with_classifier else 'OFF'}).",
            flush=True,
        )
        result = await run_sitl_guidance_scenario(
            name="Real UMA-16 HIL | PX4 SITL + Gazebo",
            drone=drone,
            att=att,
            target_fn=None,
            guidance_overrides=overrides or None,
            hover_alt_m=args.hover_alt,
            duration_s=args.duration,
            gz_visual=None,
            live_viz=not args.headless,
            plot_path=Path(args.save_plot) if args.save_plot else None,
            stop_on_abort=True,
            observation_provider=uma16.sample,
        )
        if args.save_csv:
            write_records_csv(result.records, Path(args.save_csv))
            print(f"[HIL] Saved CSV log to {args.save_csv}", flush=True)
        if result.plot_path:
            print(f"[HIL] Saved plot to {result.plot_path}", flush=True)
        print(
            f"[HIL] Finished with {len(result.records)} samples; "
            f"final state={result.records[-1].state if result.records else 'NONE'}",
            flush=True,
        )
    finally:
        if att is not None:
            await att.stop()
        if drone is not None:
            try:
                await drone.offboard.stop()
            except Exception:
                pass
        sitl.stop()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        choices=("local", "stdin"),
        default="local",
        help="local: spawn main.py cli for PC-attached UMA-16; stdin: read piped @JSON stream",
    )
    parser.add_argument("--device", default=None, help="UMA-16 device index or name substring")
    parser.add_argument(
        "--with-classifier",
        action="store_true",
        help="enable DroneClassifier gating (disabled by default for bench acoustic HIL)",
    )
    parser.add_argument("--low", type=float, default=None, help="lower target frequency band (Hz)")
    parser.add_argument("--high", type=float, default=None, help="upper target frequency band (Hz)")
    parser.add_argument("--threshold", type=float, default=None, help="detection threshold (dBFS)")
    parser.add_argument("--average", type=int, default=None, help="averaged audio blocks")
    parser.add_argument("--suppress-bearing", type=float, default=None, help="suppressed bearing (deg)")
    parser.add_argument("--suppress-tolerance", type=float, default=None, help="suppression half-width (deg)")
    parser.add_argument("--connect", default="udp://:14540", help="PX4 SITL MAVSDK endpoint")
    parser.add_argument("--hover-alt", type=float, default=5.0, help="initial SITL hover altitude (m)")
    parser.add_argument("--duration", type=float, default=120.0, help="HIL session duration (s)")
    parser.add_argument("--geofence-m", type=float, default=None, help="override geofence radius (m)")
    parser.add_argument("--headless", action="store_true", help="disable Gazebo GUI and live 3D window")
    parser.add_argument("--save-plot", default=str(DEFAULT_PLOT_PATH), help="output PNG plot path")
    parser.add_argument("--save-csv", default=None, help="optional CSV telemetry output path")
    parser.add_argument("--startup-timeout", type=float, default=12.0, help="UMA-16 stream preflight timeout (s)")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    cfg = RealUMA16Config(
        source_mode=args.source,
        device=args.device,
        no_classifier=not args.with_classifier,
        low_hz=args.low,
        high_hz=args.high,
        threshold_dbfs=args.threshold,
        average_blocks=args.average,
        suppress_bearing_deg=args.suppress_bearing,
        suppress_tolerance_deg=args.suppress_tolerance,
        startup_timeout_s=args.startup_timeout,
    )
    uma16 = RealUMA16Source(config=cfg)
    try:
        print(f"[HIL] Waiting for live UMA-16 front-only telemetry ({args.source})...", flush=True)
        uma16.start()
        uma16.wait_ready()
        print("[HIL] Live UMA-16 telemetry verified.", flush=True)
        asyncio.run(_run_hil(args, uma16))
        return 0
    except KeyboardInterrupt:
        print("\n[HIL] Interrupted by user, shutting down cleanly.", flush=True)
        return 130
    finally:
        uma16.close()


if __name__ == "__main__":
    raise SystemExit(main())
