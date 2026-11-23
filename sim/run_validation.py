#!/usr/bin/env python3
"""Full PX4 SITL + Gazebo + MAVSDK-Python validation suite for acoustic guidance."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from typing import Dict, List, Optional

import numpy as np

from guidance import GuidanceState, wrap_angle_deg
from sim.gazebo_target import (
    GazeboTargetVisual,
    TargetSample,
    dropout_window_target,
    moving_target,
    static_target,
    step_change_target,
)
from sim.sitl_harness import (
    AttitudeWatcher,
    PX4SitlProcess,
    apply_gui_environment,
    REPO_ROOT,
    ScenarioResult,
    arm_and_hover_sitl,
    connect_sitl_drone,
    load_sim_guidance_config,
    run_sitl_guidance_scenario,
    write_sim_rig_config,
)
from sim.virtual_uma16 import AcousticSensorConfig, VirtualUMA16Sensor
from sim.visualizer import SimRecord


REPORTS_DIR = REPO_ROOT / "sim" / "reports"
PLOTS_DIR = REPORTS_DIR / "plots"


def _metrics_from_records(records: List[SimRecord]) -> Dict:
    if not records:
        return {}
    states = [r.state for r in records]
    unique_states = []
    for s in states:
        if not unique_states or unique_states[-1] != s:
            unique_states.append(s)
    az_errs = [
        abs(wrap_angle_deg(r.meas_az_deg) - r.ideal_az_deg)
        for r in records
        if r.meas_az_deg is not None
    ]
    el_errs = [
        abs(r.meas_el_deg - r.ideal_el_deg)
        for r in records
        if r.meas_el_deg is not None
    ]
    start = records[0]
    end = records[-1]
    start_dist = math.hypot(start.target_n - start.drone_n, start.target_e - start.drone_e)
    end_dist = math.hypot(end.target_n - end.drone_n, end.target_e - end.drone_e)
    return {
        "duration_s": round(end.t_s, 2),
        "samples": len(records),
        "state_sequence": unique_states,
        "final_state": end.state,
        "final_reason": end.reason,
        "start_drone_alt_m": round(start.drone_alt, 2),
        "end_drone_alt_m": round(end.drone_alt, 2),
        "end_target_alt_m": round(end.target_alt, 2),
        "remaining_alt_diff_m": round(end.target_alt - end.drone_alt, 2),
        "end_yaw_deg": round(end.drone_yaw_deg, 1),
        "end_meas_az_deg": None if end.meas_az_deg is None else round(wrap_angle_deg(end.meas_az_deg), 1),
        "end_meas_el_deg": None if end.meas_el_deg is None else round(end.meas_el_deg, 1),
        "start_horiz_dist_m": round(start_dist, 2),
        "end_horiz_dist_m": round(end_dist, 2),
        "horiz_progress_m": round(start_dist - end_dist, 2),
        "max_forward_cmd_m_s": round(max(r.cmd_forward_m_s for r in records), 2),
        "min_down_cmd_m_s": round(min(r.cmd_down_m_s for r in records), 2),
        "max_down_cmd_m_s": round(max(r.cmd_down_m_s for r in records), 2),
        "mean_az_err_deg": round(float(np.mean(az_errs)), 2) if az_errs else None,
        "mean_el_err_deg": round(float(np.mean(el_errs)), 2) if el_errs else None,
    }


async def run_t0_subprocess_px4_controller(drone, att: AttitudeWatcher, gz: GazeboTargetVisual) -> ScenarioResult:
    """Test 0: Spawn the real px4_controller.py subprocess fed @JSON: telemetry from VirtualUMA16Sensor."""
    from pymavlink import mavutil

    await arm_and_hover_sitl(drone, att, hover_alt_m=5.0)
    try:
        await drone.offboard.stop()
    except Exception:
        pass
    await att.stop()
    await asyncio.sleep(0.3)

    # Release UDP 14540 / gRPC 50051 held by harness mavsdk_server so px4_controller.py owns default udp://:14540
    subprocess.run(["pkill", "-9", "-f", "[m]avsdk_server"], capture_output=True)
    await asyncio.sleep(0.5)

    # Read PX4 GCS telemetry port 14550 solely to synthesize virtual 3D acoustic waves
    mav = mavutil.mavlink_connection("udpin:0.0.0.0:14550")
    mav.wait_heartbeat(timeout=5.0)

    sim_cfg_path = write_sim_rig_config(REPORTS_DIR / "sim_rig_config.json")
    proc = subprocess.Popen(
        [
            sys.executable,
            str(REPO_ROOT / "px4_controller.py"),
            "--authorize-offboard",
            "--enable-motion",
            "--config",
            str(sim_cfg_path),
            "--connect",
            "udp://:14540",
        ],
        cwd=str(REPO_ROOT),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    sensor = VirtualUMA16Sensor(sensor_config=AcousticSensorConfig(snr_db=22.0))
    target = TargetSample(north_m=8.0, east_m=5.0, altitude_m=7.5, active=True)
    if not gz.spawned:
        gz.spawn(target)
    else:
        gz.update_pose(target)

    pose = {"n": 0.0, "e": 0.0, "d": -5.0, "roll": 0.0, "pitch": 0.0, "yaw": 0.0, "custom_mode": 0}
    def drain_mav():
        while True:
            msg = mav.recv_match(blocking=False)
            if msg is None:
                break
            mtype = msg.get_type()
            if mtype == "LOCAL_POSITION_NED":
                pose["n"], pose["e"], pose["d"] = float(msg.x), float(msg.y), float(msg.z)
            elif mtype == "ATTITUDE":
                pose["roll"] = math.degrees(float(msg.roll))
                pose["pitch"] = math.degrees(float(msg.pitch))
                pose["yaw"] = math.degrees(float(msg.yaw))
            elif mtype == "HEARTBEAT" and msg.get_srcSystem() == 1:
                pose["custom_mode"] = int(msg.custom_mode)

    for _ in range(10):
        drain_mav()
        await asyncio.sleep(0.05)

    start_n, start_e, start_alt, start_yaw = pose["n"], pose["e"], -pose["d"], pose["yaw"]
    t0 = time.monotonic()
    try:
        while time.monotonic() - t0 < 34.0:
            if proc.poll() is not None:
                break
            drain_mav()
            now = time.monotonic()
            payload, _ = sensor.sample(
                now=now,
                drone_ned=np.array([pose["n"], pose["e"], pose["d"]]),
                drone_rpy_deg=(pose["roll"], pose["pitch"], pose["yaw"]),
                target_ned=target.ned,
                target_active=True,
            )
            line = "@JSON:" + json.dumps({"type": "telemetry", "guidance": payload}, separators=(",", ":"))
            proc.stdin.write(line + "\n")
            proc.stdin.flush()
            await asyncio.sleep(0.2)

        try:
            proc.stdin.close()
        except Exception:
            pass
        rc = proc.wait(timeout=10.0)
        stderr_txt = proc.stderr.read()
    finally:
        if proc.poll() is None:
            proc.kill()

    # Wait up to 1.5s for PX4's 1 Hz HEARTBEAT to reflect exit from OFFBOARD
    hb_deadline = time.monotonic() + 1.5
    offboard_released = False
    while time.monotonic() < hb_deadline:
        drain_mav()
        if ((pose["custom_mode"] >> 16) & 0xFF) != 6:
            offboard_released = True
            break
        await asyncio.sleep(0.1)
    mav.close()

    end_alt = -pose["d"]
    end_yaw = pose["yaw"]
    start_dist = math.hypot(target.north_m - start_n, target.east_m - start_e)
    end_dist = math.hypot(target.north_m - pose["n"], target.east_m - pose["e"])
    saw_states = all(s in stderr_txt for s in ("ALIGN_YAW", "ALIGN_VERTICAL", "FOLLOW"))
    passed = (
        rc == 0
        and saw_states
        and offboard_released
        and end_yaw > start_yaw + 10.0
        and end_alt > start_alt + 0.8
        and end_dist < start_dist - 0.5
    )
    summary = (
        f"Real px4_controller.py subprocess exit={rc}; yaw {start_yaw:.1f}°->{end_yaw:.1f}°, "
        f"alt {start_alt:.2f}m->{end_alt:.2f}m, dist {start_dist:.2f}m->{end_dist:.2f}m, "
        f"states_seen={saw_states}, offboard_released={offboard_released}"
    )
    return ScenarioResult(
        name="T0_subprocess_px4_controller",
        passed=passed,
        summary=summary,
        records=[],
        metrics={
            "exit_code": rc,
            "start_yaw_deg": round(start_yaw, 1),
            "end_yaw_deg": round(end_yaw, 1),
            "start_alt_m": round(start_alt, 2),
            "end_alt_m": round(end_alt, 2),
            "horiz_progress_m": round(start_dist - end_dist, 2),
            "offboard_released": offboard_released,
            "stderr_tail": stderr_txt.strip().splitlines()[-12:],
        },
    )


async def run_all_scenarios(
    headless_gz: bool = True,
    live_viz: bool = False,
    only_filter: Optional[str] = None,
) -> List[ScenarioResult]:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    sitl = PX4SitlProcess(headless=headless_gz)
    sitl.start(startup_wait_s=12.0)
    results: List[ScenarioResult] = []
    gz = GazeboTargetVisual()

    try:
        drone = await connect_sitl_drone("udpin://0.0.0.0:14540", timeout_s=30.0)
        att = AttitudeWatcher(drone)
        att.start()
        await asyncio.sleep(0.5)

        def wanted(name: str) -> bool:
            return not only_filter or only_filter.lower() in name.lower()

        # --- B1: Full 7-stage behavioral loop ---
        if wanted("B1_full_state_sequence"):
            b1_events = {"shift_at": None, "dropout_at": None}
            initial_target = TargetSample(8.0, 5.5, 6.5, active=True)
            shifted_target = TargetSample(9.5, 1.2, 6.2, active=True)

            def b1_target(t: float) -> TargetSample:
                shift_at = b1_events["shift_at"]
                if shift_at is None or t < shift_at:
                    return initial_target
                dropout_at = b1_events["dropout_at"]
                active = dropout_at is None or not dropout_at <= t < dropout_at + 3.5
                return TargetSample(
                    shifted_target.north_m,
                    shifted_target.east_m,
                    shifted_target.altitude_m,
                    active=active,
                )

            def b1_command_hook(t: float, command) -> None:
                if command.state != GuidanceState.FOLLOW:
                    return
                if b1_events["shift_at"] is None:
                    b1_events["shift_at"] = t + 2.0
                elif (
                    b1_events["dropout_at"] is None
                    and t >= b1_events["shift_at"] + 10.0
                ):
                    b1_events["dropout_at"] = t

            res = await run_sitl_guidance_scenario(
                name="B1_full_state_sequence",
                drone=drone,
                att=att,
                target_fn=b1_target,
                command_hook=b1_command_hook,
                hover_alt_m=5.0,
                duration_s=65.0,
                gz_visual=gz,
                live_viz=live_viz,
                plot_path=PLOTS_DIR / "B1_full_state_sequence.png",
            )
            m = _metrics_from_records(res.records)
            seq = m["state_sequence"]
            yaw_rot_ok = any(r.cmd_yaw_rate_deg_s > 5.0 for r in res.records if r.state == "ALIGN_YAW")
            climb_ok = any(r.cmd_down_m_s < -0.1 for r in res.records if r.state == "ALIGN_VERTICAL")
            follow_ok = "FOLLOW" in seq and m["horiz_progress_m"] > 0.5
            realign_ok = any(r.reason in ("yaw_alignment_lost", "elevation_alignment_lost") for r in res.records)
            hold_ok = any(
                r.state == "TARGET_LOST_HOLD" and r.cmd_forward_m_s == 0.0 and r.cmd_down_m_s == 0.0
                for r in res.records
            )
            res.passed = all(s in seq for s in ("HOVER_SEARCH", "ALIGN_YAW", "ALIGN_VERTICAL", "FOLLOW", "TARGET_LOST_HOLD")) and yaw_rot_ok and climb_ok and follow_ok and realign_ok and hold_ok
            res.metrics = {
                **m,
                "shift_at_s": b1_events["shift_at"],
                "dropout_at_s": b1_events["dropout_at"],
            }
            res.summary = (
                f"Stages={seq}; yaw_rot_ok={yaw_rot_ok}, climb_ok={climb_ok}, "
                f"realign_ok={realign_ok}, hold_ok={hold_ok}, horiz_progress={m['horiz_progress_m']}m, "
                f"shift_at={b1_events['shift_at']}, dropout_at={b1_events['dropout_at']}"
            )
            results.append(res)
            print(f"[{'PASS' if res.passed else 'FAIL'}] {res.name}: {res.summary}", flush=True)

        # --- B2: Negative elevation (descent sign verification) ---
        if wanted("B2_descend_sign_check"):
            res = await run_sitl_guidance_scenario(
                name="B2_descend_sign_check",
                drone=drone,
                att=att,
                target_fn=static_target(7.0, 0.0, 5.5),
                hover_alt_m=8.0,
                duration_s=30.0,
                gz_visual=gz,
                live_viz=live_viz,
                plot_path=PLOTS_DIR / "B2_descend_sign_check.png",
            )
            m = _metrics_from_records(res.records)
            res.passed = (
                m["max_down_cmd_m_s"] > 0.1
                and m["end_drone_alt_m"] < m["start_drone_alt_m"] - 1.0
                and abs(m["remaining_alt_diff_m"]) < 0.8
            )
            res.metrics = m
            res.summary = (
                f"Alt {m['start_drone_alt_m']}m -> {m['end_drone_alt_m']}m (target {m['end_target_alt_m']}m), "
                f"max_down_cmd=+{m['max_down_cmd_m_s']}m/s, rem_diff={m['remaining_alt_diff_m']:+.2f}m"
            )
            results.append(res)
            print(f"[{'PASS' if res.passed else 'FAIL'}] {res.name}: {res.summary}", flush=True)

        # --- Critical Experiments (Requirement 4) ---
        if wanted("E1_close_climb"):
            res = await run_sitl_guidance_scenario(
                name="E1_close_climb",
                drone=drone,
                att=att,
                target_fn=static_target(6.0, 2.0, 7.0),
                hover_alt_m=5.0,
                duration_s=28.0,
                gz_visual=gz,
                live_viz=live_viz,
                plot_path=PLOTS_DIR / "E1_close_climb.png",
            )
            m = _metrics_from_records(res.records)
            res.passed = "FOLLOW" in m["state_sequence"] and abs(m["remaining_alt_diff_m"]) < 0.7
            res.metrics = m
            res.summary = f"Alt {m['start_drone_alt_m']}->{m['end_drone_alt_m']}m (tgt {m['end_target_alt_m']}m), rem={m['remaining_alt_diff_m']:+.2f}m, el={m['end_meas_el_deg']}°, progress={m['horiz_progress_m']}m"
            results.append(res)
            print(f"[{'PASS' if res.passed else 'FAIL'}] {res.name}: {res.summary}", flush=True)

        if wanted("E2_large_climb_travel_cap"):
            res = await run_sitl_guidance_scenario(
                name="E2_large_climb_travel_cap",
                drone=drone,
                att=att,
                target_fn=static_target(8.0, 0.0, 9.5),  # +4.5m > max_vertical_travel_m (3.0m)
                hover_alt_m=5.0,
                duration_s=30.0,
                gz_visual=gz,
                live_viz=live_viz,
                plot_path=PLOTS_DIR / "E2_large_climb_travel_cap.png",
            )
            m = _metrics_from_records(res.records)
            climb_recs = [r for r in res.records if r.state == "ALIGN_VERTICAL" and r.cmd_down_m_s < -0.01]
            capped_alt = climb_recs[-1].drone_alt if climb_recs else m["end_drone_alt_m"]
            commanded_climb = capped_alt - m["start_drone_alt_m"]
            res.passed = (2.6 <= commanded_climb <= 3.3) and res.records[-1].cmd_down_m_s == 0.0 and (m["remaining_alt_diff_m"] > 0.3)
            res.metrics = {**m, "commanded_climb_before_cap_m": round(commanded_climb, 2)}
            res.summary = (
                f"Cmd climb stopped at +{commanded_climb:.2f}m (limit=3.0m), end_alt={m['end_drone_alt_m']}m vs tgt={m['end_target_alt_m']}m, "
                f"rem_diff={m['remaining_alt_diff_m']:+.2f}m, end_cmd_down={res.records[-1].cmd_down_m_s}"
            )
            results.append(res)
            print(f"[{'PASS' if res.passed else 'FAIL'}] {res.name}: {res.summary}", flush=True)

        if wanted("E3_long_range_deadband"):
            res = await run_sitl_guidance_scenario(
                name="E3_long_range_deadband",
                drone=drone,
                att=att,
                target_fn=static_target(12.0, 0.0, 5.95),  # +0.95m at 12m -> initial el=+4.5°
                hover_alt_m=5.0,
                duration_s=30.0,
                gz_visual=gz,
                live_viz=live_viz,
                plot_path=PLOTS_DIR / "E3_long_range_deadband.png",
            )
            m = _metrics_from_records(res.records)
            first_follow = next((r for r in res.records if r.state == "FOLLOW"), None)
            re_climbed = any(r.reason == "elevation_alignment_lost" for r in res.records)
            res.passed = first_follow is not None and m["horiz_progress_m"] > 2.0
            res.metrics = {
                **m,
                "first_follow_alt_diff_m": None if first_follow is None else round(first_follow.target_alt - first_follow.drone_alt, 2),
                "re_climbed_as_range_closed": re_climbed,
            }
            res.summary = (
                f"At first FOLLOW: rem_diff={res.metrics['first_follow_alt_diff_m']}m; "
                f"end_diff={m['remaining_alt_diff_m']:+.2f}m, re_climbed={re_climbed}, progress={m['horiz_progress_m']}m"
            )
            results.append(res)
            print(f"[{'PASS' if res.passed else 'FAIL'}] {res.name}: {res.summary}", flush=True)

        if wanted("E4_moving_target_3d"):
            res = await run_sitl_guidance_scenario(
                name="E4_moving_target_3d",
                drone=drone,
                att=att,
                target_fn=moving_target(7.5, 1.5, 5.8, vel_north=0.1, vel_east=0.20, vel_alt=0.04, move_after_s=5.0),
                hover_alt_m=5.0,
                duration_s=32.0,
                gz_visual=gz,
                live_viz=live_viz,
                plot_path=PLOTS_DIR / "E4_moving_target_3d.png",
            )
            m = _metrics_from_records(res.records)
            res.passed = m["end_drone_alt_m"] > m["start_drone_alt_m"] + 0.6 and m["end_yaw_deg"] > 10.0 and "FOLLOW" in m["state_sequence"]
            res.metrics = m
            res.summary = f"Tracked moving target: alt {m['start_drone_alt_m']}->{m['end_drone_alt_m']}m (tgt {m['end_target_alt_m']}m), yaw={m['end_yaw_deg']}°"
            results.append(res)
            print(f"[{'PASS' if res.passed else 'FAIL'}] {res.name}: {res.summary}", flush=True)

        if wanted("E5_noisy_delayed_propeller"):
            res = await run_sitl_guidance_scenario(
                name="E5_noisy_delayed_propeller",
                drone=drone,
                att=att,
                target_fn=static_target(7.5, 3.0, 6.8),
                sensor_config=AcousticSensorConfig(
                    snr_db=8.0,
                    delay_s=0.18,
                    unstable_elevation_jitter_deg=2.5,
                    propeller_noise_db=3.0,
                    propeller_freq_hz=1180.0,
                    propeller_bearing_deg=89.0,
                    suppress_propeller_bearing=True,
                    suppress_tolerance_deg=20.0,
                ),
                hover_alt_m=5.0,
                duration_s=34.0,
                gz_visual=gz,
                live_viz=live_viz,
                plot_path=PLOTS_DIR / "E5_noisy_delayed_propeller.png",
            )
            m = _metrics_from_records(res.records)
            res.passed = "FOLLOW" in m["state_sequence"] and m["horiz_progress_m"] > 1.0 and abs(m["remaining_alt_diff_m"]) < 1.0
            res.metrics = m
            res.summary = f"Noisy+propeller(suppressed): states={m['state_sequence']}, rem_diff={m['remaining_alt_diff_m']:+.2f}m, mean_az_err={m['mean_az_err_deg']}°, mean_el_err={m['mean_el_err_deg']}°"
            results.append(res)
            print(f"[{'PASS' if res.passed else 'FAIL'}] {res.name}: {res.summary}", flush=True)

        # --- Safety & Failure Tests (Requirement 5) ---
        if wanted("S1_loss_during_climb"):
            seen_climb = {"hit": False, "off_t": None}

            def s1_hook(t, _i, sensor, _mon, _dr):
                # Trigger dropout right after climb begins
                pass

            def s1_target(t: float) -> TargetSample:
                # Dropout from t=5.0 to t=9.0 while ALIGN_VERTICAL is climbing
                active = not (4.5 <= t < 9.5)
                return TargetSample(8.0, 0.0, 7.5, active=active)

            res = await run_sitl_guidance_scenario(
                name="S1_loss_during_climb",
                drone=drone,
                att=att,
                target_fn=s1_target,
                hover_alt_m=5.0,
                duration_s=13.0,
                gz_visual=gz,
                plot_path=PLOTS_DIR / "S1_loss_during_climb.png",
            )
            m = _metrics_from_records(res.records)
            drop_recs = [r for r in res.records if 5.5 <= r.t_s <= 9.2]
            entered_climb = any(r.state == "ALIGN_VERTICAL" for r in res.records if r.t_s < 5.0)
            held_zero = bool(drop_recs) and any(r.state == "TARGET_LOST_HOLD" and r.cmd_down_m_s == 0.0 for r in drop_recs)
            res.passed = entered_climb and held_zero
            res.metrics = m
            res.summary = f"entered_climb={entered_climb}, held_zero_on_loss={held_zero}, states={m['state_sequence']}"
            results.append(res)
            print(f"[{'PASS' if res.passed else 'FAIL'}] {res.name}: {res.summary}", flush=True)

        if wanted("S2_loss_during_follow"):
            def s2_target(t: float) -> TargetSample:
                return TargetSample(9.0, 0.0, 5.0, active=(t < 9.0))

            res = await run_sitl_guidance_scenario(
                name="S2_loss_during_follow",
                drone=drone,
                att=att,
                target_fn=s2_target,
                hover_alt_m=5.0,
                duration_s=14.0,
                gz_visual=gz,
                plot_path=PLOTS_DIR / "S2_loss_during_follow.png",
            )
            m = _metrics_from_records(res.records)
            res.passed = "FOLLOW" in m["state_sequence"] and res.records[-1].state == "TARGET_LOST_HOLD" and res.records[-1].cmd_forward_m_s == 0.0
            res.metrics = m
            res.summary = f"states={m['state_sequence']}, final={m['final_state']} ({m['final_reason']}), cmd_fwd={res.records[-1].cmd_forward_m_s}"
            results.append(res)
            print(f"[{'PASS' if res.passed else 'FAIL'}] {res.name}: {res.summary}", flush=True)

        if wanted("S3_stale_and_repeated_obs"):
            def s3_hook(t, _i, sensor, _mon, _dr):
                if t >= 7.0:
                    sensor.sensor.repeat_last_observation = True

            res = await run_sitl_guidance_scenario(
                name="S3_stale_and_repeated_obs",
                drone=drone,
                att=att,
                target_fn=static_target(9.0, 0.0, 5.0),
                hover_alt_m=5.0,
                duration_s=12.0,
                gz_visual=gz,
                fault_hook=s3_hook,
                plot_path=PLOTS_DIR / "S3_stale_and_repeated_obs.png",
            )
            m = _metrics_from_records(res.records)
            res.passed = "FOLLOW" in m["state_sequence"] and m["final_state"] == "TARGET_LOST_HOLD" and m["final_reason"] == "acoustic_observation_timeout"
            res.metrics = m
            res.summary = f"states={m['state_sequence']}, final={m['final_state']} ({m['final_reason']})"
            results.append(res)
            print(f"[{'PASS' if res.passed else 'FAIL'}] {res.name}: {res.summary}", flush=True)

        if wanted("S4_unreliable_elevation"):
            def s4_hook(t, _i, sensor, _mon, _dr):
                if t >= 7.0:
                    sensor.sensor.force_low_elev_prominence = True

            res = await run_sitl_guidance_scenario(
                name="S4_unreliable_elevation",
                drone=drone,
                att=att,
                target_fn=static_target(9.0, 0.0, 5.0),
                hover_alt_m=5.0,
                duration_s=11.0,
                gz_visual=gz,
                fault_hook=s4_hook,
                plot_path=PLOTS_DIR / "S4_unreliable_elevation.png",
            )
            m = _metrics_from_records(res.records)
            saw_invalid = any(r.reason == "acoustic_observation_invalid" for r in res.records if r.t_s >= 7.0)
            res.passed = (
                "FOLLOW" in m["state_sequence"]
                and saw_invalid
                and m["final_state"] == "TARGET_LOST_HOLD"
                and res.records[-1].cmd_forward_m_s == 0.0
            )
            res.metrics = m
            res.summary = f"states={m['state_sequence']}, saw_invalid={saw_invalid}, final={m['final_state']} ({m['final_reason']})"
            results.append(res)
            print(f"[{'PASS' if res.passed else 'FAIL'}] {res.name}: {res.summary}", flush=True)

        if wanted("S5_telemetry_stale_abort"):
            def s5_hook(t, _i, _sensor, mon, _dr):
                if t >= 7.5:
                    for k in list(mon.received_at):
                        mon.received_at[k] -= 6.0

            res = await run_sitl_guidance_scenario(
                name="S5_telemetry_stale_abort",
                drone=drone,
                att=att,
                target_fn=static_target(9.0, 0.0, 5.0),
                hover_alt_m=5.0,
                duration_s=12.0,
                gz_visual=gz,
                fault_hook=s5_hook,
                plot_path=PLOTS_DIR / "S5_telemetry_stale_abort.png",
            )
            m = _metrics_from_records(res.records)
            res.passed = m["final_state"] == "ABORTED" and m["final_reason"] == "flight_telemetry_stale"
            res.metrics = m
            res.summary = f"final={m['final_state']} ({m['final_reason']})"
            results.append(res)
            print(f"[{'PASS' if res.passed else 'FAIL'}] {res.name}: {res.summary}", flush=True)

        if wanted("S6_altitude_limits"):
            # Climb above tight max_altitude_m=6.0 from hover_alt=5.0 -> triggers altitude_envelope_violation
            res = await run_sitl_guidance_scenario(
                name="S6_altitude_limits",
                drone=drone,
                att=att,
                target_fn=static_target(8.0, 0.0, 8.5),
                guidance_overrides={"max_altitude_m": 6.0},
                hover_alt_m=5.0,
                duration_s=16.0,
                gz_visual=gz,
                plot_path=PLOTS_DIR / "S6_altitude_limits.png",
            )
            m = _metrics_from_records(res.records)
            # In SITL, _vertical_speed stops down_m_s at >=6.0, OR overshoot triggers altitude_envelope_violation
            clamped_or_aborted = (
                (m["final_state"] == "ABORTED" and m["final_reason"] == "altitude_envelope_violation")
                or (m["end_drone_alt_m"] <= 6.2 and abs(res.records[-1].cmd_down_m_s) < 1e-6)
            )
            res.passed = clamped_or_aborted
            res.metrics = m
            res.summary = f"final={m['final_state']} ({m['final_reason']}), end_alt={m['end_drone_alt_m']}m (limit=6.0m), cmd_down={res.records[-1].cmd_down_m_s}"
            results.append(res)
            print(f"[{'PASS' if res.passed else 'FAIL'}] {res.name}: {res.summary}", flush=True)

        if wanted("S7_geofence_violation"):
            res = await run_sitl_guidance_scenario(
                name="S7_geofence_violation",
                drone=drone,
                att=att,
                target_fn=static_target(12.0, 0.0, 5.0),
                guidance_overrides={"geofence_radius_m": 2.5},
                hover_alt_m=5.0,
                duration_s=20.0,
                gz_visual=gz,
                plot_path=PLOTS_DIR / "S7_geofence_violation.png",
            )
            m = _metrics_from_records(res.records)
            res.passed = m["final_state"] == "ABORTED" and m["final_reason"] == "geofence_violation"
            res.metrics = m
            res.summary = f"final={m['final_state']} ({m['final_reason']}), progress={m['horiz_progress_m']}m"
            results.append(res)
            print(f"[{'PASS' if res.passed else 'FAIL'}] {res.name}: {res.summary}", flush=True)

        if wanted("S8_manual_override"):
            override_sent = {"done": False}

            async def s8_hook(t, _i, _sensor, _mon, dr):
                if t >= 7.5 and not override_sent["done"]:
                    override_sent["done"] = True
                    await dr.action.hold()

            res = await run_sitl_guidance_scenario(
                name="S8_manual_override",
                drone=drone,
                att=att,
                target_fn=static_target(9.0, 0.0, 5.0),
                hover_alt_m=5.0,
                duration_s=15.0,
                gz_visual=gz,
                fault_hook=s8_hook,
                plot_path=PLOTS_DIR / "S8_manual_override.png",
            )
            m = _metrics_from_records(res.records)
            res.passed = m["final_state"] == "ABORTED" and m["final_reason"] == "offboard_lost_or_manual_override"
            res.metrics = m
            res.summary = f"final={m['final_state']} ({m['final_reason']})"
            results.append(res)
            print(f"[{'PASS' if res.passed else 'FAIL'}] {res.name}: {res.summary}", flush=True)

        if wanted("S9_offboard_connection_loss"):
            stopped = {"done": False}

            async def s9_hook(t, _i, _sensor, _mon, dr):
                if t >= 7.5 and not stopped["done"]:
                    stopped["done"] = True
                    await dr.offboard.stop()

            res = await run_sitl_guidance_scenario(
                name="S9_offboard_connection_loss",
                drone=drone,
                att=att,
                target_fn=static_target(9.0, 0.0, 5.0),
                hover_alt_m=5.0,
                duration_s=15.0,
                gz_visual=gz,
                fault_hook=s9_hook,
                plot_path=PLOTS_DIR / "S9_offboard_connection_loss.png",
            )
            m = _metrics_from_records(res.records)
            res.passed = m["final_state"] == "ABORTED" and m["final_reason"] == "offboard_lost_or_manual_override"
            res.metrics = m
            res.summary = f"final={m['final_state']} ({m['final_reason']})"
            results.append(res)
            print(f"[{'PASS' if res.passed else 'FAIL'}] {res.name}: {res.summary}", flush=True)

        # Run T0 last because it releases the harness mavsdk_server so px4_controller.py owns udp://:14540
        if wanted("T0_subprocess_px4_controller"):
            res = await run_t0_subprocess_px4_controller(drone, att, gz)
            results.append(res)
            print(f"[{'PASS' if res.passed else 'FAIL'}] {res.name}: {res.summary}", flush=True)
        else:
            await att.stop()
    finally:
        gz.close()
        sitl.stop()

    write_validation_report(results, REPORTS_DIR / "sitl_validation_report.md")
    return results


def write_validation_report(results: List[ScenarioResult], report_path: Path):
    report_path.parent.mkdir(parents=True, exist_ok=True)
    passed_count = sum(1 for r in results if r.passed)
    lines = [
        "# PX4 SITL + Gazebo Acoustic Guidance Validation Report",
        "",
        "> **Scope:** Software-in-the-loop simulation using **PX4 SITL v1.15.4 + Gazebo Garden (`gz_x500`) + MAVSDK-Python 3.17.4**, "
        "running the actual `guidance.py`, `px4_controller.py`, and UMA-16 front-only `FrequencyDetector` + `ArrayDOA` pipeline.",
        "> Simulated results validate closed-loop control logic and PX4 Offboard behavior; they **do not** replace controlled physical hardware acoustic/propeller-noise flight tests.",
        "",
        f"**Overall Result:** {passed_count}/{len(results)} scenarios PASSED",
        "",
        "## Summary Table",
        "",
        "| Scenario | Status | Summary |",
        "|---|---|---|",
    ]
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        lines.append(f"| `{r.name}` | **{status}** | {r.summary} |")

    lines.extend(["", "## Detailed Metrics & Observations", ""])
    for r in results:
        lines.append(f"### `{r.name}` — {'PASS' if r.passed else 'FAIL'}")
        lines.append(f"- **Summary:** {r.summary}")
        if r.metrics:
            lines.append("```json")
            lines.append(json.dumps(r.metrics, indent=2))
            lines.append("```")
        if r.plot_path is not None and r.plot_path.exists():
            lines.append(f"- **Plot:** [{r.plot_path.name}](file://{r.plot_path.resolve()})")
        lines.append("")

    report_path.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gz-gui", action="store_true", help="launch Gazebo Garden with 3D GUI instead of headless")
    parser.add_argument("--live-viz", action="store_true", help="open live Matplotlib 3D visualization window")
    parser.add_argument("--only", default=None, help="run only scenarios matching substring")
    args = parser.parse_args()
    if args.gz_gui or args.live_viz:
        apply_gui_environment()

    results = asyncio.run(
        run_all_scenarios(
            headless_gz=not args.gz_gui,
            live_viz=args.live_viz,
            only_filter=args.only,
        )
    )
    failed = [r for r in results if not r.passed]
    print(f"\n=== VALIDATION COMPLETE: {len(results) - len(failed)}/{len(results)} PASSED ===")
    print(f"Report: {REPORTS_DIR / 'sitl_validation_report.md'}")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
