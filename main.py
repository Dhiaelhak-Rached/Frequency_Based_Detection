#!/usr/bin/env python3
"""Unified entry point for Frequency-Locked Dual-Array Acoustic Triangulation.

Usage:
  # Launch 2D Tactical Radar GUI:
  python3 main.py gui
  python3 main.py gui --simulate

  # Run headless terminal tracking:
  python3 main.py cli --low 900 --high 1100 --threshold -65

  # Run automated mathematical verification:
  python3 main.py simulate
  python3 main.py test-geometry

  # Check connected USB audio hardware:
  python3 main.py list-devices
"""

import argparse
import sys
import time
import tkinter as tk

import numpy as np

from geometry import RigGeometry, vector_to_bearing, bearing_to_vector
from detector import FrequencyDetector
from doa_engine import ArrayDOA
from triangulation import Triangulator
from audio_manager import MultiDeviceAudioManager


def parse_args():
    p = argparse.ArgumentParser(description="Frequency-Locked Dual-Array Acoustic Triangulation")
    sub = p.add_subparsers(dest="command", required=True)

    # 1. GUI
    pg = sub.add_parser("gui", help="launch 2D tactical radar GUI")
    pg.add_argument("--simulate", action="store_true", help="run in simulation mode with virtual moving source")
    pg.add_argument("--uma8-only", action="store_true", help="use only the front UMA-8 and report direction without range")
    pg.add_argument("--uma16-only", action="store_true", help="use only the left UMA-16 and report direction without range")
    pg.add_argument("--low", type=float, default=None, help="lower target band limit (Hz)")
    pg.add_argument("--high", type=float, default=None, help="upper target band limit (Hz)")
    pg.add_argument("--threshold", type=float, default=None, help="detection threshold in dBFS")
    pg.add_argument("--dev-front", default=None, help="device index or name for Front UMA-8")
    pg.add_argument("--dev-left", default=None, help="device index or name for Left UMA-16")

    # 2. CLI
    pc = sub.add_parser("cli", help="run headless terminal tracking")
    pc.add_argument("--simulate", action="store_true", help="run with virtual acoustic source")
    pc.add_argument("--uma8-only", action="store_true", help="use only the front UMA-8 and report direction without range")
    pc.add_argument("--uma16-only", action="store_true", help="use only the left UMA-16 and report direction without range")
    pc.add_argument("--low", type=float, default=900.0, help="lower target band limit (Hz)")
    pc.add_argument("--high", type=float, default=1100.0, help="upper target band limit (Hz)")
    pc.add_argument("--threshold", type=float, default=-65.0, help="detection threshold (dBFS)")
    pc.add_argument("--average", type=int, default=4, help="number of averaged blocks")
    pc.add_argument("--dev-front", default=None, help="device index or name for Front UMA-8")
    pc.add_argument("--dev-left", default=None, help="device index or name for Left UMA-16")

    # 3. Simulate
    ps = sub.add_parser("simulate", help="run offline math verification benchmark")
    ps.add_argument("--target-x", type=float, default=1.5, help="simulated target X position (m)")
    ps.add_argument("--target-y", type=float, default=3.0, help="simulated target Y position (m)")
    ps.add_argument("--freq", type=float, default=1000.0, help="simulated tone frequency (Hz)")
    ps.add_argument("--snr", type=float, default=12.0, help="signal to noise ratio in dB")

    # 4. Test Geometry
    sub.add_parser("test-geometry", help="verify dual-array coordinates and XMOS-board distance")

    # 5. List Devices
    sub.add_parser("list-devices", help="list audio input devices")

    args = p.parse_args()
    if getattr(args, "uma8_only", False) and getattr(args, "uma16_only", False):
        p.error("--uma8-only and --uma16-only are mutually exclusive")
    return args


def cmd_gui(args):
    from tactical_gui import TacticalGUI
    geo = RigGeometry()
    root = tk.Tk()
    app = TacticalGUI(root, geometry=geo, simulate=args.simulate)
    app.var_front_only.set(args.uma8_only)
    app.var_left_only.set(args.uma16_only)
    if args.low is not None:
        app.var_low.set(str(args.low))
    if args.high is not None:
        app.var_high.set(str(args.high))
    if args.threshold is not None:
        app.var_thresh.set(str(args.threshold))
    if args.dev_front is not None:
        app.var_dev_front.set(str(args.dev_front))
    if args.dev_left is not None:
        app.var_dev_left.set(str(args.dev_left))
    root.mainloop()


def cmd_cli(args):
    geo = RigGeometry()
    detector = FrequencyDetector(
        sample_rate=geo.sample_rate,
        block_duration=0.2,
        average_blocks=args.average,
        threshold_dbfs=args.threshold,
    )
    if args.uma8_only:
        selected_ids = ["front"]
    elif args.uma16_only:
        selected_ids = ["left"]
    else:
        selected_ids = list(geo.arrays)
    single_id = selected_ids[0] if len(selected_ids) == 1 else None
    doa_engines = {}
    for arr_id in selected_ids:
        arr = geo.arrays[arr_id]
        doa_engines[arr_id] = ArrayDOA(
            array_id=arr_id,
            mic_positions_local=arr.local_mics,
            orientation_deg=arr.orientation_deg,
            sample_rate=geo.sample_rate,
            mount=arr.mount,
        )
    triangulator = Triangulator(geometry=geo, min_prominence=1.8)
    audio_mgr = MultiDeviceAudioManager(
        geometry=geo,
        sample_rate=geo.sample_rate,
        block_duration=0.2,
        simulate=args.simulate,
    )

    print("===================================================================")
    print(f" {geo.arrays[single_id].name} Direction Tracker" if single_id else " Dual-Array Acoustic Triangulation Tracker")
    print(f" Target Band: {args.low:g} - {args.high:g} Hz | Threshold: {args.threshold:g} dBFS")
    print(f" Capture: {'SIMULATION' if args.simulate else 'HARDWARE'} | Arrays: {', '.join(selected_ids)}")
    print(" Press Ctrl+C to stop.")
    print("===================================================================\n")

    dev_map = {}
    if args.dev_front is not None:
        dev_map["front"] = args.dev_front
    if args.dev_left is not None:
        dev_map["left"] = args.dev_left

    audio_mgr.start(device_map=dev_map if dev_map else None, array_ids=selected_ids)
    try:
        while True:
            blocks = audio_mgr.get_latest_blocks()
            if not blocks:
                time.sleep(0.05)
                continue

            det_block = blocks.get("front", next(iter(blocks.values())))
            det_res = detector.process(det_block[:, 0], args.low, args.high)

            if det_res.detected:
                bearings = {}
                proms = {}
                results = {}
                for arr_id, doa in doa_engines.items():
                    if arr_id in blocks:
                        res = doa.estimate(blocks[arr_id], target_freq=det_res.peak_freq, band=(args.low, args.high))
                        results[arr_id] = res
                        bearings[arr_id] = res.global_angle_deg
                        proms[arr_id] = res.prominence

                if single_id:
                    single = results[single_id]
                    array_label = "UMA-8" if single_id == "front" else "UMA-16"
                    if single.global_angle_deg is not None and (single.prominence >= 1.4 or single.stable):
                        print(
                            f"[{array_label} BEARING] {single.global_angle_deg:5.1f}° | "
                            f"Freq: {det_res.peak_freq:6.1f}Hz ({det_res.peak_db:5.1f} dBFS) | Direction only",
                            flush=True,
                        )
                    elif single.error == "spatial_aliasing":
                        print(
                            f"[DOA BLOCKED] {det_res.peak_freq:6.1f}Hz exceeds "
                            f"{doa_engines[single_id].max_direction_freq_hz:.0f}Hz safe direction limit",
                            flush=True,
                        )
                    else:
                        print(f"[FREQ DETECTED] {det_res.peak_freq:6.1f}Hz | Waiting reliable {array_label} bearing", flush=True)
                    time.sleep(0.1)
                    continue

                tri = triangulator.triangulate(
                    bearings,
                    proms,
                    stable={arr_id: res.stable for arr_id, res in results.items()},
                )
                if tri.success and tri.x is not None:
                    print(
                        f"[TARGET LOCKED] (X: {tri.x:+5.2f}m, Y: {tri.y:+5.2f}m) | "
                        f"Range: {tri.range_m:4.2f}m | Bearing: {tri.bearing_deg:5.1f}° | "
                        f"Freq: {det_res.peak_freq:6.1f}Hz ({det_res.peak_db:5.1f} dBFS) | "
                        f"Rays: {tri.active_rays}/{len(selected_ids)}",
                        flush=True,
                    )
                else:
                    b_str = " ".join(f"{k}:{v:.0f}°" for k, v in bearings.items() if v is not None)
                    print(
                        f"[FREQ DETECTED] {det_res.peak_freq:6.1f}Hz ({det_res.peak_db:5.1f} dBFS) | "
                        f"Bearings: [{b_str}] | Waiting line intersection",
                        flush=True,
                    )
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nStopping audio capture...")
    finally:
        audio_mgr.stop()


def cmd_simulate(args):
    geo = RigGeometry()
    print("===================================================================")
    print(" Running Offline Wavefield Simulation & Triangulation Benchmark")
    print(f" Virtual Target Position: X = {args.target_x:+.2f} m, Y = {args.target_y:+.2f} m")
    print(f" Target Frequency: {args.freq:.1f} Hz | SNR: {args.snr:.1f} dB")
    print("===================================================================\n")

    audio_mgr = MultiDeviceAudioManager(geometry=geo, simulate=True)
    audio_mgr.sim_speed_rad_per_s = 0.0  # static target
    audio_mgr.set_simulation_target(x=args.target_x, y=args.target_y, freq_hz=args.freq, snr_db=args.snr)

    detector = FrequencyDetector(sample_rate=geo.sample_rate, threshold_dbfs=-65.0)
    doa_engines = {
        arr_id: ArrayDOA(
            arr_id,
            arr.local_mics,
            arr.orientation_deg,
            geo.sample_rate,
            mount=arr.mount,
        )
        for arr_id, arr in geo.arrays.items()
    }
    triangulator = Triangulator(geo, min_prominence=1.5)

    errors = []
    bearings_log = {arr_id: [] for arr_id in geo.arrays}

    for step in range(15):
        blocks = audio_mgr.get_latest_blocks()
        det_res = detector.process(blocks["front"][:, 0], args.freq - 100, args.freq + 100)
        bearings = {}
        proms = {}
        stable = {}
        for arr_id, doa in doa_engines.items():
            res = doa.estimate(blocks[arr_id], target_freq=det_res.peak_freq, band=(args.freq - 100, args.freq + 100))
            bearings[arr_id] = res.global_angle_deg
            proms[arr_id] = res.prominence
            stable[arr_id] = res.stable
            if res.global_angle_deg is not None:
                bearings_log[arr_id].append(res.global_angle_deg)

        tri = triangulator.triangulate(bearings, proms, stable=stable)
        if tri.success and tri.x is not None:
            err_m = np.linalg.norm(np.array([tri.x, tri.y]) - np.array([args.target_x, args.target_y]))
            errors.append(err_m)

    true_range = np.linalg.norm([args.target_x, args.target_y])
    true_bearing = vector_to_bearing(np.array([args.target_x, args.target_y]))

    print(f"True Target:  X = {args.target_x:+5.2f} m, Y = {args.target_y:+5.2f} m | Range: {true_range:4.2f} m | Bearing: {true_bearing:5.1f}°")
    for arr_id in geo.arrays:
        if bearings_log[arr_id]:
            angles_rad = np.deg2rad(bearings_log[arr_id])
            mean_vector = np.mean(np.exp(1j * angles_rad))
            mean_bearing = float(np.rad2deg(np.angle(mean_vector)) % 360.0)
            print(f"  {geo.arrays[arr_id].name:<15}: Estimated Bearing = {mean_bearing:5.1f}°")

    if errors:
        med_err = np.median(errors)
        max_err = np.max(errors)
        print(f"\nTriangulation Accuracy:")
        print(f"  Median Position Error: {med_err*100:4.1f} cm")
        print(f"  Maximum Position Error: {max_err*100:4.1f} cm")
        if med_err < 0.25:
            print("\n[SUCCESS] Simulation test PASSED (accuracy within 25 cm).")
        else:
            print("\n[WARNING] Simulation error larger than expected.")
    else:
        print("\n[FAIL] Triangulation failed to achieve lock.")


def cmd_test_geometry(args):
    geo = RigGeometry()
    print("===================================================================")
    print(" Rig Geometry & Microphone Array Configuration")
    print(f" Sound Speed: {geo.sound_speed} m/s | Sample Rate: {geo.sample_rate} Hz")
    print("===================================================================\n")

    for arr_id, arr in geo.arrays.items():
        cfg = geo.cfg["arrays"][arr_id]
        print(f"Array: {arr.name} (Type: {arr.array_type})")
        print(f"  Acoustic Center: X = {arr.center[0]:+6.3f} m, Y = {arr.center[1]:+6.3f} m")
        if "xmos_center" in cfg:
            ref = cfg["xmos_center"]
            print(f"  XMOS Reference:  X = {ref[0]:+6.3f} m, Y = {ref[1]:+6.3f} m")
        print(f"  Orientation:     {arr.orientation_deg:+5.1f}° relative to forward (+Y)")
        print(f"  Active Mics:     {len(arr.mic_positions)} microphones {sorted(arr.mic_positions)}")
        xs = [pos[0] for pos in arr.mic_positions.values()]
        ys = [pos[1] for pos in arr.mic_positions.values()]
        print(f"  Aperture X:      [{min(xs):+6.3f} m, {max(xs):+6.3f} m]")
        print(f"  Aperture Y:      [{min(ys):+6.3f} m, {max(ys):+6.3f} m]\n")

    meas = geo.cfg.get("physical_measurements", {})
    refs = {
        arr_id: np.array(cfg["xmos_center"] if "xmos_center" in cfg else cfg["center"], dtype=float)
        for arr_id, cfg in geo.cfg["arrays"].items()
    }
    checks = (
        ("Apex to Left", "front", "left", "apex_to_left_m"),
    )
    print("Physical Rig Checks (XMOS board to XMOS board):")
    for label, first, second, key in checks:
        actual = np.linalg.norm(refs[second] - refs[first])
        expected = float(meas[key])
        if not np.isclose(actual, expected, atol=1e-6):
            print(f"  {label}: MISMATCH, configured {actual*100:.1f} cm vs measured {expected*100:.1f} cm")
            return
        print(f"  {label}: {actual*100:.1f} cm")
    print(f"  Total Channels: {sum(len(a.mic_positions) for a in geo.arrays.values())} active microphones")
    print("\n[SUCCESS] Geometry model verified successfully.")


def cmd_list_devices(args):
    devices = MultiDeviceAudioManager.list_available_devices()
    print(f"Found {len(devices)} audio input device(s):\n")
    for d in devices:
        print(f"  [{d['index']:2d}] {d['name']} ({d['channels']} in, {d['samplerate']:.0f} Hz)")


def main():
    args = parse_args()
    if args.command == "gui":
        cmd_gui(args)
    elif args.command == "cli":
        cmd_cli(args)
    elif args.command == "simulate":
        cmd_simulate(args)
    elif args.command == "test-geometry":
        cmd_test_geometry(args)
    elif args.command == "list-devices":
        cmd_list_devices(args)


if __name__ == "__main__":
    main()
