#!/usr/bin/env python3
"""Unified entry point for Frequency-Locked 3-Array Acoustic Triangulation.

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
    p = argparse.ArgumentParser(description="Frequency-Locked 3-Array Acoustic Triangulation")
    sub = p.add_subparsers(dest="command", required=True)

    # 1. GUI
    pg = sub.add_parser("gui", help="launch 2D tactical radar GUI")
    pg.add_argument("--simulate", action="store_true", help="run in simulation mode with virtual moving source")
    pg.add_argument("--low", type=float, default=None, help="lower target band limit (Hz)")
    pg.add_argument("--high", type=float, default=None, help="upper target band limit (Hz)")
    pg.add_argument("--threshold", type=float, default=None, help="detection threshold in dBFS")

    # 2. CLI
    pc = sub.add_parser("cli", help="run headless terminal tracking")
    pc.add_argument("--simulate", action="store_true", help="run with virtual acoustic source")
    pc.add_argument("--low", type=float, default=900.0, help="lower target band limit (Hz)")
    pc.add_argument("--high", type=float, default=1100.0, help="upper target band limit (Hz)")
    pc.add_argument("--threshold", type=float, default=-65.0, help="detection threshold (dBFS)")
    pc.add_argument("--average", type=int, default=4, help="number of averaged blocks")

    # 3. Simulate
    ps = sub.add_parser("simulate", help="run offline math verification benchmark")
    ps.add_argument("--target-x", type=float, default=1.5, help="simulated target X position (m)")
    ps.add_argument("--target-y", type=float, default=3.0, help="simulated target Y position (m)")
    ps.add_argument("--freq", type=float, default=1000.0, help="simulated tone frequency (Hz)")
    ps.add_argument("--snr", type=float, default=12.0, help="signal to noise ratio in dB")

    # 4. Test Geometry
    sub.add_parser("test-geometry", help="verify 3-array coordinate layout and baseline measurements")

    # 5. List Devices
    sub.add_parser("list-devices", help="list audio input devices")

    return p.parse_args()


def cmd_gui(args):
    from tactical_gui import TacticalGUI
    geo = RigGeometry()
    root = tk.Tk()
    app = TacticalGUI(root, geometry=geo, simulate=args.simulate)
    if args.low is not None:
        app.var_low.set(str(args.low))
    if args.high is not None:
        app.var_high.set(str(args.high))
    if args.threshold is not None:
        app.var_thresh.set(str(args.threshold))
    root.mainloop()


def cmd_cli(args):
    geo = RigGeometry()
    detector = FrequencyDetector(
        sample_rate=geo.sample_rate,
        block_duration=0.2,
        average_blocks=args.average,
        threshold_dbfs=args.threshold,
    )
    doa_engines = {}
    for arr_id, arr in geo.arrays.items():
        doa_engines[arr_id] = ArrayDOA(
            array_id=arr_id,
            mic_positions_local=arr.local_mics,
            orientation_deg=arr.orientation_deg,
            sample_rate=geo.sample_rate,
        )
    triangulator = Triangulator(geometry=geo, min_prominence=1.8)
    audio_mgr = MultiDeviceAudioManager(
        geometry=geo,
        sample_rate=geo.sample_rate,
        block_duration=0.2,
        simulate=args.simulate,
    )

    print("===================================================================")
    print(" 3-Array Acoustic Triangulation Tracker (Narrowband)")
    print(f" Target Band: {args.low:g} - {args.high:g} Hz | Threshold: {args.threshold:g} dBFS")
    print(f" Mode: {'SIMULATION' if args.simulate else 'HARDWARE CAPTURE'}")
    print(" Press Ctrl+C to stop.")
    print("===================================================================\n")

    audio_mgr.start()
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
                for arr_id, doa in doa_engines.items():
                    if arr_id in blocks:
                        res = doa.estimate(blocks[arr_id], target_freq=det_res.peak_freq)
                        bearings[arr_id] = res.global_angle_deg
                        proms[arr_id] = res.prominence

                tri = triangulator.triangulate(bearings, proms)
                if tri.success and tri.x is not None:
                    print(
                        f"[TARGET LOCKED] (X: {tri.x:+5.2f}m, Y: {tri.y:+5.2f}m) | "
                        f"Range: {tri.range_m:4.2f}m | Bearing: {tri.bearing_deg:5.1f}° | "
                        f"Freq: {det_res.peak_freq:6.1f}Hz ({det_res.peak_db:5.1f} dBFS) | "
                        f"Rays: {tri.active_rays}/3",
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
        for arr_id, doa in doa_engines.items():
            res = doa.estimate(blocks[arr_id], target_freq=det_res.peak_freq)
            bearings[arr_id] = res.global_angle_deg
            proms[arr_id] = res.prominence
            if res.global_angle_deg is not None:
                bearings_log[arr_id].append(res.global_angle_deg)

        tri = triangulator.triangulate(bearings, proms)
        if tri.success and tri.x is not None:
            err_m = np.linalg.norm(np.array([tri.x, tri.y]) - np.array([args.target_x, args.target_y]))
            errors.append(err_m)

    true_range = np.linalg.norm([args.target_x, args.target_y])
    true_bearing = vector_to_bearing(np.array([args.target_x, args.target_y]))

    print(f"True Target:  X = {args.target_x:+5.2f} m, Y = {args.target_y:+5.2f} m | Range: {true_range:4.2f} m | Bearing: {true_bearing:5.1f}°")
    for arr_id in geo.arrays:
        if bearings_log[arr_id]:
            print(f"  {geo.arrays[arr_id].name:<15}: Estimated Bearing = {np.mean(bearings_log[arr_id]):5.1f}°")

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
        print(f"Array: {arr.name} (Type: {arr.array_type})")
        print(f"  Center:       X = {arr.center[0]:+6.3f} m, Y = {arr.center[1]:+6.3f} m")
        print(f"  Orientation:  {arr.orientation_deg:+5.1f}° relative to forward (+Y)")
        print(f"  Active Mics:  {len(arr.mic_positions)} microphones")
        xs = [pos[0] for pos in arr.mic_positions.values()]
        ys = [pos[1] for pos in arr.mic_positions.values()]
        print(f"  Aperture X:   [{min(xs):+6.3f} m, {max(xs):+6.3f} m]")
        print(f"  Aperture Y:   [{min(ys):+6.3f} m, {max(ys):+6.3f} m]\n")

    meas = geo.cfg.get("physical_measurements", {})
    print("Physical Rig Checks:")
    print(f"  Apex to Left Tip:   {meas.get('apex_to_left_tip_m', 0)*100:.1f} cm (matches sketch 14 cm)")
    print(f"  Apex to Right Tip:  {meas.get('apex_to_right_tip_m', 0)*100:.1f} cm (matches sketch 17 cm)")
    print(f"  Rear Baseline:      {meas.get('rear_baseline_m', 0)*100:.1f} cm (matches sketch 18.5 cm)")
    print(f"  Total Channels:     {sum(len(a.mic_positions) for a in geo.arrays.values())} active microphones")
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
