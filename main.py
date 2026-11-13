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

  # Check connected USB audio hardware and cameras:
  python3 main.py list-devices
  python3 main.py list-cameras
"""

import argparse
import base64
import json
from pathlib import Path
import time
import tkinter as tk

import cv2
import numpy as np

from geometry import RigGeometry, vector_to_bearing
from detector import FrequencyDetector
from direction_suppression import DirectionSuppression, displayable_doa, select_front_only_target
from doa_engine import ArrayDOA
from triangulation import Triangulator
from audio_manager import MultiDeviceAudioManager
from classifier import DroneClassifier
from vision_detector import DroneCamera, discover_cameras, normalize_camera_source
from tracked_camera import TrackedDroneCamera
from target_state_machine import StateMachineConfig, TargetStateMachine, format_fused_state


def resolve_config_path(geo, value):
    path = Path(value)
    if path.is_absolute():
        return path
    return Path(geo.config_path).resolve().parent / path


def create_classifier(geo, model_path=None, disabled=False):
    cfg = geo.cfg.get("classifier", {})
    if disabled or not cfg.get("enabled", True):
        return None
    configured_model = model_path or cfg.get("model_path", "model/mobilenetv2_drone.onnx")
    metadata_path = cfg.get("metadata_path", "model/metadata.json")
    return DroneClassifier(
        model_path=resolve_config_path(geo, configured_model),
        metadata_path=resolve_config_path(geo, metadata_path),
        sample_rate=geo.sample_rate,
        threshold=cfg.get("threshold"),
        votes_required=cfg.get("votes_required"),
        vote_window=cfg.get("vote_window"),
    )


def format_classification(result, classifier, signal_detected=True):
    if classifier is None:
        return "Classifier: OFF"
    if not signal_detected:
        return "Classifier: NO TARGET SIGNAL"
    if result is None or not result.is_ready:
        return "Classifier: BUFFERING"
    if result.is_drone:
        label = "DRONE CONFIRMED"
    elif result.probability >= result.threshold:
        label = "DRONE CANDIDATE"
    else:
        label = "NO DRONE"
    return (
        f"{label} {result.probability * 100.0:.1f}% "
        f"({result.decision_votes}/{result.vote_window})"
    )


def format_elevation(result):
    if result is None or result.elevation_deg is None:
        return "Elevation: UNCERTAIN"
    elevation = result.elevation_deg
    direction = "ABOVE" if elevation > 2.0 else "BELOW" if elevation < -2.0 else "LEVEL"
    return f"Elevation: {elevation:+.1f}° ({direction})"


def add_classifier_arguments(parser):
    parser.add_argument("--model", default=None, help="custom path to the drone classifier ONNX model")
    parser.add_argument("--no-classifier", action="store_true", help="disable drone classification")
    parser.add_argument(
        "--require-drone",
        action="store_true",
        help="require confirmed drone classification before declaring a target lock",
    )


def create_vision(
    geo,
    camera=None,
    model_path=None,
    camera_fov=None,
    disabled=False,
    tracker=None,
    detector_every=None,
    no_tracker=False,
):
    cfg = geo.cfg.get("vision", {})
    if disabled or not cfg.get("enabled", True):
        return None
    configured_model = model_path or cfg.get("model_path", "model/Yolo/best.pt")
    selected_camera = camera if camera is not None else cfg.get("camera", "0")
    camera_key = str(normalize_camera_source(selected_camera))
    camera_fovs = cfg.get("camera_fovs", {})
    if camera_fov is not None:
        horizontal_fov = camera_fov
    elif camera_key in camera_fovs:
        horizontal_fov = camera_fovs[camera_key]
    else:
        horizontal_fov = cfg.get("horizontal_fov_deg") if camera is None else None
    common = dict(
        model_path=resolve_config_path(geo, configured_model),
        camera=selected_camera,
        confidence=float(cfg.get("confidence", 0.5)),
        inference_hz=float(cfg.get("inference_hz", 10.0)),
        horizontal_fov_deg=horizontal_fov,
        orientation_deg=float(cfg.get("orientation_deg", 0.0)),
        max_target_step_px=float(cfg.get("max_target_step_px", 120.0)),
        confirmation_frames=int(cfg.get("confirmation_frames", 3)),
        lost_timeout_s=float(cfg.get("lost_timeout_s", 0.75)),
    )
    tracker_config = dict(cfg.get("tracker", {}))
    tracker_enabled = not no_tracker and (tracker_config.get("enabled", False) or tracker is not None)
    if not tracker_enabled:
        return DroneCamera(**common)
    if tracker is not None:
        tracker_config["name"] = tracker
    if detector_every is not None:
        tracker_config["detector_every"] = detector_every
    for key in ("library_path", "checkpoint_path"):
        if key in tracker_config:
            tracker_config[key] = str(resolve_config_path(geo, tracker_config[key]))
    return TrackedDroneCamera(**common, tracker_config=tracker_config)


def format_vision(result, vision):
    if vision is None:
        return "Camera: OFF"
    if result is None or result.timestamp == 0.0 and result.error is None:
        return "Camera: CONNECTING"
    if result.error:
        if result.error.startswith("Tracker error:"):
            return "Camera: TRACKER OFFLINE"
        return "Camera: OFFLINE"
    if result.target is not None:
        tracked = result.output_source == "tracker"
        label = "DRONE TRACKING" if tracked else f"DRONE {result.best_confidence:.1%}"
        if result.direction is None:
            return f"Camera: {label} | DIRECTION UNCALIBRATED"
        direction = result.direction
        return (
            f"Camera: {label} | "
            f"Bearing: {direction.global_bearing_deg:.1f}° | "
            f"Elevation: {direction.elevation_deg:+.1f}°"
        )
    if result.detections:
        return f"Camera: ACQUIRING {result.best_confidence:.1%}"
    if result.tracking_state == "search":
        return "Camera: SEARCHING"
    return "Camera: CLEAR"


def add_vision_arguments(parser):
    parser.add_argument("--camera", default=None, help="camera index or /dev/video device path")
    parser.add_argument("--camera-fov", type=float, default=None, help="horizontal camera FOV in degrees; required for direction")
    parser.add_argument("--yolo-model", default=None, help="custom path to the YOLO .pt model")
    parser.add_argument("--no-vision", action="store_true", help="disable camera detection")
    parser.add_argument("--tracker", choices=("dyhit",), default=None, help="enable the visual tracker")
    parser.add_argument("--detector-every", type=int, default=None, help="YOLO verification interval in tracked frames")
    parser.add_argument("--no-tracker", action="store_true", help="use YOLO detection without visual tracking")


def add_suppression_arguments(parser):
    parser.add_argument(
        "--suppress-bearing",
        type=float,
        default=None,
        help="hide a fixed sound bearing in UMA-16 front-only mode (degrees)",
    )
    parser.add_argument(
        "--suppress-tolerance",
        type=float,
        default=None,
        help="half-width of the suppressed bearing sector in degrees",
    )


def create_direction_suppression(geo, args):
    config = geo.cfg.get("uma16_front_only", {}).get("direction_suppression", {})
    return DirectionSuppression.from_config(
        config,
        bearing_override=args.suppress_bearing,
        tolerance_override=args.suppress_tolerance,
    )


def parse_args():
    p = argparse.ArgumentParser(description="Frequency-Locked Dual-Array Acoustic Triangulation")
    sub = p.add_subparsers(dest="command", required=True)

    # 1. GUI
    pg = sub.add_parser("gui", help="launch 2D tactical radar GUI")
    pg.add_argument("--simulate", action="store_true", help="run in simulation mode with virtual moving source")
    pg.add_argument("--uma8-only", action="store_true", help="use only the front UMA-8 and report direction without range")
    pg.add_argument("--uma16-only", action="store_true", help="use only the left UMA-16 and report direction without range")
    pg.add_argument(
        "--uma16-front-only",
        action="store_true",
        help="use only the UMA-16 at the front origin, facing forward, and report direction without range",
    )
    pg.add_argument("--low", type=float, default=None, help="lower target band limit (Hz)")
    pg.add_argument("--high", type=float, default=None, help="upper target band limit (Hz)")
    pg.add_argument("--threshold", type=float, default=None, help="detection threshold in dBFS")
    pg.add_argument("--dev-front", default=None, help="device index or name for Front UMA-8")
    pg.add_argument("--dev-left", default=None, help="device index or name for UMA-16")
    add_classifier_arguments(pg)
    add_vision_arguments(pg)
    add_suppression_arguments(pg)

    # 2. CLI
    pc = sub.add_parser("cli", help="run headless terminal tracking")
    pc.add_argument("--simulate", action="store_true", help="run with virtual acoustic source")
    pc.add_argument("--uma8-only", action="store_true", help="use only the front UMA-8 and report direction without range")
    pc.add_argument("--uma16-only", action="store_true", help="use only the left UMA-16 and report direction without range")
    pc.add_argument(
        "--uma16-front-only",
        action="store_true",
        help="use only the UMA-16 at the front origin, facing forward, and report direction without range",
    )
    pc.add_argument("--low", type=float, default=900.0, help="lower target band limit (Hz)")
    pc.add_argument("--high", type=float, default=1100.0, help="upper target band limit (Hz)")
    pc.add_argument("--threshold", type=float, default=-65.0, help="detection threshold (dBFS)")
    pc.add_argument("--average", type=int, default=4, help="number of averaged blocks")
    pc.add_argument("--dev-front", default=None, help="device index or name for Front UMA-8")
    pc.add_argument("--dev-left", default=None, help="device index or name for UMA-16")
    add_classifier_arguments(pc)
    add_vision_arguments(pc)
    add_suppression_arguments(pc)
    pc.add_argument("--stream-json", action="store_true", help="emit machine-readable @JSON telemetry lines")
    pc.add_argument("--stream-video", action="store_true", help="include JPEG camera preview in --stream-json")

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
    sub.add_parser("list-cameras", help="list usable camera indices and device paths")

    args = p.parse_args()
    array_modes = sum(
        bool(getattr(args, name, False))
        for name in ("uma8_only", "uma16_only", "uma16_front_only")
    )
    if array_modes > 1:
        p.error("--uma8-only, --uma16-only, and --uma16-front-only are mutually exclusive")
    if getattr(args, "require_drone", False) and getattr(args, "no_classifier", False):
        p.error("--require-drone cannot be combined with --no-classifier")
    if getattr(args, "no_tracker", False) and (
        getattr(args, "tracker", None) is not None or getattr(args, "detector_every", None) is not None
    ):
        p.error("--no-tracker cannot be combined with --tracker or --detector-every")
    if getattr(args, "detector_every", None) is not None and args.detector_every < 1:
        p.error("--detector-every must be at least 1")
    suppression_args = (
        getattr(args, "suppress_bearing", None) is not None
        or getattr(args, "suppress_tolerance", None) is not None
    )
    if suppression_args and not args.uma16_front_only:
        p.error("--suppress-bearing and --suppress-tolerance require --uma16-front-only")
    if suppression_args:
        try:
            create_direction_suppression(RigGeometry(), args)
        except ValueError as exc:
            p.error(str(exc))
    return args


def cmd_gui(args):
    from tactical_gui import TacticalGUI
    geo = RigGeometry()
    if args.uma16_front_only:
        geo.set_uma16_front_only(True)
    try:
        classifier = create_classifier(geo, model_path=args.model, disabled=args.no_classifier)
    except Exception as exc:
        print(f"[WARNING] Classifier unavailable: {exc}")
        classifier = None
    try:
        vision = create_vision(
            geo,
            camera=args.camera,
            model_path=args.yolo_model,
            camera_fov=args.camera_fov,
            disabled=args.no_vision,
            tracker=args.tracker,
            detector_every=args.detector_every,
            no_tracker=args.no_tracker,
        )
    except Exception as exc:
        print(f"[WARNING] Camera detection unavailable: {exc}")
        vision = None
    root = tk.Tk()
    app = TacticalGUI(
        root,
        geometry=geo,
        simulate=args.simulate,
        classifier=classifier,
        require_drone=args.require_drone,
        vision=vision,
        suppression=create_direction_suppression(geo, args),
    )
    app.var_front_only.set(args.uma8_only)
    app.var_left_only.set(args.uma16_only)
    app.var_uma16_front_only.set(args.uma16_front_only)
    app._update_suppression_controls()
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


def build_telemetry_payload(
    fused,
    det_res=None,
    single_doa=None,
    single_id=None,
    tri=None,
    classification=None,
    classifier=None,
    vision_res=None,
    vision=None,
    suppression=None,
    doa_errors=None,
    array_mode="dual",
    suppressed=False,
    audio_waiting=False,
    stream_video=False,
):
    """Build a compact JSON-serializable telemetry dict for remote GUI streaming."""
    det_payload = None
    if det_res is not None:
        max_f = min(max(float(det_res.band[1]) * 1.4, 3000.0), 16000.0)
        freqs = np.asarray(det_res.freqs, dtype=float)
        mags = np.asarray(det_res.mags, dtype=float)
        mask = freqs <= max_f
        freqs = freqs[mask]
        mags = mags[mask]
        step = max(1, len(freqs) // 120)
        det_payload = {
            "detected": bool(det_res.detected) and not suppressed,
            "peak_freq": None if suppressed or det_res.peak_freq is None else round(float(det_res.peak_freq), 2),
            "peak_db": None if suppressed or det_res.peak_db is None else round(float(det_res.peak_db), 2),
            "threshold_db": round(float(det_res.threshold_db), 2),
            "band": [float(det_res.band[0]), float(det_res.band[1])],
            "freqs": [round(float(x), 1) for x in freqs[::step]],
            "mags": [round(float(x), 1) for x in mags[::step]],
        }

    cam_status = "OFF"
    frame_b64 = None
    v_bearing = None
    v_el = None
    v_target = False
    v_uncal = False
    if vision is not None:
        if vision_res is None:
            cam_status = "CONNECTING"
        elif vision_res.error:
            cam_status = "OFFLINE"
        else:
            v_target = vision_res.target is not None
            if vision_res.direction is not None:
                v_bearing = round(float(vision_res.direction.global_bearing_deg), 2)
                v_el = round(float(vision_res.direction.elevation_deg), 2)
            elif v_target:
                v_uncal = True
            if vision_res.tracking_state == "track":
                cam_status = "DRONE TRACKING"
            elif v_target:
                cam_status = f"DRONE ({vision_res.best_confidence:.0%})"
            else:
                cam_status = "SEARCHING"
            if stream_video and vision_res.frame is not None:
                thumb = cv2.resize(vision_res.frame, (384, 216), interpolation=cv2.INTER_AREA)
                ok, buf = cv2.imencode(".jpg", thumb, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
                if ok:
                    frame_b64 = base64.b64encode(buf.tobytes()).decode("ascii")

    return {
        "type": "telemetry",
        "audio_waiting": bool(audio_waiting),
        "suppressed": bool(suppressed),
        "array_mode": array_mode,
        "fused": {
            "mode": fused.mode.value,
            "source": fused.active_source,
            "bearing_deg": None if fused.bearing_deg is None else round(float(fused.bearing_deg), 2),
            "elevation_deg": None if fused.elevation_deg is None else round(float(fused.elevation_deg), 2),
            "direction_vector": (
                None if fused.direction_vector is None
                else [round(float(v), 4) for v in fused.direction_vector]
            ),
            "vision_uncalibrated": bool(fused.vision_uncalibrated),
        },
        "detection": det_payload,
        "classification": {
            "text": format_classification(classification, classifier, bool(det_res and det_res.detected and not suppressed)),
            "is_drone": bool(classification is not None and classification.is_drone),
        },
        "camera": {
            "status": cam_status,
            "target_detected": v_target,
            "bearing_deg": v_bearing,
            "elevation_deg": v_el,
            "uncalibrated": v_uncal,
            "frame_jpg_b64": frame_b64,
        },
        "single_doa": (
            None
            if single_doa is None
            else {
                "array_id": single_id,
                "global_angle_deg": None if single_doa.global_angle_deg is None else round(float(single_doa.global_angle_deg), 2),
                "elevation_deg": None if single_doa.elevation_deg is None else round(float(single_doa.elevation_deg), 2),
                "direction_vector": (
                    None if single_doa.direction_vector is None
                    else [round(float(v), 4) for v in single_doa.direction_vector]
                ),
                "prominence": round(float(single_doa.prominence), 2),
                "stable": bool(single_doa.stable),
                "error": single_doa.error,
            }
        ),
        "triangulation": (
            None
            if tri is None
            else {
                "success": bool(tri.success),
                "x": None if tri.x is None else round(float(tri.x), 3),
                "y": None if tri.y is None else round(float(tri.y), 3),
                "range_m": None if tri.range_m is None else round(float(tri.range_m), 3),
                "bearing_deg": None if tri.bearing_deg is None else round(float(tri.bearing_deg), 2),
                "bearings": {k: round(float(v[1]), 2) for k, v in tri.rays.items()},
            }
        ),
        "suppression": (
            None
            if suppression is None
            else {
                "enabled": bool(suppression.enabled),
                "bearing_deg": float(suppression.bearing_deg),
                "tolerance_deg": float(suppression.tolerance_deg),
            }
        ),
        "doa_errors": doa_errors or {},
    }


def cmd_cli(args):
    geo = RigGeometry()
    if args.uma16_front_only:
        geo.set_uma16_front_only(True)
    suppression = create_direction_suppression(geo, args)
    detector = FrequencyDetector(
        sample_rate=geo.sample_rate,
        block_duration=0.2,
        average_blocks=args.average,
        threshold_dbfs=args.threshold,
    )
    try:
        classifier = create_classifier(geo, model_path=args.model, disabled=args.no_classifier)
    except Exception as exc:
        print(f"[WARNING] Classifier unavailable: {exc}")
        classifier = None
    try:
        vision = create_vision(
            geo,
            camera=args.camera,
            model_path=args.yolo_model,
            camera_fov=args.camera_fov,
            disabled=args.no_vision,
            tracker=args.tracker,
            detector_every=args.detector_every,
            no_tracker=args.no_tracker,
        )
    except Exception as exc:
        print(f"[WARNING] Camera detection unavailable: {exc}")
        vision = None
    classifier_hop = max(0.05, float(geo.cfg.get("classifier", {}).get("hop_seconds", 0.2)))
    classification = None
    last_classification = 0.0
    last_vision_report = 0.0

    if args.uma8_only:
        selected_ids = ["front"]
    elif args.uma16_only or args.uma16_front_only:
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
            mic_heights_local=arr.mic_heights_local,
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
    print(f" Classifier: {'ACTIVE' if classifier else 'OFF'} | Require Drone: {args.require_drone}")
    print(f" Camera: {vision.camera if vision else 'OFF'}")
    if suppression.enabled:
        print(
            f" Fixed Direction Suppression: {suppression.bearing_deg:.1f}° "
            f"± {suppression.tolerance_deg:.1f}°"
        )
    print(" Press Ctrl+C to stop.")
    print("===================================================================\n")

    dev_map = {}
    if args.dev_front is not None:
        dev_map["front"] = args.dev_front
    if args.dev_left is not None:
        dev_map["left"] = args.dev_left

    if vision is not None:
        vision.start()
    state_machine = TargetStateMachine(
        StateMachineConfig(require_drone_for_audio=args.require_drone)
    )
    array_mode = (
        "uma8_only" if args.uma8_only
        else "uma16_only" if args.uma16_only
        else "uma16_front_only" if args.uma16_front_only
        else "dual"
    )
    stream_json = bool(getattr(args, "stream_json", False))
    stream_video = bool(getattr(args, "stream_video", False))

    audio_mgr.start(device_map=dev_map if dev_map else None, array_ids=selected_ids)
    try:
        while True:
            now = time.monotonic()
            vision_res = vision.get_result() if vision is not None else None
            blocks = audio_mgr.get_latest_blocks()
            if not blocks:
                fused = state_machine.from_pipeline_outputs(now=now, vision=vision_res)
                if stream_json:
                    payload = build_telemetry_payload(
                        fused,
                        vision_res=vision_res,
                        vision=vision,
                        suppression=suppression if args.uma16_front_only else None,
                        array_mode=array_mode,
                        audio_waiting=True,
                        stream_video=stream_video,
                    )
                    print("@JSON:" + json.dumps(payload, separators=(",", ":")), flush=True)
                else:
                    print(format_fused_state(fused) + "\n", flush=True)
                time.sleep(0.1)
                continue

            det_block = blocks.get("front", next(iter(blocks.values())))
            det_channel = det_block[:, 0]
            det_res = detector.process(det_channel, args.low, args.high)
            front_doa = None
            suppressed = False
            if args.uma16_front_only:
                selection = select_front_only_target(
                    det_res,
                    blocks["left"],
                    doa_engines["left"],
                    (args.low, args.high),
                    suppression,
                )
                det_res = selection.detection
                if selection.suppressed:
                    if classifier is not None:
                        classifier.reset()
                    classification = None
                    last_classification = 0.0
                    suppressed = True
                else:
                    front_doa = selection.doa

            if not suppressed:
                if not det_res.detected:
                    if classifier is not None:
                        classifier.reset()
                    classification = None
                    last_classification = 0.0
                elif classifier is not None and now - last_classification >= classifier_hop:
                    classification = classifier.classify(det_channel)
                    last_classification = now

            single_doa = None
            tri = None
            doa_errors = {}
            if not suppressed and det_res.detected:
                bearings = {}
                proms = {}
                results = {}
                for arr_id, doa in doa_engines.items():
                    if arr_id in blocks:
                        res = front_doa if arr_id == "left" and front_doa is not None else doa.estimate(
                            blocks[arr_id],
                            target_freq=det_res.peak_freq,
                            band=(args.low, args.high),
                        )
                        results[arr_id] = res
                        bearings[arr_id] = res.global_angle_deg
                        proms[arr_id] = res.prominence
                        if res.error:
                            doa_errors[arr_id] = res.error

                if single_id:
                    single_doa = results.get(single_id)
                    if single_doa is not None and not displayable_doa(single_doa, suppression):
                        suppressed = True
                else:
                    tri = triangulator.triangulate(
                        bearings,
                        proms,
                        stable={arr_id: res.stable for arr_id, res in results.items()},
                    )

            fused = state_machine.from_pipeline_outputs(
                now=now,
                vision=vision_res,
                detection=det_res,
                single_doa=single_doa,
                triangulation=tri,
                classification=classification,
                suppressed=suppressed,
            )
            if stream_json:
                payload = build_telemetry_payload(
                    fused,
                    det_res=det_res,
                    single_doa=single_doa,
                    single_id=single_id,
                    tri=tri,
                    classification=classification,
                    classifier=classifier,
                    vision_res=vision_res,
                    vision=vision,
                    suppression=suppression if args.uma16_front_only else None,
                    doa_errors=doa_errors,
                    array_mode=array_mode,
                    suppressed=suppressed,
                    stream_video=stream_video,
                )
                print("@JSON:" + json.dumps(payload, separators=(",", ":")), flush=True)
            else:
                print(format_fused_state(fused) + "\n", flush=True)
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nStopping audio capture...")
    finally:
        audio_mgr.stop()
        if vision is not None:
            vision.stop()


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
            mic_heights_local=arr.mic_heights_local,
        )
        for arr_id, arr in geo.arrays.items()
    }
    triangulator = Triangulator(geo, min_prominence=1.5)
    try:
        classifier = create_classifier(geo)
    except Exception as exc:
        print(f"[WARNING] Classifier unavailable: {exc}")
        classifier = None
    classification = None

    if classifier is not None:
        reference_path = resolve_config_path(
            geo,
            geo.cfg.get("classifier", {}).get("reference_path", "model/reference_outputs.npz"),
        )
        try:
            reference_passed, max_diff = classifier.validate_against_reference(reference_path)
            status = "PASS" if reference_passed else "FAIL"
            print(f"Classifier Reference Validation: [{status}] max probability error = {max_diff:.6f}\n")
        except Exception as exc:
            print(f"[WARNING] Classifier reference validation failed: {exc}\n")

    errors = []
    bearings_log = {arr_id: [] for arr_id in geo.arrays}

    for step in range(15):
        blocks = audio_mgr.get_latest_blocks()
        det_res = detector.process(blocks["front"][:, 0], args.freq - 100, args.freq + 100)
        if classifier is not None:
            classification = classifier.classify(blocks["front"][:, 0])
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

    print(f"Simulated Source Classification: {format_classification(classification, classifier)}")

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


def cmd_list_cameras(args):
    devices = discover_cameras()
    if not devices:
        print("No usable cameras found. Check the USB connection and camera permissions.")
        return
    print(f"Found {len(devices)} camera(s):")
    for device in devices:
        stable = f" | stable: {device.stable_path}" if device.stable_path else ""
        print(f"  [{device.index}] {device.path}{stable}")
    print("\nUse the index or a listed path with --camera.")


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
    elif args.command == "list-cameras":
        cmd_list_cameras(args)


if __name__ == "__main__":
    main()
