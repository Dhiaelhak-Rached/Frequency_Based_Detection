"""Telemetry payload serialization and formatting helpers shared across CLI, GUI, and simulation."""

from __future__ import annotations

import base64
import time

import cv2
import numpy as np


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


def build_guidance_payload(
    array_mode,
    audio_block=None,
    doa=None,
    suppressed=False,
    mount_calibration_id=None,
):
    if array_mode != "uma16_front_only" or audio_block is None:
        return None
    return {
        "array_id": "left",
        "array_mode": array_mode,
        "mount_calibration_id": mount_calibration_id,
        "captured_at": float(audio_block.captured_at),
        "sequence": int(audio_block.sequence),
        "azimuth_deg": None if doa is None else doa.global_angle_deg,
        "elevation_deg": None if doa is None else doa.elevation_deg,
        "azimuth_prominence": 0.0 if doa is None else float(doa.prominence),
        "elevation_prominence": 0.0 if doa is None else float(doa.elevation_prominence),
        "azimuth_stable": bool(doa is not None and doa.stable),
        "elevation_stable": bool(doa is not None and doa.elevation_stable),
        "error": None if doa is None else doa.error,
        "suppressed": bool(suppressed),
    }


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
    guidance_audio_block=None,
    mount_calibration_id=None,
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
        "emitted_at": round(float(time.monotonic()), 6),
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
            "reason": fused.reason,
        },
        "detection": det_payload,
        "classification": {
            "text": format_classification(
                classification,
                classifier,
                bool(det_res and det_res.detected and not suppressed),
            ),
            "is_drone": bool(classification is not None and classification.is_drone),
        },
        "camera": {
            "status": cam_status,
            "target_detected": v_target,
            "bearing_deg": v_bearing,
            "elevation_deg": v_el,
            "uncalibrated": v_uncal,
            "timestamp": None if vision_res is None else round(float(vision_res.timestamp), 6),
            "frame_jpg_b64": frame_b64,
        },
        "single_doa": (
            None
            if single_doa is None
            else {
                "array_id": single_id,
                "global_angle_deg": (
                    None if single_doa.global_angle_deg is None
                    else round(float(single_doa.global_angle_deg), 2)
                ),
                "elevation_deg": (
                    None if single_doa.elevation_deg is None
                    else round(float(single_doa.elevation_deg), 2)
                ),
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
        "guidance": build_guidance_payload(
            array_mode=array_mode,
            audio_block=guidance_audio_block,
            doa=single_doa,
            suppressed=suppressed,
            mount_calibration_id=mount_calibration_id,
        ),
    }
