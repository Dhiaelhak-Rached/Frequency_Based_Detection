from dataclasses import dataclass
import math
import queue
import threading
import time
from typing import Optional

import cv2
import numpy as np

from vision_detector import DroneCamera, VisionDetection, VisionResult, detection_center, pixel_to_direction
from visual_tracking.core import TrackerCore
from visual_tracking.logic import Detection, TargetLogic


@dataclass(frozen=True)
class TrackingSettings:
    name: str = "dyhit"
    library_path: str = "visual_tracking/tracker_lib"
    checkpoint_path: str = "visual_tracking/tracker_lib/trackers/dyhit/models/DyHiT_ep0060.pth.tar"
    device: str = "auto"
    detector_every: int = 5
    verify_iou: float = 0.3
    reacquire_gate: float = 1.5
    lost_after_s: float = 0.5
    trust_tracker_s: float = 2.0
    search_growth: float = 200.0
    stamp_tolerance: float = 0.02
    max_catch_up: int = 3
    buffer_size: int = 60

    def __post_init__(self):
        if self.name != "dyhit":
            raise ValueError("Only the vendored 'dyhit' tracker is available")
        if self.device not in ("auto", "cuda", "cpu"):
            raise ValueError("Tracker device must be auto, cuda, or cpu")
        if self.detector_every < 1:
            raise ValueError("Detector interval must be at least 1 in tracked mode")
        if not 0.0 < self.verify_iou <= 1.0:
            raise ValueError("Tracker verification IoU must be in (0, 1]")
        if self.reacquire_gate <= 0.0 or self.lost_after_s <= 0.0 or self.trust_tracker_s < 0.0:
            raise ValueError("Tracker gates and timeouts are invalid")
        if self.search_growth < 0.0 or self.stamp_tolerance <= 0.0:
            raise ValueError("Tracker search growth and timestamp tolerance are invalid")
        if self.max_catch_up < 0 or self.buffer_size < self.max_catch_up + 2:
            raise ValueError("Tracker frame buffer is too small for catch-up")

    @classmethod
    def from_config(cls, config):
        fields = cls.__dataclass_fields__
        values = {key: config[key] for key in fields if key in config}
        return cls(**values)


class LatestRequestQueue:
    def __init__(self):
        self._queue = queue.Queue(maxsize=1)
        self.dropped = 0

    def put(self, frame, stamp):
        request = (frame, stamp)
        try:
            self._queue.put_nowait(request)
            return
        except queue.Full:
            pass
        try:
            self._queue.get_nowait()
            self.dropped += 1
        except queue.Empty:
            pass
        try:
            self._queue.put_nowait(request)
        except queue.Full:
            self.dropped += 1

    def get(self, timeout=0.1):
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def clear(self):
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                return

    @property
    def size(self):
        return self._queue.qsize()


class TrackedDroneCamera(DroneCamera):
    def __init__(self, *args, tracker_config=None, tracker=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.tracking = TrackingSettings.from_config(tracker_config or {})
        self._tracker = tracker
        self._tracker_lost_score = None
        self._core = None
        self._core_lock = threading.Lock()
        self._detector_thread = None
        self._detector_stream = None
        self._torch = None
        self._detection_requests = LatestRequestQueue()
        self._last_yolo_confidence = None
        self._metrics_lock = threading.Lock()
        self._frame_stamps = []
        self._tracker_times = []
        self._output_latencies = []
        self._detector_times = []

    @property
    def tracker_name(self):
        return self.tracking.name

    def start(self):
        if self.is_running:
            return
        self._stop_event.clear()
        self._detection_requests.clear()
        self._last_yolo_confidence = None
        self._core = None
        self._set_result(VisionResult(tracking_state="search"))
        self._thread = threading.Thread(target=self._run, name="drone-tracker", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=3.0)
        detector = self._detector_thread
        if detector is not None and detector is not threading.current_thread():
            detector.join(timeout=3.0)
        self._thread = None
        self._detector_thread = None
        self._release_capture()
        self._detection_requests.clear()
        self._core = None
        self._last_yolo_confidence = None
        self._set_result(VisionResult(tracking_state="search"))

    def _load_tracker(self):
        if self._tracker is not None:
            return
        from visual_tracking.core import load_tracker

        try:
            self._tracker, self._tracker_lost_score = load_tracker(
                self.tracking.library_path,
                self.tracking.name,
                device=self.tracking.device,
                checkpoint=self.tracking.checkpoint_path,
            )
        except SystemExit as exc:
            raise RuntimeError(str(exc)) from exc

    def _make_core(self):
        logic = TargetLogic(
            verify_iou=self.tracking.verify_iou,
            reacquire_gate=self.tracking.reacquire_gate,
            lost_after_s=self.tracking.lost_after_s,
            trust_tracker_s=self.tracking.trust_tracker_s,
            search_growth=self.tracking.search_growth,
            initial_target="best",
        )
        return TrackerCore(
            self._tracker,
            logic,
            lost_score=self._tracker_lost_score,
            max_catch_up=self.tracking.max_catch_up,
            stamp_tolerance=self.tracking.stamp_tolerance,
            buffer_size=self.tracking.buffer_size,
            detector_every=self.tracking.detector_every,
            check_on_low_score=False,
        )

    def _reset_tracking(self):
        with self._core_lock:
            if self._tracker is not None:
                self._core = self._make_core()
        self._detection_requests.clear()
        self._last_yolo_confidence = None

    def _run(self):
        try:
            self._load_model()
        except Exception as exc:
            self._set_result(VisionResult(error=f"Model error: {exc}", tracking_state="search"))
            return
        try:
            self._load_tracker()
        except Exception as exc:
            self._set_result(VisionResult(error=f"Tracker error: {exc}", tracking_state="search"))
            return

        if getattr(self._tracker, "cuda", False):
            import torch

            if torch.cuda.is_available():
                self._torch = torch
                self._detector_stream = torch.cuda.Stream()
        self._core = self._make_core()
        self._detector_thread = threading.Thread(
            target=self._detector_loop,
            name="drone-yolo-verifier",
            daemon=True,
        )
        self._detector_thread.start()
        try:
            while not self._stop_event.is_set():
                if self._capture is None:
                    try:
                        self._capture = self._open_capture()
                    except Exception as exc:
                        self._set_result(VisionResult(error=f"Camera error: {exc}", tracking_state="search"))
                        self._stop_event.wait(1.0)
                        continue

                if self._capture is None or not self._capture.isOpened():
                    self._release_capture()
                    self._set_result(
                        VisionResult(error=f"Cannot open camera {self.camera}", tracking_state="search")
                    )
                    self._stop_event.wait(1.0)
                    continue

                try:
                    ok, frame = self._capture.read()
                    if not ok or frame is None:
                        self._release_capture()
                        self._reset_tracking()
                        self._set_result(
                            VisionResult(error=f"Camera {self.camera} disconnected", tracking_state="search")
                        )
                        self._stop_event.wait(1.0)
                        continue
                    self._track_frame(frame, time.monotonic())
                except Exception as exc:
                    self._release_capture()
                    self._set_result(VisionResult(error=f"Tracking error: {exc}", tracking_state="search"))
                    self._stop_event.set()
        finally:
            self._stop_event.set()
            self._release_capture()

    def _track_frame(self, frame, stamp):
        with self._core_lock:
            send = self._core.pre_image(stamp)
        if send:
            self._detection_requests.put(frame, stamp)

        with self._core_lock:
            tracked = self._core.on_image(frame, stamp, sent=send)
            event = None
            if self._core.last_decision is not None and abs(self._core.last_decision[0] - stamp) <= self.tracking.stamp_tolerance:
                event = self._core.last_decision[1].event
            result = self._make_result(
                frame,
                stamp,
                tracked.box,
                tracked.mode,
                event=event,
            )
        self._set_result(result)
        self._record_frame_metrics(stamp, tracked.trk_ms)

    def _detector_loop(self):
        while not self._stop_event.is_set():
            request = self._detection_requests.get()
            if request is None:
                continue
            frame, stamp = request
            started = time.perf_counter()
            try:
                if self._detector_stream is None:
                    detections = self._detect_drones(frame)
                else:
                    with self._torch.cuda.stream(self._detector_stream):
                        detections = self._detect_drones(frame)
            except Exception as exc:
                self._set_result(VisionResult(error=f"Vision error: {exc}", tracking_state="search"))
                self._stop_event.set()
                return
            detector_ms = (time.perf_counter() - started) * 1000.0
            with self._metrics_lock:
                self._detector_times.append(detector_ms)
                self._detector_times = self._detector_times[-200:]

            core_detections = tuple(
                Detection(
                    (
                        float(detection.bbox[0]),
                        float(detection.bbox[1]),
                        float(detection.bbox[2] - detection.bbox[0]),
                        float(detection.bbox[3] - detection.bbox[1]),
                    ),
                    detection.confidence,
                    detection.label,
                )
                for detection in detections
            )
            with self._core_lock:
                if self._core is None:
                    continue
                decision = self._core.on_detections(core_detections, stamp)
                if decision is None or not self._core.buf:
                    continue
                if decision.event in ("init", "confirm", "correct", "reacquire") and detections:
                    chosen = min(
                        detections,
                        key=lambda detection: math.dist(
                            detection_center(detection),
                            (
                                self._core.logic.last_conf[0] + self._core.logic.last_conf[2] / 2.0,
                                self._core.logic.last_conf[1] + self._core.logic.last_conf[3] / 2.0,
                            ),
                        ),
                    )
                    self._last_yolo_confidence = chosen.confidence
                latest = self._core.buf[-1]
                publish = (
                    abs(latest.stamp - stamp) <= self.tracking.stamp_tolerance
                    or decision.event in ("init", "correct", "reacquire", "lost")
                )
                if publish:
                    same_frame = abs(latest.stamp - stamp) <= self.tracking.stamp_tolerance
                    result = self._make_result(
                        latest.frame,
                        latest.stamp,
                        latest.box if self._core.logic.mode == "track" else None,
                        self._core.logic.mode,
                        detections=detections if same_frame else (),
                        event=decision.event,
                    )
                else:
                    result = None
            if result is not None:
                self._set_result(result)

            remaining = self.inference_interval - detector_ms / 1000.0
            if remaining > 0:
                self._stop_event.wait(remaining)

    def _make_result(self, frame, stamp, box, state, detections=(), event=None):
        target = None
        direction = None
        if state == "track" and box is not None:
            x, y, width, height = box
            if all(math.isfinite(value) for value in box) and width > 0.0 and height > 0.0:
                x1 = max(0, min(frame.shape[1] - 1, int(round(x))))
                y1 = max(0, min(frame.shape[0] - 1, int(round(y))))
                x2 = max(0, min(frame.shape[1], int(round(x + width))))
                y2 = max(0, min(frame.shape[0], int(round(y + height))))
                if x2 > x1 and y2 > y1:
                    target = VisionDetection(
                        bbox=(x1, y1, x2, y2),
                        confidence=self._last_yolo_confidence or 0.0,
                        class_id=0,
                        label="drone",
                    )
                    if self.horizontal_fov_deg is not None:
                        u, v = detection_center(target)
                        direction = pixel_to_direction(
                            u,
                            v,
                            frame.shape[1],
                            frame.shape[0],
                            self.horizontal_fov_deg,
                            self.orientation_deg,
                        )

        annotated = frame.copy()
        for detection in detections:
            if target is not None and detection.bbox == target.bbox:
                continue
            self._draw_box(
                annotated,
                detection,
                (65, 179, 227),
                f"YOLO {detection.confidence:.0%}",
            )
        if target is not None:
            label = "DRONE TRACKING"
            if self._last_yolo_confidence is not None:
                label += f" | LAST YOLO {self._last_yolo_confidence:.0%}"
            if direction is not None:
                label += f" | {direction.global_bearing_deg:.1f} deg"
            self._draw_box(annotated, target, (68, 68, 255), label)

        return VisionResult(
            frame=annotated,
            detections=tuple(detections),
            target=target,
            direction=direction,
            timestamp=stamp,
            tracking_state=state,
            tracking_event=event,
            output_source="tracker" if target is not None else "yolo",
            last_yolo_confidence=self._last_yolo_confidence,
        )

    @staticmethod
    def _draw_box(frame, detection, color, label):
        x1, y1, x2, y2 = detection.bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            label,
            (x1, max(18, y1 - 7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )

    def _record_frame_metrics(self, stamp, tracker_ms):
        now = time.monotonic()
        with self._metrics_lock:
            self._frame_stamps.append(now)
            self._frame_stamps = self._frame_stamps[-200:]
            self._output_latencies.append((now - stamp) * 1000.0)
            self._output_latencies = self._output_latencies[-200:]
            if tracker_ms is not None:
                self._tracker_times.append(tracker_ms)
                self._tracker_times = self._tracker_times[-200:]

    @staticmethod
    def _percentile(values, percentile):
        return float(np.percentile(values, percentile)) if values else 0.0

    def get_metrics(self):
        with self._metrics_lock:
            frame_stamps = list(self._frame_stamps)
            tracker_times = list(self._tracker_times)
            output_latencies = list(self._output_latencies)
            detector_times = list(self._detector_times)
        duration = frame_stamps[-1] - frame_stamps[0] if len(frame_stamps) > 1 else 0.0
        with self._core_lock:
            core = self._core
            counts = dict(core.logic.counts) if core is not None else {}
            stats = dict(core.stats) if core is not None else {}
            state = core.logic.mode if core is not None else "search"
            buffered_frames = len(core.buf) if core is not None else 0
        return {
            "state": state,
            "camera_fps": (len(frame_stamps) - 1) / duration if duration > 0.0 else 0.0,
            "tracker_ms_mean": float(np.mean(tracker_times)) if tracker_times else 0.0,
            "tracker_ms_p95": self._percentile(tracker_times, 95),
            "output_ms_p95": self._percentile(output_latencies, 95),
            "detector_ms_mean": float(np.mean(detector_times)) if detector_times else 0.0,
            "detector_ms_p95": self._percentile(detector_times, 95),
            "detector_queue_size": self._detection_requests.size,
            "detector_queue_dropped": self._detection_requests.dropped,
            "buffered_frames": buffered_frames,
            "counts": counts,
            "stats": stats,
        }
