from dataclasses import dataclass
import math
from pathlib import Path
import re
import threading
import time
from typing import Optional, Sequence, Tuple, Union

import cv2
import numpy as np


CameraSource = Union[int, str]


@dataclass(frozen=True)
class CameraDevice:
    index: int
    path: str
    stable_path: Optional[str] = None

    @property
    def source(self) -> str:
        return self.stable_path or self.path


@dataclass(frozen=True)
class VisionDetection:
    bbox: Tuple[int, int, int, int]
    confidence: float
    class_id: int
    label: str


@dataclass(frozen=True)
class CameraDirection:
    local_bearing_deg: float
    global_bearing_deg: float
    elevation_deg: float
    ray: Tuple[float, float, float]


@dataclass(frozen=True)
class VisionResult:
    frame: Optional[np.ndarray] = None
    detections: Tuple[VisionDetection, ...] = ()
    target: Optional[VisionDetection] = None
    direction: Optional[CameraDirection] = None
    timestamp: float = 0.0
    error: Optional[str] = None

    @property
    def best_confidence(self) -> float:
        if self.target is not None:
            return self.target.confidence
        if not self.detections:
            return 0.0
        return max(detection.confidence for detection in self.detections)


def validate_horizontal_fov(value) -> Optional[float]:
    if value is None or isinstance(value, str) and not value.strip():
        return None
    fov = float(value)
    if not math.isfinite(fov) or not 0.0 < fov < 180.0:
        raise ValueError("Horizontal FOV must be between 0 and 180 degrees")
    return fov


def detection_center(detection: VisionDetection) -> Tuple[float, float]:
    x1, y1, x2, y2 = detection.bbox
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def pixel_to_direction(
    u: float,
    v: float,
    image_width: int,
    image_height: int,
    horizontal_fov_deg,
    orientation_deg: float = 0.0,
) -> CameraDirection:
    fov = validate_horizontal_fov(horizontal_fov_deg)
    if fov is None:
        raise ValueError("Horizontal FOV is required for camera direction")
    if image_width <= 0 or image_height <= 0:
        raise ValueError("Image dimensions must be positive")
    fx = image_width / (2.0 * math.tan(math.radians(fov) / 2.0))
    cx = (image_width - 1.0) / 2.0
    cy = (image_height - 1.0) / 2.0
    right = (float(u) - cx) / fx
    up = -(float(v) - cy) / fx
    norm = math.sqrt(right * right + up * up + 1.0)
    local_bearing = math.degrees(math.atan(right))
    elevation = math.degrees(math.atan2(up, math.sqrt(1.0 + right * right)))
    return CameraDirection(
        local_bearing_deg=local_bearing,
        global_bearing_deg=(local_bearing + float(orientation_deg)) % 360.0,
        elevation_deg=elevation,
        ray=(right / norm, up / norm, 1.0 / norm),
    )


class DetectionLock:
    def __init__(
        self,
        max_step_px: float = 120.0,
        confirmation_frames: int = 3,
        lost_timeout_s: float = 0.75,
    ):
        self.max_step_px = float(max_step_px)
        self.confirmation_frames = int(confirmation_frames)
        self.lost_timeout_s = float(lost_timeout_s)
        self.center = None
        self.last_seen = None
        self.pending_center = None
        self.pending_count = 0
        self.pending_stamp = None

    def reset(self):
        self.center = None
        self.last_seen = None
        self.pending_center = None
        self.pending_count = 0
        self.pending_stamp = None

    def _near(self, first, second):
        return math.dist(first, second) <= self.max_step_px

    def select(
        self,
        detections: Sequence[VisionDetection],
        stamp_s: float,
    ) -> Optional[VisionDetection]:
        if self.last_seen is not None and stamp_s - self.last_seen > self.lost_timeout_s:
            self.center = None
        if not detections:
            self.pending_center = None
            self.pending_count = 0
            self.pending_stamp = None
            return None

        if self.center is not None:
            nearby = [
                detection
                for detection in detections
                if self._near(self.center, detection_center(detection))
            ]
            if nearby:
                chosen = max(nearby, key=lambda detection: detection.confidence)
                self.center = detection_center(chosen)
                self.last_seen = stamp_s
                self.pending_center = None
                self.pending_count = 0
                self.pending_stamp = None
                return chosen

        chosen = max(detections, key=lambda detection: detection.confidence)
        candidate = detection_center(chosen)
        consecutive = (
            self.pending_center is not None
            and self._near(self.pending_center, candidate)
            and self.pending_stamp is not None
            and 0.0 <= stamp_s - self.pending_stamp <= 0.2
        )
        self.pending_count = self.pending_count + 1 if consecutive else 1
        self.pending_center = candidate
        self.pending_stamp = stamp_s
        if self.pending_count < self.confirmation_frames:
            return None
        if self.center is not None and self.last_seen is not None and (
            stamp_s - self.last_seen < self.lost_timeout_s
        ):
            return None

        self.center = candidate
        self.last_seen = stamp_s
        self.pending_center = None
        self.pending_count = 0
        self.pending_stamp = None
        return chosen


def normalize_camera_source(source: CameraSource) -> CameraSource:
    if isinstance(source, int):
        return source
    source = str(source).strip()
    if re.fullmatch(r"\d+", source):
        return int(source)
    return source


def discover_cameras(max_devices: int = 16) -> Tuple[CameraDevice, ...]:
    devices = []
    stable_paths = {}
    by_id = Path("/dev/v4l/by-id")
    if by_id.is_dir():
        for path in by_id.iterdir():
            try:
                stable_paths[str(path.resolve())] = str(path)
            except OSError:
                pass

    for index in range(max_devices):
        path = f"/dev/video{index}"
        if not Path(path).exists():
            continue
        capture = cv2.VideoCapture(index, cv2.CAP_V4L2)
        try:
            if capture.isOpened():
                devices.append(CameraDevice(index, path, stable_paths.get(path)))
        finally:
            capture.release()
    return tuple(devices)


class DroneCamera:
    def __init__(
        self,
        model_path: Union[str, Path],
        camera: CameraSource = 0,
        confidence: float = 0.5,
        inference_hz: float = 10.0,
        horizontal_fov_deg=None,
        orientation_deg: float = 0.0,
        max_target_step_px: float = 120.0,
        confirmation_frames: int = 3,
        lost_timeout_s: float = 0.75,
        model=None,
        capture_factory=None,
    ):
        self.model_path = Path(model_path)
        self.camera = normalize_camera_source(camera)
        self.confidence = float(confidence)
        self.inference_interval = 1.0 / max(0.1, float(inference_hz))
        self.horizontal_fov_deg = validate_horizontal_fov(horizontal_fov_deg)
        self.orientation_deg = float(orientation_deg)
        self.target_lock = DetectionLock(
            max_step_px=max_target_step_px,
            confirmation_frames=confirmation_frames,
            lost_timeout_s=lost_timeout_s,
        )
        self._model = model
        self._capture_factory = capture_factory
        self._capture = None
        self._thread = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._result = VisionResult()
        if self._model is None and not self.model_path.is_file():
            raise FileNotFoundError(f"YOLO model not found: {self.model_path}")

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self):
        if self.is_running:
            return
        self.target_lock.reset()
        self._stop_event.clear()
        self._set_result(VisionResult())
        self._thread = threading.Thread(target=self._run, name="drone-camera", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=3.0)
        self._thread = None
        self._release_capture()
        self.target_lock.reset()
        self._set_result(VisionResult())

    def set_camera(self, camera: CameraSource):
        was_running = self.is_running
        self.stop()
        self.camera = normalize_camera_source(camera)
        self._set_result(VisionResult())
        if was_running:
            self.start()

    def set_horizontal_fov(self, horizontal_fov_deg):
        self.horizontal_fov_deg = validate_horizontal_fov(horizontal_fov_deg)

    def get_result(self) -> VisionResult:
        with self._lock:
            return self._result

    def process_frame(self, frame: np.ndarray) -> VisionResult:
        self._load_model()
        names = self._model.names
        drone_class_ids = {
            int(class_id)
            for class_id, name in names.items()
            if str(name).strip().lower() == "drone"
        }
        if not drone_class_ids:
            raise ValueError(f"YOLO checkpoint has no 'drone' class: {names}")

        predictions = self._model.predict(
            frame,
            conf=self.confidence,
            verbose=False,
        )
        detections = []
        if predictions:
            boxes = predictions[0].boxes
            if boxes is not None:
                coordinates = boxes.xyxy.cpu().numpy()
                confidences = boxes.conf.cpu().numpy()
                class_ids = boxes.cls.cpu().numpy()
                for xyxy, score, class_id in zip(coordinates, confidences, class_ids):
                    class_id = int(class_id)
                    if class_id not in drone_class_ids:
                        continue
                    detections.append(
                        VisionDetection(
                            bbox=tuple(int(round(value)) for value in xyxy),
                            confidence=float(score),
                            class_id=class_id,
                            label=str(names[class_id]),
                        )
                    )

        detections = tuple(detections)
        timestamp = time.monotonic()
        target = self.target_lock.select(detections, timestamp)
        direction = None
        if target is not None and self.horizontal_fov_deg is not None:
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
            confirmed = detection is target
            color = (68, 68, 255) if confirmed else (65, 179, 227)
            label = f"DRONE {detection.confidence:.0%}"
            if confirmed and direction is not None:
                label += f" {direction.global_bearing_deg:.1f} deg"
            elif confirmed:
                label += " NO FOV"
            else:
                label += " ACQUIRING"
            x1, y1, x2, y2 = detection.bbox
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                annotated,
                label,
                (x1, max(18, y1 - 7)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA,
            )
        return VisionResult(
            frame=annotated,
            detections=detections,
            target=target,
            direction=direction,
            timestamp=timestamp,
        )

    def _set_result(self, result: VisionResult):
        with self._lock:
            self._result = result

    def _open_capture(self):
        if self._capture_factory is not None:
            return self._capture_factory(self.camera)
        if isinstance(self.camera, str) and self.camera.startswith("/dev/"):
            return cv2.VideoCapture(self.camera, cv2.CAP_V4L2)
        return cv2.VideoCapture(self.camera)

    def _release_capture(self):
        capture = self._capture
        self._capture = None
        if capture is not None:
            capture.release()

    def _load_model(self):
        if self._model is None:
            from ultralytics import YOLO

            self._model = YOLO(str(self.model_path))

    def _run(self):
        try:
            self._load_model()
        except Exception as exc:
            self._set_result(VisionResult(error=f"Model error: {exc}"))
            return

        while not self._stop_event.is_set():
            started = time.monotonic()
            if self._capture is None:
                try:
                    self._capture = self._open_capture()
                except Exception as exc:
                    self._set_result(VisionResult(error=f"Camera error: {exc}"))
                    self._stop_event.wait(1.0)
                    continue

            if self._capture is None or not self._capture.isOpened():
                self._release_capture()
                self._set_result(VisionResult(error=f"Cannot open camera {self.camera}"))
                self._stop_event.wait(1.0)
                continue

            try:
                ok, frame = self._capture.read()
                if not ok or frame is None:
                    self._release_capture()
                    self._set_result(VisionResult(error=f"Camera {self.camera} disconnected"))
                    self._stop_event.wait(1.0)
                    continue
                self._set_result(self.process_frame(frame))
            except Exception as exc:
                self._release_capture()
                self._set_result(VisionResult(error=f"Vision error: {exc}"))
                self._stop_event.wait(1.0)
                continue

            remaining = self.inference_interval - (time.monotonic() - started)
            if remaining > 0:
                self._stop_event.wait(remaining)

        self._release_capture()
