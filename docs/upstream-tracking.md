# Vendored DyHiT tracker provenance

- Integration source: https://github.com/mariemchikhaouiakkurad-web/drone_tracking_project/tree/main/yolo_pipeline/drone_target_tracker
- Pinned commit: `871c3bb014a1bb96d9a93b6d50f22893b2b72f40`
- Vendored integration files: `src/frequency_id/vision/core.py`, `src/frequency_id/vision/logic.py`, and their upstream tests under `tests/unit/vision/` (`test_logic.py`, `test_low_latency.py`).
- Tracker source: DyHiT from https://github.com/kangben258/HiT, vendored under `visual_tracking/tracker_lib/external/HiT/` with its MIT license.
- The upstream `drone_target_tracker` package (v0.1.0, maintainer Ahmed Zghibi) declares the MIT license in its ROS2 `package.xml`; upstream notices are preserved here while the unused ROS2 `ament_python` manifest has been removed.
- ROS, Isaac ROS, SUTrack, HiT-Base, unrelated launch/configuration files, and unrelated model downloads are intentionally excluded.

## Local integration changes

- Tracker loading accepts explicit device and checkpoint overrides.
- Skipped catch-up clears stale buffered output instead of publishing an old box on a newer frame.
- GPU synchronization waits for the tracker's current stream; `src/frequency_id/vision/tracked_camera.py` runs YOLO on a separate CUDA stream.
- The asynchronous camera workers, latest-request queue, drone-class conversion, FOV directions, metrics, and GUI/CLI states are project integration code.

## DyHiT checkpoint

- File: `visual_tracking/tracker_lib/trackers/dyhit/models/DyHiT_ep0060.pth.tar`
- Upstream Google Drive file ID: `16zfFdgA77KScEfPrEsCKja6VINVo_8nD`
- Size: `182028385` bytes
- SHA-256: `8bd307baf7ecd10b930a9d2a7288bb74f87c03dc6dd984356f7be9a458b27f61`
- Download command: `python3 visual_tracking/tracker_lib/trackers/dyhit/download_model.py`

DyHiT's score is not a drone detection probability. YOLO remains responsible for semantic drone confirmation.
