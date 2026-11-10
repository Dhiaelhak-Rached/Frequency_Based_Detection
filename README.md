# Frequency-Locked Dual-Array Acoustic Triangulation System

This package provides real-time acoustic frequency detection and 2D spatial triangulation using:
- **Apex / Front**: upright miniDSP UMA-8 circular array (7 active microphones), facing global forward (0°)
- **Left Flank**: full upright miniDSP UMA-16 v2 array (16 microphones on a 42 mm grid), facing global left (270°)
- **XMOS-board separation**: 36.5 cm between the UMA-8 and left UMA-16

```
                        [ UMA-8 Apex ]
                              /
                       36.5 cm /
                            /
             [ Left UMA-16 ]
               (Black USB)
```

The distance is measured between XMOS boards. The configurable
`mic_center_offset_local_m` value translates the UMA-16 XMOS reference point to
its acoustic array center; it currently defaults to zero until that offset is measured.

Both arrays use `"mount": "vertical"`. For a vertical array, `orientation_deg`
is the outward acoustic-face normal, local +90° is the viewer's right on that face,
and only horizontal microphone displacement contributes to azimuth steering. The
scan is restricted to ±90° around the normal because a vertical array has front-back
ambiguity. Weak UMA-16 microphones can be excluded with `active_channels`.

High-prominence bearings use the existing acceptance cutoffs. Vertical arrays also
accept low-prominence bearings after at least three consistent samples in a five-sample
history with circular confidence of at least 0.85. Raw prominence is still reported
unchanged and used for triangulation weighting.

---

## Quick Start

Install the Python dependencies (Tkinter is provided by the system Python):

```bash
python3 -m pip install -r requirements.txt
```

### 1. Launch the Tactical Radar GUI
```bash
# With physical hardware connected:
python3 main.py gui

# In simulation mode (no hardware required):
python3 main.py gui --simulate

# Require confirmed drone classification before declaring a target lock:
python3 main.py gui --simulate --require-drone

# Use a custom ONNX model or disable classification:
python3 main.py gui --model /path/to/mobilenetv2_drone.onnx
python3 main.py gui --no-classifier

# One array only (direction without range or coordinates):
python3 main.py gui --uma8-only
python3 main.py gui --uma16-only

# Only the UMA-16, positioned at the front origin and facing forward:
python3 main.py gui --uma16-front-only

# Select a camera by index or stable Linux device path:
python3 main.py gui --camera 0
python3 main.py gui --camera /dev/v4l/by-id/your-camera-path

# Supply this camera's measured horizontal FOV to enable direction:
python3 main.py gui --camera 0 --camera-fov 75

# Use a different YOLO checkpoint or disable video detection:
python3 main.py gui --yolo-model /path/to/best.pt
python3 main.py gui --no-vision
```

The GUI provides mutually exclusive **UMA-8 Only**, **UMA-16 Only**, and
**UMA-16 Front Only** checkboxes. Front-only UMA-16 mode places the acoustic
center at `(0, 0)` with a 0° forward orientation and continues to use the UMA-16
device setting (`--dev-left`). Switching modes while listening restarts capture
with only the selected arrays and restores the normal left geometry when needed.
The classification tile shows live drone probability and voting state only when
the target-band frequency detector has a valid detection. Otherwise it shows
**NO TARGET SIGNAL** and clears old voting history. The **Require Drone ID for Lock**
checkbox gates target-lock status on confirmed audio classification; acoustic bearings
and coordinates remain visible while awaiting ID.

Use the **Camera Detection** checkbox, editable camera drop-down, and **Refresh**
button to select video hardware at runtime. The selected source is saved in
`rig_config.json`. The camera panel draws YOLO boxes and confidence, while the
**CAMERA** tile independently reports detection and acquisition state.

**No FOV is assumed.** Enter the selected camera's horizontal field of view in
**HFOV °** and click **Apply FOV**. Each camera's value is stored separately in
`vision.camera_fovs`; switching cameras loads its own value. Blank or missing FOV
keeps detection active but reports `DIRECTION UNCALIBRATED` instead of estimating
an angle. Valid values are greater than 0° and less than 180°.

After three consistent detections, the confirmed box and camera tile show global
bearing, and a dashed purple **CAMERA** ray appears on the tactical map. Bearing is
clockwise from rig forward (0°), with right positive. Elevation is positive above
the camera horizon. Set `vision.orientation_deg` if the camera does not face rig
forward. Camera direction never creates a position, range, or acoustic lock.

To calibrate FOV, place a small object at a measured left/right angle from the
camera, adjust HFOV until the reported local bearing matches, and repeat on the
other side. A manufacturer specification is a useful starting point, but a measured
value is preferred. Recalibrate when changing to a different camera or lens.

### 2. Headless Terminal Tracking (CLI)
```bash
# Monitor audio and camera 0, using its measured 75-degree horizontal FOV:
python3 main.py cli --camera 0 --camera-fov 75 --low 900 --high 1100 --threshold -65

# Require both triangulation and confirmed audio classification for an acoustic lock:
python3 main.py cli --simulate --require-drone --low 900 --high 1100

# One array only:
python3 main.py cli --uma8-only --low 900 --high 1100
python3 main.py cli --uma16-only --low 900 --high 1100
python3 main.py cli --uma16-front-only --low 900 --high 1100

# Pure spatial tracking without the ONNX classifier or camera:
python3 main.py cli --no-classifier --no-vision --low 900 --high 1100
```

### 3. Verify Hardware, Classification & Calibration
```bash
# List connected USB audio inputs and usable cameras:
python3 main.py list-devices
python3 main.py list-cameras

# Verify geometry coordinates & aperture checks:
python3 main.py test-geometry

# Run audio, geometry, and mocked camera worker tests:
python3 -m unittest test_classifier test_vertical_uma16 test_vision

# Run offline classification and triangulation benchmark:
python3 main.py simulate --target-x -0.6 --target-y 1.5
```

---

## File Architecture

* [`rig_config.json`](file:///home/dhia/Desktop/projects/freq_project/frequency_id/rig_config.json): Physical rig, audio classifier, camera selection, and detection settings.
* [`geometry.py`](file:///home/dhia/Desktop/projects/freq_project/frequency_id/geometry.py): Mathematical transformation between local array frames and the global forward (+Y) coordinate system.
* [`detector.py`](file:///home/dhia/Desktop/projects/freq_project/frequency_id/detector.py): FFT analyzer with sub-bin parabolic peak interpolation and temporal noise floor depression.
* [`doa_engine.py`](file:///home/dhia/Desktop/projects/freq_project/frequency_id/doa_engine.py): Frequency-gated narrowband SRP-PHAT direction finder with wrap-around safe circular tracking.
* [`triangulation.py`](file:///home/dhia/Desktop/projects/freq_project/frequency_id/triangulation.py): Weighted least-squares ray intersection solver computing (X, Y) coordinates, range, bearing, and residual error.
* [`audio_manager.py`](file:///home/dhia/Desktop/projects/freq_project/frequency_id/audio_manager.py): Multi-stream USB audio capture coordinator and synthetic drone wavefield simulator.
* [`classifier.py`](file:///home/dhia/Desktop/projects/freq_project/frequency_id/classifier.py): MobileNetV2 ONNX inference, resampling, audio buffering, and temporal drone voting.
* [`vision_detector.py`](file:///home/dhia/Desktop/projects/freq_project/frequency_id/vision_detector.py): Selectable OpenCV capture and background Ultralytics YOLO drone detection.
* [`tactical_gui.py`](file:///home/dhia/Desktop/projects/freq_project/frequency_id/tactical_gui.py): Tactical map, spectrum, camera preview, classification telemetry, and lock controls.
* [`main.py`](file:///home/dhia/Desktop/projects/freq_project/frequency_id/main.py): Unified CLI, audio/vision configuration, device listing, and launcher.
* [`test_classifier.py`](file:///home/dhia/Desktop/projects/freq_project/frequency_id/test_classifier.py): ONNX reference, resampling, voting, and end-to-end classification/DOA tests.
* [`test_vision.py`](file:///home/dhia/Desktop/projects/freq_project/frequency_id/test_vision.py): Mocked YOLO filtering, camera worker, retry, and shutdown tests.
* [`model/`](file:///home/dhia/Desktop/projects/freq_project/frequency_id/model): MobileNetV2 ONNX assets and YOLO `best.pt` checkpoint.
