# Frequency-Locked 3-Array Acoustic Triangulation System

This package provides real-time acoustic frequency detection and 2D spatial triangulation tailored for a custom multi-array rig:
- **Apex / Front**: miniDSP UMA-8 circular array (7 active microphones, 8.55 cm radius)
- **Left Flank**: miniDSP UMA-16 array (16 microphones on 42 mm grid, offset 14 cm from UMA-8)
- **Right Flank**: miniDSP UMA-16 array (16 microphones on 42 mm grid, offset 17 cm from UMA-8)
- **Rear Baseline**: 18.5 cm between UMA-16 tails

```
                        [ UMA-8 Apex ]
                            /      \
                 14 cm     /        \    17 cm
                          /          \
                         /            \
           [ Left UMA-16 ]            [ Right UMA-16 ]
             (Black USB)                (White USB + Tape)
                         \            /
                          \          /
                           \ 18.5 cm/
                            <------>
```

---

## Quick Start

### 1. Launch the 2D Tactical Radar GUI
```bash
# With physical hardware connected:
python3 main.py gui

# In simulation mode (no hardware required):
python3 main.py gui --simulate
```

### 2. Headless Terminal Tracking (CLI)
```bash
# Monitor drone whine between 900 and 1100 Hz:
python3 main.py cli --low 900 --high 1100 --threshold -65
```

### 3. Verify Hardware & Calibration
```bash
# List all connected USB audio inputs:
python3 main.py list-devices

# Verify geometry coordinates & aperture checks:
python3 main.py test-geometry

# Run offline math benchmark:
python3 main.py simulate --target-x 0.6 --target-y 1.5
```

---

## File Architecture

* [`rig_config.json`](file:///home/dhia/Desktop/projects/freq_project/frequency_id/rig_config.json): Physical rig geometry, mic coordinates, offsets, and rotation angles.
* [`geometry.py`](file:///home/dhia/Desktop/projects/freq_project/frequency_id/geometry.py): Mathematical transformation between local array frames and the global forward (+Y) coordinate system.
* [`detector.py`](file:///home/dhia/Desktop/projects/freq_project/frequency_id/detector.py): FFT analyzer with sub-bin parabolic peak interpolation and temporal noise floor depression.
* [`doa_engine.py`](file:///home/dhia/Desktop/projects/freq_project/frequency_id/doa_engine.py): Frequency-gated narrowband SRP-PHAT direction finder with wrap-around safe circular tracking.
* [`triangulation.py`](file:///home/dhia/Desktop/projects/freq_project/frequency_id/triangulation.py): Weighted least-squares ray intersection solver computing (X, Y) coordinates, range, bearing, and residual error.
* [`audio_manager.py`](file:///home/dhia/Desktop/projects/freq_project/frequency_id/audio_manager.py): Multi-stream USB audio capture coordinator and synthetic wavefield simulator.
* [`tactical_gui.py`](file:///home/dhia/Desktop/projects/freq_project/frequency_id/tactical_gui.py): 2D Tactical Map and real-time power spectrum interface.
* [`main.py`](file:///home/dhia/Desktop/projects/freq_project/frequency_id/main.py): Unified CLI and launcher.
