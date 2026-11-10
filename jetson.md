# AI Agent Runbook: Jetson Orin Deployment Through SSH

## Agent objective

You are the AI deployment agent. Deploy and verify the complete acoustic classification and triangulation system on the remote Jetson Orin through SSH. Work through this document sequentially and do not stop at copying the files: install dependencies, run the tests, configure both USB arrays, and verify the running application.

The current classifier explicitly uses `CPUExecutionProvider` (`classifier.py:84-91`). Use CPU ONNX inference for this deployment. Do not attempt CUDA/TensorRT optimization during this task.

## Mandatory execution rules for the agent

1. Run all installation, configuration, and test commands on the remote Jetson through the SSH connection in step 0. Do not accidentally install Jetson dependencies or run the remote application on the local PC.
2. The local PC is only the source of the project files and the SSH/rsync client. The local project root is `/home/dhia/Desktop/projects/freq_project/frequency_id`.
3. Use the supplied credentials for non-interactive SSH as shown below. If SSH, sudo, or hardware access fails, report the exact error; do not claim the deployment succeeded.
4. Execute the steps in order. After each step, inspect the command output and proceed only when that step passed. Do not hide errors, skip failing tests, or weaken test thresholds.
5. Do not copy a local virtual environment or x86 Python wheels to the Jetson. Create a fresh ARM64 environment remotely.
6. Do not change application code or physical geometry unless a concrete Jetson compatibility problem requires it. Audio device indices are expected to change.
7. A plain SSH session does not prove that the desktop GUI is usable. Complete every headless check remotely; report GUI verification as pending unless a working Jetson desktop/display is available.
8. Finish with a concise report listing completed steps, test results, selected audio device indices, remaining problems, and any manual display/hardware checks still required.

## Approach

Create a clean Python virtual environment on the Jetson, copy the complete project including model assets, and validate in stages: connection → dependencies → offline tests → USB audio → single arrays → dual array → GUI/display where available → stability check.

## Deployment steps

### 0. Connect to the Jetson

Use these SSH connection details:

```text
Host: 41.228.129.118
Port: 2222
Username: meltem
Password: meltem123
Remote project directory: /home/meltem/frequency_id
```

Connect from the PC:

```bash
ssh -p 2222 meltem@41.228.129.118
```

Enter the password above when prompted. Run every Jetson-side command in this guide through that SSH session. For non-interactive agent commands on a Linux PC, install `sshpass` if needed and use:

```bash
SSHPASS='meltem123' sshpass -e ssh -p 2222 meltem@41.228.129.118 "<remote-command>"
```

This file contains a plaintext password; do not publish or commit it to a public repository.

### 1. Check the Jetson baseline

On the Jetson, confirm its operating system and Python details:

```bash
uname -m
cat /etc/nv_tegra_release
python3 --version
```

Expected architecture: `aarch64`. Use the JetPack/L4T and Python versions from these commands when selecting the ONNX Runtime wheel.

Use a local monitor/desktop for GUI testing. SSH is suitable for installation and headless CLI testing.

### 2. Copy the complete project

From the PC, copy the whole directory to the Jetson home directory. For non-interactive agent use with `sshpass`:

```bash
SSHPASS='meltem123' sshpass -e rsync -av \
  -e "ssh -p 2222" \
  --exclude '.git' \
  --exclude '__pycache__' \
  --exclude '*.pyc' \
  --exclude '.venv' \
  /home/dhia/Desktop/projects/freq_project/frequency_id/ \
  meltem@41.228.129.118:/home/meltem/frequency_id/
```

Do not rely on a partial source-only copy. These files must exist on the Jetson:

```text
model/mobilenetv2_drone.onnx
model/metadata.json
model/reference_outputs.npz
classifier.py
rig_config.json
```

### 3. Install system and Python dependencies

On the Jetson:

```bash
sudo apt update
sudo apt install -y python3-venv python3-tk portaudio19-dev libportaudio2

cd ~/frequency_id
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install numpy scipy sounddevice
```

Install an **ARM64 ONNX Runtime wheel compatible with the Jetson's JetPack/L4T and Python versions**, preferably from NVIDIA's Jetson resources/Jetson Zoo:

```bash
python -m pip install /path/to/the/matching/onnxruntime-wheel.whl
```

If the normal `python -m pip install onnxruntime` offers a compatible ARM64 wheel for that Python version, it can be used instead. Do not copy an x86 wheel from the PC.

Verify all imports and CPU inference availability:

```bash
python -c "import numpy, scipy, sounddevice, tkinter, onnxruntime as ort; print('ONNX Runtime:', ort.__version__); print(ort.get_available_providers())"
```

`CPUExecutionProvider` must appear in the provider list.

### 4. Verify model integrity and offline operation

With the virtual environment active:

```bash
cd ~/frequency_id
sha256sum model/mobilenetv2_drone.onnx model/reference_outputs.npz
```

Compare the hashes with `model/metadata.json`, then run the headless checks:

```bash
python test_classifier.py
python test_vertical_uma16.py
python main.py test-geometry
python main.py simulate --target-x -0.6 --target-y 1.5
```

Expected results:

- All 12 automated tests pass.
- ONNX maximum probability error is below `0.001`.
- Geometry reports 23 active microphones and 36.5 cm board separation.
- Simulation confirms `DRONE` and reaches `TARGET LOCKED - DRONE CONFIRMED`.

Only if the agent has confirmed access to a working Jetson desktop/display, also run:

```bash
python main.py gui --simulate --require-drone
```

Verify that the simulated GUI shows classification, bearing, coordinates, and range. Otherwise, record GUI verification as pending rather than reporting a display-related failure as an application failure.

Do not connect/troubleshoot hardware until these offline checks pass.

### 5. Connect and identify the microphone arrays

Keep the physical rig unchanged: both arrays vertical, UMA-8 facing forward, UMA-16 facing left, with the measured 36.5 cm XMOS-board separation.

Connect both USB arrays directly to the Jetson or through a reliable powered USB hub. Then run:

```bash
python main.py list-devices
```

Identify:

- Front miniDSP UMA-8/micArray: at least 7 input channels.
- Left MCHStreamer/UMA-16: at least 16 input channels.
- Both must support the configured 44,100 Hz sample rate.

Update `rig_config.json` with the Jetson's actual `device_index` values. The PC values (`front: 4`, `left: 1`) must not be assumed to work on the Jetson. Recheck indices after reconnecting devices or rebooting if they change.

### 6. Test each array separately

Substitute the indices shown by `list-devices`:

```bash
python main.py cli --uma8-only --dev-front <front-index> --low 900 --high 1100
python main.py cli --uma16-only --dev-left <left-index> --low 900 --high 1100
python main.py cli --uma16-front-only --dev-left <left-index> --low 900 --high 1100
```

For each array, confirm:

- The correct USB device opens without PortAudio errors.
- Frequency and direction update.
- Classification progresses from `BUFFERING` to a live percentage and vote count.
- Single-array mode displays direction only, without range or coordinates.
- UMA-16 front-only mode uses the UMA-16 device selected by `--dev-left` and reports bearings relative to its 0° forward-facing orientation.

### 7. Test the complete dual-array system

Run the headless tracker first:

```bash
python main.py cli \
  --dev-front <front-index> \
  --dev-left <left-index> \
  --low 900 --high 1100
```

Place a drone recording/speaker or test source in the arrays' shared forward-left field of view. Confirm bearing, classification, coordinates, and range. Then test strict locking:

```bash
python main.py cli \
  --dev-front <front-index> \
  --dev-left <left-index> \
  --require-drone --low 900 --high 1100
```

The CLI trackers run continuously by design. During automated verification, the agent should use a controlled timeout or send SIGINT after collecting enough output, then inspect the log. A deliberate stop after successful output is not a test failure.

Only if a working Jetson desktop/display is available, launch the GUI:

```bash
python main.py gui --dev-front <front-index> --dev-left <left-index>
python main.py gui --uma16-front-only --dev-left <left-index>
```

Verify the classification tile, vote counter, tactical marker, **Require Drone ID for Lock** checkbox, and **UMA-16 Front Only** mode. Confirm that front-only mode shows direction without range/coordinates and that switching back to normal UMA-16 mode restores the left-facing geometry. If no display is available through SSH, leave this as a clearly reported manual follow-up.

### 8. Perform a stability check before deployment use

Run the complete dual-array CLI or GUI for at least 15 minutes with a known sound source. Confirm continuous classification/location updates, no audio-stream failures, no growing lag, and acceptable CPU usage.

Only after manual operation is stable should autostart/systemd setup or CUDA/TensorRT optimization be considered. Any future inference-provider change must rerun `test_classifier.py` and the simulation benchmark.

## Verification / acceptance checklist

The Jetson setup is ready when:

- Offline classifier and vertical-array tests all pass.
- Both USB arrays open at 44.1 kHz with their required channel counts.
- UMA-8-only and UMA-16-only modes work independently.
- UMA-16 front-only mode reports forward-referenced classification and bearing, uses `--dev-left`, and shows no range or coordinates.
- Switching from UMA-16 front-only mode back to normal mode restores the left-facing geometry.
- Dual-array mode reports classification, bearing, coordinates, and range with a real sound source.
- Strict mode waits for drone confirmation before declaring a target lock.
- The GUI works on the Jetson desktop.
- A 15-minute real-audio run remains stable.

## Assumptions

- The Jetson already has a working JetPack/Ubuntu installation, network access, and a user with `sudo` privileges.
- The same microphone arrays and physical rig geometry will be used.
- Initial deployment uses the existing CPU ONNX inference path; the TensorRT engine mentioned in metadata is not required.
- No application code changes are expected. Jetson-specific changes should normally be limited to audio device indices and installed dependencies.
