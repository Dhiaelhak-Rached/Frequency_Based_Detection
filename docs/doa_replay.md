# Offline DOA replay and known-angle capture protocol

## Scope

`sim.harness.doa_replay` validates the existing `RigGeometry` → `create_doa_engine` → `ArrayDOA` path without microphone hardware, a GUI, PX4, or Gazebo. It accepts deterministic synthetic fixtures and uncompressed multichannel PCM WAV recordings. The committed example is synthetic only; it is not physical-hardware evidence and does not define an accuracy acceptance threshold.

The estimator uses one joint azimuth/elevation search for elevation-capable arrays, overlapping Hann FFT frames, averaged auto/cross spectra, coherence/SNR-derived bin qualities, bounded bin weights, and narrowband steering-frequency refinement. Replay does not alter those algorithms or their calibration.

## Run the synthetic example

From the repository root after installing the package, or with `PYTHONPATH=src:.` for a source checkout:

```bash
PYTHONPATH=src:. python -m sim.harness.doa_replay \
  config/doa_replay.example.json
```

The command prints one JSON report containing per-attempt observations, per-scenario summaries, and an aggregate summary. Add `--output data/doa_replay/report.local.json` to save the same report in the ignored local-data directory. JSON output rejects NaN and Infinity. An empty scenario list is valid and produces zero counts plus null rates and angle statistics; a missing recording, invalid manifest, sample-rate mismatch, or insufficient channels exits with status 2 and an explicit error.

Each scenario constructs a fresh engine and replays the same block `repetitions` times at deterministic 0.2-second timestamps. Repetition exercises freshness and temporal stability; it is not an independent Monte Carlo trial. Use separate captured clips as separate scenarios for independent physical evidence.

## Local data and WAV requirements

Real recordings and local manifests belong under:

```text
data/doa_replay/
```

That directory is gitignored. Do not commit WAV files, large binary fixtures, generated plots, or local recordings. Only the small example manifest, replay code, tests, and this protocol are committed.

The WAV loader supports uncompressed 8-, 16-, 24-, and 32-bit integer PCM. It requires:

- the exact manifest sample rate;
- interleaved samples shaped as `(frames, channels)`;
- every original array channel index, so a full UMA-16 recording normally has 16 channels even if calibration disables some;
- finite-duration PCM data without a compressed or floating-point codec.

Recording paths and `rig_config` are project-relative unless absolute. Path resolution uses `frequency_id.paths.resolve_project_path`, including linked-worktree fallback for existing local assets.

## Manifest schema

The top-level fields are:

| Field | Meaning |
|---|---|
| `schema_version` | Must be `1`. |
| `array_id` | Geometry key such as `left` or `front`. |
| `calibration_identity` | Operator-assigned identity tying the clips to the exact measured/calibrated rig state; do not invent an identity for an unknown rig. |
| `rig_config` | Geometry and calibration JSON used to construct the engine. |
| `sample_rate` | Required WAV and synthetic sample rate in Hz. |
| `angle_reference` | `local` for array-local ground truth or `global` for rig/body bearings. |
| `uma16_front_only` | Optional boolean selecting the same front-only geometry as the application. |
| `target_bandwidth_hz` | Optional target gate width, default 60 Hz. |
| `band_hz` | Optional analysis `[low_hz, high_hz]`; a scenario can override it. |
| `scenarios` | List of recording or synthetic scenario objects; an empty list is valid. |

Every scenario records:

| Field | Meaning |
|---|---|
| `id` | Unique nonempty scenario name. |
| `scenario_type` | `tone`, `interferer`, `silence`, or `noise_only`. |
| `recording` | Project-relative/absolute WAV path, or `null` for a deterministic synthetic fixture. |
| `duration_s` | Synthetic block duration; use the actual clip duration as descriptive recording metadata. |
| `seed` | Nonnegative synthetic random seed. |
| `repetitions` | Positive number of timed/tracked passes over the same block. |
| `true_azimuth_deg` | Measured ground-truth azimuth for a tone/interferer; otherwise `null`. |
| `true_elevation_deg` | Measured ground-truth elevation in `[-90, 90]`; otherwise `null`. |
| `true_distance_m` | Measured source-to-acoustic-center distance for a tone/interferer; otherwise `null`. |
| `source_frequency_hz` | Target tone frequency for a tone/interferer; otherwise `null`. |
| `snr_db` | Measured or synthetic target SNR, or `null` when not defined. |
| `reverberation` | Conditions and measurement method, for example `anechoic`, `RT60 0.42 s; measured ...`, or `not measured; office ...`. |
| `channel_mismatch` | Applied/measured gain, delay, position, and phase mismatch state, or `not measured`; calibration identity alone is not a residual measurement. |
| `noise_rms` | Optional synthetic noise-only RMS, default 0.01. |
| `interferer` | Required only for `interferer`: `azimuth_deg`, `elevation_deg`, `frequency_hz`, and target-relative `relative_level_db`. |

Synthetic fixtures are ideal far-field plane waves at the geometry's effective sound speed. Distance, reverberation text, and channel-mismatch text are metadata; the current lightweight generator does not claim to simulate spherical spreading, room impulse responses, or unprovided mismatch measurements.

## Metrics and missing observations

- **Circular azimuth error:** `abs((estimated - true + 180) mod 360 - 180)` in degrees, using local or global angles according to `angle_reference`.
- **Elevation error:** absolute estimated-minus-true elevation in degrees when both exist.
- **3D direction-vector angular error:** the arc-cosine of the clipped dot product between the reported unit direction and the ground-truth unit direction. Local truth is rotated by the array orientation before comparison with the body's `[right, forward, up]` vector convention.
- **Valid-detection rate:** attempts with no DOA error and finite local and global azimuths divided by all attempts. It is a structural validity metric, not proof that an angle is correct; use per-scenario angle errors and negative controls alongside it.
- **False-stable rate:** silence/noise-only attempts where azimuth or elevation is marked stable, divided by silence/noise-only attempts. Silence misses and noise false detections remain visible per attempt.
- **DOA runtime:** `time.perf_counter_ns()` around each `ArrayDOA.estimate` call. Reports contain count, total, mean, median, 95th percentile (NumPy linear percentile interpolation), and maximum in milliseconds.
- **Replay wall runtime:** end-to-end geometry setup, loading/synthesis, engine construction, estimation, and report aggregation, measured with `time.perf_counter_ns()`.
- **Peak memory:** `tracemalloc` peak over the complete replay, reported in MiB. This is Python traced allocation, not total process RSS or accelerator memory.

A missing estimate, missing elevation, missing vector, or missing ground-truth angle produces a `null` corresponding error. Aggregates include only available pairs and expose their `count`; empty sets have null mean/median/p95/max. No zero-degree or perfect score is substituted for unavailable data.

## Owner-dependent physical capture steps

No physical known-angle clips are shipped. To produce hardware evidence:

1. Record the source commit and exact `rig_config.json`. Verify array orientation, active channels, measured local microphone coordinates/heights, channel-delay units/signs, phase-response data, and sound-speed source. `temperature_c`, when non-null, takes precedence over configured `sound_speed`; leave unmeasured calibration at zero/disabled defaults.
2. Fix the array and speaker positions. Survey the speaker relative to the array acoustic center in the declared local or global frame, including distance and signed elevation. Include left/right, above/below, an endfire/boundary case where physically meaningful, and several distances. Do not infer “known” angles from an unmeasured camera image.
3. Capture raw synchronized multichannel PCM WAV at the configured rate with all required channel indices preserved. Disable unrelated processing where possible and document any device gain, AGC, or beamforming that cannot be disabled. Keep clips short (for example 0.2 seconds) for block-level replay; use separate clips for independent repetitions.
4. Capture digital silence and ordinary ambient/device noise with no target. For interferer cases, independently measure/log both source directions and frequencies plus their relative level. Include operating propeller noise only under a separately approved safe rig procedure; never energize motors merely for this offline tool.
5. Log source frequency, distance, azimuth/elevation, SNR and how it was estimated, reverberation and any RT60 measurement method, channel-mismatch state, and a calibration identity for every clip. Use `not measured` where appropriate rather than fabricating metadata.
6. Store the WAV files and a copy of the example manifest under `data/doa_replay/`. Replace each synthetic scenario's `recording: null` with its project-relative WAV path and enter the measured ground truth. Do not use the example's synthetic calibration identity for a physical capture.
7. Run the replay command and preserve the JSON output with the source commit/configuration identity. Report circular azimuth, elevation, vector error, valid detections, false-stable rate, runtime, and memory separately for each scenario and in aggregate. Physical accuracy limits must be derived from these real measurements and an owner-approved sample design, never from synthetic fixtures.
8. If a configured Jetson and the same rig are available, copy/install the same commit through the existing documented deployment process and run the tool locally on that target. Do not start remote connections, motors, or interactive GUI processes from the replay tool. Compare the same clips and manifest on both machines; report host/target, exact commands, runtime, memory, and any unavailable dependency.

Until those captures exist, only synthetic/replay correctness and performance can be claimed. Hardware angular accuracy, propeller-noise robustness, and Jetson benchmark results remain owner-dependent validation steps.
