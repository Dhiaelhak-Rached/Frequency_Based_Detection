"""Offline multichannel DOA replay, synthetic fixtures, metrics, and benchmarks."""

from __future__ import annotations

import argparse
import json
import math
import time
import tracemalloc
import wave

import numpy as np

from frequency_id.acoustics.doa_factory import create_doa_engine
from frequency_id.acoustics.geometry import RigGeometry
from frequency_id.paths import resolve_project_path

SCENARIO_TYPES = {"silence", "noise_only", "interferer", "tone"}
NO_SOURCE_TYPES = {"silence", "noise_only"}
REQUIRED_SCENARIO_FIELDS = {
    "id",
    "scenario_type",
    "recording",
    "duration_s",
    "seed",
    "repetitions",
    "true_azimuth_deg",
    "true_elevation_deg",
    "true_distance_m",
    "source_frequency_hz",
    "snr_db",
    "reverberation",
    "channel_mismatch",
}


def _finite_number(value, field, *, allow_null=False, positive=False):
    if value is None and allow_null:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{field} must be a finite number" + (" or null" if allow_null else ""))
    value = float(value)
    if positive and value <= 0.0:
        raise ValueError(f"{field} must be positive")
    return value


def _integer(value, field, *, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{field} must be an integer >= {minimum}")
    return value


def _text(value, field):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _band(value, field, sample_rate):
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{field} must contain [low_hz, high_hz] or null")
    low = _finite_number(value[0], f"{field}[0]")
    high = _finite_number(value[1], f"{field}[1]")
    if low < 0.0 or high <= low or high > sample_rate / 2.0:
        raise ValueError(f"{field} is outside the available frequency range")
    return [low, high]


def validate_manifest(manifest):
    """Validate and return a version-1 replay manifest without resolving its recordings."""
    if not isinstance(manifest, dict):
        raise ValueError("Replay manifest must be a JSON object")
    required = {
        "schema_version",
        "array_id",
        "calibration_identity",
        "rig_config",
        "sample_rate",
        "angle_reference",
        "scenarios",
    }
    missing = sorted(required - set(manifest))
    if missing:
        raise ValueError(f"Replay manifest is missing fields: {', '.join(missing)}")
    if _integer(manifest["schema_version"], "schema_version", minimum=1) != 1:
        raise ValueError("Only replay schema_version 1 is supported")
    _text(manifest["array_id"], "array_id")
    _text(manifest["calibration_identity"], "calibration_identity")
    _text(manifest["rig_config"], "rig_config")
    sample_rate = _integer(manifest["sample_rate"], "sample_rate", minimum=1)
    if manifest["angle_reference"] not in ("local", "global"):
        raise ValueError("angle_reference must be 'local' or 'global'")
    if not isinstance(manifest.get("uma16_front_only", False), bool):
        raise ValueError("uma16_front_only must be a boolean")
    _finite_number(manifest.get("target_bandwidth_hz", 60.0), "target_bandwidth_hz", positive=True)
    _band(manifest.get("band_hz"), "band_hz", sample_rate)
    if not isinstance(manifest["scenarios"], list):
        raise ValueError("scenarios must be a list")

    identifiers = set()
    for index, scenario in enumerate(manifest["scenarios"]):
        label = f"scenarios[{index}]"
        if not isinstance(scenario, dict):
            raise ValueError(f"{label} must be an object")
        missing = sorted(REQUIRED_SCENARIO_FIELDS - set(scenario))
        if missing:
            raise ValueError(f"{label} is missing fields: {', '.join(missing)}")
        identifier = _text(scenario["id"], f"{label}.id")
        if identifier in identifiers:
            raise ValueError(f"Duplicate scenario id: {identifier}")
        identifiers.add(identifier)
        label = f"scenario {identifier!r}"
        scenario_type = scenario["scenario_type"]
        if scenario_type not in SCENARIO_TYPES:
            raise ValueError(f"{label} has unsupported scenario_type {scenario_type!r}")
        if scenario["recording"] is not None:
            _text(scenario["recording"], f"{label}.recording")
        _finite_number(scenario["duration_s"], f"{label}.duration_s", positive=True)
        _integer(scenario["seed"], f"{label}.seed")
        _integer(scenario["repetitions"], f"{label}.repetitions", minimum=1)
        azimuth = _finite_number(scenario["true_azimuth_deg"], f"{label}.true_azimuth_deg", allow_null=True)
        elevation = _finite_number(scenario["true_elevation_deg"], f"{label}.true_elevation_deg", allow_null=True)
        if elevation is not None and not -90.0 <= elevation <= 90.0:
            raise ValueError(f"{label}.true_elevation_deg must be within [-90, 90]")
        distance = _finite_number(
            scenario["true_distance_m"], f"{label}.true_distance_m", allow_null=True, positive=True
        )
        frequency = _finite_number(
            scenario["source_frequency_hz"], f"{label}.source_frequency_hz", allow_null=True, positive=True
        )
        if frequency is not None and frequency > sample_rate / 2.0:
            raise ValueError(f"{label}.source_frequency_hz exceeds Nyquist")
        _finite_number(scenario["snr_db"], f"{label}.snr_db", allow_null=True)
        _text(scenario["reverberation"], f"{label}.reverberation")
        _text(scenario["channel_mismatch"], f"{label}.channel_mismatch")
        _band(scenario.get("band_hz"), f"{label}.band_hz", sample_rate)
        if "noise_rms" in scenario:
            _finite_number(scenario["noise_rms"], f"{label}.noise_rms", positive=True)
        if scenario_type in NO_SOURCE_TYPES:
            if any(value is not None for value in (azimuth, elevation, distance, frequency)):
                raise ValueError(f"{label} must use null source ground truth for silence or noise-only data")
            if "interferer" in scenario:
                raise ValueError(f"{label} cannot define an interferer without a target")
        else:
            if azimuth is None or elevation is None or distance is None or frequency is None:
                raise ValueError(f"{label} requires true azimuth, elevation, distance, and source frequency")
        if scenario_type == "interferer":
            interferer = scenario.get("interferer")
            if not isinstance(interferer, dict):
                raise ValueError(f"{label} requires an interferer object")
            for field in ("azimuth_deg", "elevation_deg", "frequency_hz", "relative_level_db"):
                if field not in interferer:
                    raise ValueError(f"{label}.interferer is missing {field}")
            _finite_number(interferer["azimuth_deg"], f"{label}.interferer.azimuth_deg")
            interferer_elevation = _finite_number(
                interferer["elevation_deg"], f"{label}.interferer.elevation_deg"
            )
            if not -90.0 <= interferer_elevation <= 90.0:
                raise ValueError(f"{label}.interferer.elevation_deg must be within [-90, 90]")
            interferer_frequency = _finite_number(
                interferer["frequency_hz"], f"{label}.interferer.frequency_hz", positive=True
            )
            if interferer_frequency > sample_rate / 2.0:
                raise ValueError(f"{label}.interferer.frequency_hz exceeds Nyquist")
            _finite_number(interferer["relative_level_db"], f"{label}.interferer.relative_level_db")
        elif "interferer" in scenario:
            raise ValueError(f"{label} defines interferer metadata but is not an interferer scenario")
    return manifest


def load_manifest(path):
    """Load and validate a JSON replay manifest."""
    manifest_path = resolve_project_path(path)
    with manifest_path.open("r", encoding="utf-8") as handle:
        return validate_manifest(json.load(handle))


def load_recording(path, expected_sample_rate, required_channels):
    """Load an uncompressed PCM WAV as float64 samples in [-1, 1]."""
    recording_path = resolve_project_path(path)
    if not recording_path.is_file():
        raise FileNotFoundError(f"Replay recording not found: {recording_path}")
    with wave.open(str(recording_path), "rb") as source:
        channels = source.getnchannels()
        sample_rate = source.getframerate()
        sample_width = source.getsampwidth()
        compression = source.getcomptype()
        frame_count = source.getnframes()
        payload = source.readframes(frame_count)
    if compression != "NONE":
        raise ValueError(f"Replay recording must be uncompressed PCM: {recording_path}")
    if sample_rate != expected_sample_rate:
        raise ValueError(
            f"Replay recording sample rate {sample_rate} does not match manifest {expected_sample_rate}: {recording_path}"
        )
    if channels < required_channels:
        raise ValueError(f"Replay recording has {channels} channels; {required_channels} required: {recording_path}")
    if sample_width not in (1, 2, 3, 4):
        raise ValueError(f"Unsupported PCM sample width {sample_width}: {recording_path}")
    if len(payload) != frame_count * channels * sample_width:
        raise ValueError(f"Replay recording has truncated PCM data: {recording_path}")

    if sample_width == 1:
        samples = (np.frombuffer(payload, dtype=np.uint8).astype(np.float64) - 128.0) / 128.0
    elif sample_width == 2:
        samples = np.frombuffer(payload, dtype="<i2").astype(np.float64) / 32768.0
    elif sample_width == 3:
        octets = np.frombuffer(payload, dtype=np.uint8).reshape(-1, 3).astype(np.int64)
        integers = octets[:, 0] | (octets[:, 1] << 8) | (octets[:, 2] << 16)
        integers = np.where(integers & 0x800000, integers - 0x1000000, integers)
        samples = integers.astype(np.float64) / 8388608.0
    else:
        samples = np.frombuffer(payload, dtype="<i4").astype(np.float64) / 2147483648.0
    return np.ascontiguousarray(samples.reshape(frame_count, channels))


def _array_positions(array):
    positions = []
    for channel in sorted(array.local_mics):
        position = np.asarray(array.local_mics[channel], dtype=float).reshape(-1)
        height = position[2] if position.size == 3 else array.mic_heights_local.get(channel, 0.0)
        positions.append([position[0], position[1], height])
    return np.asarray(positions, dtype=float)


def _direction(azimuth_deg, elevation_deg):
    azimuth = math.radians(azimuth_deg)
    elevation = math.radians(elevation_deg)
    return np.array(
        [
            math.cos(elevation) * math.cos(azimuth),
            math.cos(elevation) * math.sin(azimuth),
            math.sin(elevation),
        ]
    )


def synthesize_block(scenario, array, sample_rate, sound_speed):
    """Generate one deterministic far-field multichannel fixture from manifest metadata."""
    scenario_type = scenario["scenario_type"]
    sample_count = int(round(scenario["duration_s"] * sample_rate))
    channels = sorted(array.local_mics)
    channel_count = max(channels) + 1
    block = np.zeros((sample_count, channel_count), dtype=np.float64)
    if scenario_type == "silence":
        return block
    rng = np.random.default_rng(scenario["seed"])
    if scenario_type == "noise_only":
        block += rng.normal(0.0, scenario.get("noise_rms", 0.01), block.shape)
        return block

    positions = _array_positions(array)
    times = np.arange(sample_count, dtype=np.float64) / sample_rate
    amplitude = 0.5
    frequency = scenario["source_frequency_hz"]
    direction = _direction(scenario["true_azimuth_deg"], scenario["true_elevation_deg"])
    delays = -(positions @ direction) / sound_speed
    block[:, channels] = amplitude * np.sin(2.0 * np.pi * frequency * (times[:, None] - delays[None, :]))
    if scenario_type == "interferer":
        interferer = scenario["interferer"]
        other_direction = _direction(interferer["azimuth_deg"], interferer["elevation_deg"])
        other_delays = -(positions @ other_direction) / sound_speed
        other_amplitude = amplitude * 10.0 ** (interferer["relative_level_db"] / 20.0)
        block[:, channels] += other_amplitude * np.sin(
            2.0 * np.pi * interferer["frequency_hz"] * (times[:, None] - other_delays[None, :])
        )
    if scenario["snr_db"] is not None:
        noise_std = amplitude * 10.0 ** (-scenario["snr_db"] / 20.0)
        block += rng.normal(0.0, noise_std, block.shape)
    return block


def circular_angle_error(actual_deg, expected_deg):
    """Return absolute circular angular error, or null when either angle is unavailable."""
    if actual_deg is None or expected_deg is None:
        return None
    actual = float(actual_deg)
    expected = float(expected_deg)
    if not math.isfinite(actual) or not math.isfinite(expected):
        return None
    return abs((actual - expected + 180.0) % 360.0 - 180.0)


def direction_angular_error(vector, expected_global_azimuth_deg, expected_elevation_deg):
    """Return 3D direction-vector angular error without substituting missing angles."""
    if vector is None or expected_global_azimuth_deg is None or expected_elevation_deg is None:
        return None
    actual = np.asarray(vector, dtype=float)
    if actual.shape != (3,) or not np.all(np.isfinite(actual)):
        return None
    norm = float(np.linalg.norm(actual))
    if norm <= 0.0:
        return None
    azimuth = math.radians(expected_global_azimuth_deg)
    elevation = math.radians(expected_elevation_deg)
    expected = np.array(
        [
            math.cos(elevation) * math.sin(azimuth),
            math.cos(elevation) * math.cos(azimuth),
            math.sin(elevation),
        ]
    )
    cosine = float(np.dot(actual / norm, expected))
    return math.degrees(math.acos(float(np.clip(cosine, -1.0, 1.0))))


def _distribution(values):
    values = [float(value) for value in values if value is not None and math.isfinite(value)]
    if not values:
        return {"count": 0, "mean": None, "median": None, "p95": None, "max": None}
    return {
        "count": len(values),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": max(values),
    }


def _runtime_distribution(values):
    values = list(values)
    result = _distribution(values)
    result["total"] = float(sum(value for value in values if value is not None and math.isfinite(value)))
    return result


def _ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def summarize_attempts(attempts):
    """Aggregate validity, false stability, available angle errors, and DOA runtime."""
    negative = [attempt for attempt in attempts if attempt["scenario_type"] in NO_SOURCE_TYPES]
    false_stable = sum(attempt["false_stable"] for attempt in negative)
    return {
        "attempt_count": len(attempts),
        "valid_detection_count": sum(attempt["valid_detection"] for attempt in attempts),
        "valid_detection_rate": _ratio(sum(attempt["valid_detection"] for attempt in attempts), len(attempts)),
        "negative_attempt_count": len(negative),
        "false_stable_count": false_stable,
        "false_stable_rate": _ratio(false_stable, len(negative)),
        "circular_azimuth_error_deg": _distribution(
            attempt["circular_azimuth_error_deg"] for attempt in attempts
        ),
        "elevation_error_deg": _distribution(attempt["elevation_error_deg"] for attempt in attempts),
        "direction_vector_angular_error_deg": _distribution(
            attempt["direction_vector_angular_error_deg"] for attempt in attempts
        ),
        "doa_runtime_ms": _runtime_distribution(attempt["runtime_ms"] for attempt in attempts),
    }


def _optional_float(value):
    value = float(value) if value is not None else None
    return value if value is not None and math.isfinite(value) else None


def run_replay(manifest):
    """Replay a validated manifest and return JSON-serializable per-scenario evidence."""
    validate_manifest(manifest)
    owns_tracing = not tracemalloc.is_tracing()
    if owns_tracing:
        tracemalloc.start()
    tracemalloc.reset_peak()
    wall_start = time.perf_counter_ns()
    try:
        config_path = resolve_project_path(manifest["rig_config"])
        geometry = RigGeometry(str(config_path))
        if manifest.get("uma16_front_only", False):
            geometry.set_uma16_front_only(True)
        array_id = manifest["array_id"]
        if array_id not in geometry.arrays:
            raise ValueError(f"Unknown acoustic array: {array_id}")
        array = geometry.arrays[array_id]
        required_channels = max(array.local_mics) + 1
        scenario_reports = []
        all_attempts = []
        for scenario in manifest["scenarios"]:
            if scenario["recording"] is None:
                block = synthesize_block(scenario, array, manifest["sample_rate"], geometry.sound_speed)
                recording = None
            else:
                recording = resolve_project_path(scenario["recording"])
                if not recording.is_file():
                    raise FileNotFoundError(f"Scenario {scenario['id']!r} recording not found: {recording}")
                block = load_recording(recording, manifest["sample_rate"], required_channels)
            engine = create_doa_engine(geometry, array_id, sample_rate=manifest["sample_rate"])
            band = scenario.get("band_hz", manifest.get("band_hz"))
            attempts = []
            for repetition in range(scenario["repetitions"]):
                timestamp = 1000.0 + repetition * 0.2
                start = time.perf_counter_ns()
                result = engine.estimate(
                    block,
                    target_freq=scenario["source_frequency_hz"],
                    target_bandwidth_hz=manifest.get("target_bandwidth_hz", 60.0),
                    band=band,
                    track=True,
                    captured_at=timestamp,
                    now=timestamp,
                )
                runtime_ms = (time.perf_counter_ns() - start) / 1e6
                if manifest["angle_reference"] == "local":
                    estimated_azimuth = result.local_angle_deg
                    expected_azimuth = scenario["true_azimuth_deg"]
                    expected_global = (
                        None if expected_azimuth is None else expected_azimuth + array.orientation_deg
                    )
                else:
                    estimated_azimuth = result.global_angle_deg
                    expected_azimuth = scenario["true_azimuth_deg"]
                    expected_global = expected_azimuth
                elevation_error = None
                if result.elevation_deg is not None and scenario["true_elevation_deg"] is not None:
                    elevation_error = abs(result.elevation_deg - scenario["true_elevation_deg"])
                valid_detection = bool(
                    result.error is None
                    and result.local_angle_deg is not None
                    and result.global_angle_deg is not None
                    and math.isfinite(result.local_angle_deg)
                    and math.isfinite(result.global_angle_deg)
                )
                attempt = {
                    "scenario_id": scenario["id"],
                    "scenario_type": scenario["scenario_type"],
                    "repetition": len(attempts),
                    "valid_detection": valid_detection,
                    "stable": bool(result.stable),
                    "elevation_stable": bool(result.elevation_stable),
                    "false_stable": bool(
                        scenario["scenario_type"] in NO_SOURCE_TYPES
                        and (result.stable or result.elevation_stable)
                    ),
                    "error": result.error,
                    "estimated_azimuth_deg": _optional_float(estimated_azimuth),
                    "estimated_elevation_deg": _optional_float(result.elevation_deg),
                    "circular_azimuth_error_deg": circular_angle_error(estimated_azimuth, expected_azimuth),
                    "elevation_error_deg": _optional_float(elevation_error),
                    "direction_vector_angular_error_deg": direction_angular_error(
                        result.direction_vector, expected_global, scenario["true_elevation_deg"]
                    ),
                    "runtime_ms": runtime_ms,
                }
                attempts.append(attempt)
            all_attempts.extend(attempts)
            scenario_reports.append(
                {
                    "metadata": dict(scenario),
                    "recording": None if recording is None else str(recording),
                    "sample_count": int(block.shape[0]),
                    "channel_count": int(block.shape[1]),
                    "summary": summarize_attempts(attempts),
                    "attempts": attempts,
                }
            )
        summary = summarize_attempts(all_attempts)
        summary["scenario_count"] = len(scenario_reports)
        summary["replay_wall_runtime_ms"] = (time.perf_counter_ns() - wall_start) / 1e6
        report = {
            "schema_version": 1,
            "array_id": array_id,
            "calibration_identity": manifest["calibration_identity"],
            "angle_reference": manifest["angle_reference"],
            "sample_rate": manifest["sample_rate"],
            "rig_config": str(config_path),
            "array_orientation_deg": float(array.orientation_deg),
            "summary": summary,
            "scenarios": scenario_reports,
        }
        summary["peak_memory_mib"] = tracemalloc.get_traced_memory()[1] / 1024**2
        return report
    finally:
        if owns_tracing:
            tracemalloc.stop()


def build_argument_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", help="project-relative or absolute replay manifest JSON")
    parser.add_argument("--output", help="optional project-relative or absolute JSON report destination")
    return parser


def main(argv=None):
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    try:
        report = run_replay(load_manifest(args.manifest))
        rendered = json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
        if args.output:
            output_path = resolve_project_path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(rendered + "\n", encoding="utf-8")
    except (OSError, ValueError, wave.Error) as exc:
        parser.exit(2, f"DOA replay failed: {exc}\n")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
