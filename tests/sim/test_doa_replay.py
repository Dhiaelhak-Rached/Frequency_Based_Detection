import copy
import json

import numpy as np
import pytest

from frequency_id.acoustics.geometry import RigGeometry
from frequency_id.paths import resolve_project_path
from sim.harness.doa_replay import (
    circular_angle_error,
    direction_angular_error,
    load_manifest,
    load_recording,
    main,
    run_replay,
    summarize_attempts,
    synthesize_block,
    validate_manifest,
)
from tests.conftest import write_pcm16_wav


EXAMPLE_MANIFEST = "config/doa_replay.example.json"


def example_manifest():
    return copy.deepcopy(load_manifest(EXAMPLE_MANIFEST))


def empty_manifest():
    manifest = example_manifest()
    manifest["scenarios"] = []
    return manifest


def test_example_manifest_is_self_contained_and_calibrated_explicitly():
    manifest = load_manifest(EXAMPLE_MANIFEST)
    assert manifest["schema_version"] == 1
    assert manifest["calibration_identity"] == "example-uma16-zero-delay-zero-phase-v1"
    assert len(manifest["scenarios"]) == 5
    assert {scenario["scenario_type"] for scenario in manifest["scenarios"]} == {
        "silence",
        "noise_only",
        "interferer",
        "tone",
    }
    assert all(scenario["recording"] is None for scenario in manifest["scenarios"])


@pytest.mark.parametrize(
    "case,match",
    [
        ("missing", "array_id"),
        ("scenario_type", "unsupported scenario_type"),
        ("duplicate", "Duplicate scenario id"),
        ("duration", "duration_s must be positive"),
        ("noise_truth", "null source ground truth"),
        ("missing_elevation", "requires true azimuth, elevation"),
        ("interferer", "requires an interferer object"),
    ],
)
def test_manifest_validation_rejects_invalid_definitions(case, match):
    manifest = example_manifest()
    if case == "missing":
        del manifest["array_id"]
    elif case == "scenario_type":
        manifest["scenarios"][0]["scenario_type"] = "music"
    elif case == "duplicate":
        manifest["scenarios"][1]["id"] = manifest["scenarios"][0]["id"]
    elif case == "duration":
        manifest["scenarios"][0]["duration_s"] = 0.0
    elif case == "noise_truth":
        manifest["scenarios"] = [manifest["scenarios"][3]]
        manifest["scenarios"][0]["true_azimuth_deg"] = 20.0
    elif case == "missing_elevation":
        manifest["scenarios"] = [manifest["scenarios"][0]]
        manifest["scenarios"][0]["true_elevation_deg"] = None
    elif case == "interferer":
        manifest["scenarios"] = [manifest["scenarios"][4]]
        del manifest["scenarios"][0]["interferer"]
    with pytest.raises(ValueError, match=match):
        validate_manifest(manifest)


def test_empty_manifest_returns_null_accuracy_and_rates():
    report = run_replay(empty_manifest())
    summary = report["summary"]
    assert summary["scenario_count"] == 0
    assert summary["attempt_count"] == 0
    assert summary["valid_detection_rate"] is None
    assert summary["false_stable_rate"] is None
    for field in (
        "circular_azimuth_error_deg",
        "elevation_error_deg",
        "direction_vector_angular_error_deg",
    ):
        assert summary[field]["count"] == 0
        assert summary[field]["mean"] is None
        assert summary[field]["median"] is None
        assert summary[field]["p95"] is None
        assert summary[field]["max"] is None
    assert summary["doa_runtime_ms"]["total"] == 0.0
    assert summary["replay_wall_runtime_ms"] >= 0.0
    assert summary["peak_memory_mib"] > 0.0
    json.dumps(report, allow_nan=False)


def test_missing_recording_identifies_scenario_and_path(tmp_path):
    manifest = example_manifest()
    manifest["scenarios"] = manifest["scenarios"][:1]
    missing = tmp_path / "missing.wav"
    manifest["scenarios"][0]["recording"] = str(missing)
    with pytest.raises(FileNotFoundError, match="synthetic_tone_60_45") as error:
        run_replay(manifest)
    assert str(missing) in str(error.value)


def test_pcm16_wav_loading_round_trip(tmp_path):
    manifest = example_manifest()
    scenario = manifest["scenarios"][0]
    geometry = RigGeometry(str(resolve_project_path(manifest["rig_config"])))
    block = synthesize_block(scenario, geometry.arrays["left"], 44100, geometry.sound_speed)
    path = write_pcm16_wav(tmp_path / "tone.wav", block)
    loaded = load_recording(path, 44100, 16)
    assert loaded.shape == block.shape
    assert np.all(np.isfinite(loaded))
    assert np.max(np.abs(loaded - block)) <= 1.5 / 32768.0 + 1e-12


@pytest.mark.parametrize(
    "sample_rate,channels,match",
    [(22050, 16, "sample rate"), (44100, 8, "channels")],
)
def test_wav_loading_rejects_rate_and_channel_mismatch(tmp_path, sample_rate, channels, match):
    path = write_pcm16_wav(tmp_path / "invalid.wav", np.zeros((32, channels)), sample_rate)
    with pytest.raises(ValueError, match=match):
        load_recording(path, 44100, 16)


def test_wav_synthetic_recording_end_to_end(tmp_path):
    manifest = example_manifest()
    scenario = manifest["scenarios"][0]
    geometry = RigGeometry(str(resolve_project_path(manifest["rig_config"])))
    block = synthesize_block(scenario, geometry.arrays["left"], 44100, geometry.sound_speed)
    scenario["recording"] = str(write_pcm16_wav(tmp_path / "known_angle.wav", block))
    manifest["scenarios"] = [scenario]
    report = run_replay(manifest)
    summary = report["summary"]
    assert summary["attempt_count"] == 3
    assert summary["valid_detection_rate"] == 1.0
    assert summary["circular_azimuth_error_deg"]["max"] <= 2.0
    assert summary["elevation_error_deg"]["max"] <= 2.0
    assert summary["direction_vector_angular_error_deg"]["max"] <= 2.0
    assert summary["false_stable_rate"] is None
    assert summary["doa_runtime_ms"]["total"] > 0.0


def test_synthetic_example_end_to_end_reports_all_metrics():
    report = run_replay(example_manifest())
    summary = report["summary"]
    assert summary["scenario_count"] == 5
    assert summary["attempt_count"] == 15
    assert summary["valid_detection_count"] == 12
    assert summary["negative_attempt_count"] == 6
    assert 0.0 <= summary["false_stable_rate"] <= 1.0
    assert summary["circular_azimuth_error_deg"]["count"] == 9
    assert summary["circular_azimuth_error_deg"]["max"] <= 2.0
    assert summary["elevation_error_deg"]["count"] == 9
    assert summary["elevation_error_deg"]["max"] <= 2.0
    assert summary["direction_vector_angular_error_deg"]["count"] == 9
    assert summary["direction_vector_angular_error_deg"]["max"] <= 2.0
    scenarios = {scenario["metadata"]["id"]: scenario for scenario in report["scenarios"]}
    assert all(
        attempt["error"] == "no_signal" for attempt in scenarios["synthetic_silence"]["attempts"]
    )
    assert scenarios["synthetic_silence"]["summary"]["false_stable_count"] == 0
    assert scenarios["synthetic_noise_only"]["summary"]["circular_azimuth_error_deg"]["count"] == 0
    runtime = summary["doa_runtime_ms"]
    assert runtime["count"] == 15
    assert runtime["median"] > 0.0
    assert runtime["p95"] >= runtime["median"]
    assert runtime["max"] >= runtime["p95"]
    assert runtime["total"] == pytest.approx(
        sum(attempt["runtime_ms"] for scenario in report["scenarios"] for attempt in scenario["attempts"])
    )
    assert summary["replay_wall_runtime_ms"] >= runtime["total"]
    assert summary["peak_memory_mib"] > 0.0
    json.dumps(report, allow_nan=False)


def test_metric_helpers_handle_circular_3d_and_missing_values():
    assert circular_angle_error(350.0, 10.0) == pytest.approx(20.0)
    assert circular_angle_error(None, 10.0) is None
    assert direction_angular_error([0.0, 1.0, 0.0], 0.0, 0.0) == pytest.approx(0.0)
    assert direction_angular_error([0.0, 1.0, 0.0], 90.0, 0.0) == pytest.approx(90.0)
    assert direction_angular_error(None, 90.0, 0.0) is None
    assert direction_angular_error([0.0, 1.0, 0.0], 90.0, None) is None


def test_summary_separates_false_stability_from_missing_accuracy():
    attempts = [
        {
            "scenario_type": "tone",
            "valid_detection": True,
            "false_stable": False,
            "circular_azimuth_error_deg": None,
            "elevation_error_deg": None,
            "direction_vector_angular_error_deg": None,
            "runtime_ms": 2.0,
        },
        {
            "scenario_type": "silence",
            "valid_detection": False,
            "false_stable": False,
            "circular_azimuth_error_deg": None,
            "elevation_error_deg": None,
            "direction_vector_angular_error_deg": None,
            "runtime_ms": 1.0,
        },
        {
            "scenario_type": "noise_only",
            "valid_detection": True,
            "false_stable": True,
            "circular_azimuth_error_deg": None,
            "elevation_error_deg": None,
            "direction_vector_angular_error_deg": None,
            "runtime_ms": 3.0,
        },
    ]
    summary = summarize_attempts(attempts)
    assert summary["valid_detection_rate"] == pytest.approx(2.0 / 3.0)
    assert summary["negative_attempt_count"] == 2
    assert summary["false_stable_count"] == 1
    assert summary["false_stable_rate"] == pytest.approx(0.5)
    assert summary["circular_azimuth_error_deg"]["count"] == 0
    assert summary["circular_azimuth_error_deg"]["mean"] is None
    assert summary["doa_runtime_ms"]["total"] == pytest.approx(6.0)


def test_cli_emits_json_for_empty_manifest(tmp_path, capsys):
    path = tmp_path / "empty.json"
    path.write_text(json.dumps(empty_manifest()), encoding="utf-8")
    assert main([str(path)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["summary"]["attempt_count"] == 0
    assert report["summary"]["valid_detection_rate"] is None
