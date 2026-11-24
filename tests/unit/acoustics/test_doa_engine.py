import time

import numpy as np
import pytest

from frequency_id.acoustics.doa_engine import CircularTracker, ElevationTracker
from tests.conftest import (
    circular_angle_error,
    make_uma16_doa,
    plane_wave_block,
    vertical_uma16_positions,
)


def estimate(engine, block, **kwargs):
    kwargs.setdefault("target_freq", 1010.0)
    kwargs.setdefault("band", (980.0, 1040.0))
    return engine.estimate(block, **kwargs)


def assert_finite_result(result):
    for value in (
        result.local_angle_deg,
        result.global_angle_deg,
        result.prominence,
        result.confidence,
        result.elevation_deg,
        result.elevation_prominence,
        result.elevation_confidence,
    ):
        if value is not None:
            assert np.isfinite(value)
    if result.spectrum is not None:
        assert np.all(np.isfinite(result.spectrum))
    if result.direction_vector is not None:
        assert np.all(np.isfinite(result.direction_vector))


@pytest.mark.parametrize("elevation", [0.0, 30.0, 45.0])
def test_joint_3d_search_removes_high_elevation_azimuth_bias(elevation):
    positions = vertical_uma16_positions()
    block = plane_wave_block(positions, 60.0, elevation, seed=6000 + int(elevation))
    result = estimate(make_uma16_doa(), block, track=False)

    assert result.error is None
    assert circular_angle_error(result.local_angle_deg, 60.0) <= 2.0
    assert result.elevation_deg is not None
    assert abs(result.elevation_deg - elevation) <= 2.0
    assert np.linalg.norm(result.direction_vector) == pytest.approx(1.0, abs=1e-9)
    expected = np.array([
        np.cos(np.deg2rad(elevation)) * np.sin(np.deg2rad(60.0)),
        np.cos(np.deg2rad(elevation)) * np.cos(np.deg2rad(60.0)),
        np.sin(np.deg2rad(elevation)),
    ])
    assert np.rad2deg(np.arccos(np.clip(np.dot(result.direction_vector, expected), -1.0, 1.0))) <= 2.0
    assert result.spectrum.shape == (len(result.spectrum),)
    assert_finite_result(result)


def assert_miss(result, error):
    assert result.error == error
    assert result.local_angle_deg is None
    assert result.global_angle_deg is None
    assert result.elevation_deg is None
    assert result.direction_vector is None
    assert result.spectrum is None
    assert result.prominence == 0.0
    assert result.confidence == 0.0
    assert result.elevation_prominence == 0.0
    assert result.elevation_confidence == 0.0
    assert not result.stable
    assert not result.elevation_stable
    assert_finite_result(result)


def test_digital_silence_is_a_documented_miss_and_expires_stability():
    positions = vertical_uma16_positions()
    engine = make_uma16_doa()
    block = plane_wave_block(positions, 20.0, 10.0)
    for captured_at in (10.0, 10.2, 10.4):
        result = estimate(engine, block, captured_at=captured_at, now=captured_at)
    assert result.stable

    silent = estimate(engine, np.zeros_like(block), captured_at=10.6, now=10.6)
    assert_miss(silent, "no_signal")

    reacquired = estimate(engine, block, captured_at=12.0, now=12.0)
    assert reacquired.error is None
    assert not reacquired.stable
    assert not reacquired.elevation_stable


@pytest.mark.parametrize("bad_sample", [np.nan, np.inf, -np.inf])
def test_non_finite_samples_never_escape_outputs(bad_sample):
    block = plane_wave_block(vertical_uma16_positions(), 15.0, 20.0)
    block[100, 4] = bad_sample
    result = estimate(make_uma16_doa(), block)
    assert_miss(result, "invalid_samples")


@pytest.mark.parametrize(
    "block,error",
    [
        (np.zeros(100), "invalid_shape"),
        (np.zeros((100, 2)), "invalid_channels"),
        (np.zeros((0, 16)), "invalid_shape"),
        (np.zeros((8820, 16), dtype=complex), "invalid_samples"),
    ],
)
def test_invalid_shape_and_channel_inputs_fail_closed(block, error):
    assert_miss(estimate(make_uma16_doa(), block), error)


@pytest.mark.parametrize(
    "kwargs,error",
    [
        ({"target_freq": np.nan}, "invalid_band"),
        ({"target_freq": 1010.0, "band": (1100.0, 900.0)}, "invalid_band"),
        ({"target_freq": 1010.0, "band": (980.0, np.inf)}, "invalid_band"),
        ({"target_freq": 1010.0, "target_bandwidth_hz": 0.0}, "invalid_band"),
    ],
)
def test_invalid_frequency_gates_fail_closed(kwargs, error):
    block = plane_wave_block(vertical_uma16_positions(), 15.0, 20.0)
    result = make_uma16_doa().estimate(block, **kwargs)
    assert_miss(result, error)


def test_timestamped_trackers_expire_old_history_and_current_misses():
    tracker = CircularTracker(window=3, min_prominence=0.1, min_samples=2, max_age=0.5)
    tracker.add(10.0, 1.0, timestamp=1.0)
    tracker.add(12.0, 1.0, timestamp=1.2)
    assert tracker.ready
    tracker.add(14.0, 1.0, timestamp=2.0)
    assert len(tracker.estimates) == 1
    assert not tracker.ready
    assert tracker.angle == pytest.approx(14.0)
    tracker.miss(timestamp=2.1)
    assert not tracker.current_accepted

    elevation = ElevationTracker(window=3, min_prominence=0.1, min_samples=2, max_age=0.5)
    elevation.add(20.0, 1.0, timestamp=3.0)
    elevation.add(22.0, 1.0, timestamp=3.2)
    elevation.add(24.0, 1.0, timestamp=4.0)
    assert len(elevation.estimates) == 1
    assert not elevation.ready


def test_track_false_does_not_mutate_tracking_history():
    positions = vertical_uma16_positions()
    engine = make_uma16_doa()
    first = plane_wave_block(positions, 10.0, 5.0)
    second = plane_wave_block(positions, 40.0, 25.0)
    estimate(engine, first, captured_at=1.0, now=1.0)
    before = list(engine.tracker.estimates)
    elevation_before = list(engine.elevation_tracker.estimates)

    result = estimate(engine, second, track=False, captured_at=1.2, now=1.2)

    assert result.error is None
    assert not result.stable
    assert list(engine.tracker.estimates) == before
    assert list(engine.elevation_tracker.estimates) == elevation_before


def test_known_channel_delay_calibration_uses_lag_sign_convention():
    positions = vertical_uma16_positions()
    delays = {ch: float((ch % 5) * 3) for ch in positions}
    block = plane_wave_block(
        positions,
        55.0,
        35.0,
        channel_delays_samples=delays,
        seed=81,
    )
    corrected = estimate(make_uma16_doa(channel_delays=delays), block, track=False)
    assert corrected.error is None
    assert circular_angle_error(corrected.local_angle_deg, 55.0) <= 2.0
    assert abs(corrected.elevation_deg - 35.0) <= 2.0


def test_measured_three_dimensional_positions_correct_geometry_errors():
    nominal = vertical_uma16_positions()
    measured = {}
    for ch, pos in nominal.items():
        offset = np.array([0.002 * ((ch % 3) - 1), 0.004 * ((ch % 4) - 1.5), 0.018 * ((ch % 2) * 2 - 1)])
        measured[ch] = pos + offset
    block = plane_wave_block(measured, 45.0, 40.0, seed=82)
    corrected = estimate(make_uma16_doa(measured), block, track=False)
    assert corrected.error is None
    assert circular_angle_error(corrected.local_angle_deg, 45.0) <= 2.0
    assert abs(corrected.elevation_deg - 40.0) <= 2.0


def test_phase_response_correction_interpolates_measured_phase():
    positions = vertical_uma16_positions()
    phases = {ch: float((ch % 5) * 25 - 50) for ch in positions}
    response = {
        ch: {
            "frequencies_hz": [500.0, 1500.0],
            "phase_deg": [phase - 10.0, phase + 10.0],
        }
        for ch, phase in phases.items()
    }
    true_phases = {ch: phase for ch, phase in phases.items()}
    block = plane_wave_block(positions, 35.0, 30.0, phase_offsets_deg=true_phases, seed=83)
    corrected = estimate(make_uma16_doa(phase_response=response), block, track=False)
    assert corrected.error is None
    assert circular_angle_error(corrected.local_angle_deg, 35.0) <= 2.0
    assert abs(corrected.elevation_deg - 30.0) <= 2.0


def test_configured_sound_speed_controls_steering_and_alias_limit():
    positions = vertical_uma16_positions()
    block = plane_wave_block(positions, 60.0, 40.0, sound_speed=320.0, seed=84)
    engine = make_uma16_doa(sound_speed=320.0)
    result = estimate(engine, block, track=False)
    assert engine.max_direction_freq_hz == pytest.approx(320.0 / 0.084)
    assert circular_angle_error(result.local_angle_deg, 60.0) <= 2.0
    assert abs(result.elevation_deg - 40.0) <= 2.0


def test_transient_tone_uses_valid_overlapping_frames():
    envelope = np.ones(8820)
    envelope[2048:4096] = 0.0
    envelope[7168:] = 0.0
    block = plane_wave_block(vertical_uma16_positions(), 50.0, 35.0, envelope=envelope, seed=85)
    result = estimate(make_uma16_doa(), block, track=False)
    assert result.error is None
    assert circular_angle_error(result.local_angle_deg, 50.0) <= 3.0
    assert abs(result.elevation_deg - 35.0) <= 3.0
    assert_finite_result(result)


def test_target_gate_rejects_loud_interferer():
    interferer = {
        "azimuth_deg": -30.0,
        "elevation_deg": -20.0,
        "frequency": 1180.0,
        "channel_gains": {ch: 4.0 for ch in range(16)},
        "seed": 86,
    }
    block = plane_wave_block(
        vertical_uma16_positions(),
        60.0,
        45.0,
        interferer=interferer,
        seed=87,
    )
    result = estimate(make_uma16_doa(), block, track=False, band=(800.0, 1300.0))
    assert result.error is None
    assert circular_angle_error(result.local_angle_deg, 60.0) <= 3.0
    assert abs(result.elevation_deg - 45.0) <= 3.0


@pytest.mark.parametrize("snr_db,tolerance", [(0.0, 5.0), (10.0, 3.0), (20.0, 2.0), (30.0, 2.0)])
def test_fixed_seed_varying_snr_remains_finite_and_accurate(snr_db, tolerance):
    block = plane_wave_block(vertical_uma16_positions(), 45.0, 30.0, snr_db=snr_db, seed=88)
    result = estimate(make_uma16_doa(), block, track=False)
    assert result.error is None
    assert circular_angle_error(result.local_angle_deg, 45.0) <= tolerance
    assert abs(result.elevation_deg - 30.0) <= tolerance
    assert_finite_result(result)


def test_moving_source_does_not_mix_stale_history_after_timeout():
    positions = vertical_uma16_positions()
    engine = make_uma16_doa()
    early = plane_wave_block(positions, 10.0, 5.0, seed=89)
    late = plane_wave_block(positions, 18.0, 9.0, seed=90)
    for captured_at in (1.0, 1.2, 1.4):
        estimate(engine, early, captured_at=captured_at, now=captured_at)
    result = estimate(engine, late, captured_at=3.0, now=3.0)
    assert result.local_angle_deg == pytest.approx(18.0, abs=2.0)
    assert result.elevation_deg == pytest.approx(9.0, abs=2.0)
    assert not result.stable
    assert len(engine.tracker.estimates) == 1


def test_monotonic_fallback_is_used_without_capture_timestamp(monkeypatch):
    engine = make_uma16_doa()
    block = plane_wave_block(vertical_uma16_positions(), 10.0, 5.0)
    monkeypatch.setattr(time, "monotonic", lambda: 50.0)
    result = estimate(engine, block)
    assert result.error is None
    assert engine.tracker.timestamps[-1] == pytest.approx(50.0)
