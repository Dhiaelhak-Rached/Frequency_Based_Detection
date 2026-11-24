"""Pytest configuration ensuring src/ and repository root are importable across pytest versions."""

from pathlib import Path
import sys
import wave

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _REPO_ROOT / "src"

for _path in (str(_SRC_DIR), str(_REPO_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from tests.helpers import (  # noqa: E402, F401
    D,
    FakeBoxes,
    FakeCapture,
    FakeModel,
    FakePrediction,
    FakeTensor,
    FakeTracker,
)

import numpy as np  # noqa: E402


UMA16_LAYOUT = {
    0: (3, 1), 1: (3, 0), 2: (2, 1), 3: (2, 0),
    4: (1, 1), 5: (1, 0), 6: (0, 1), 7: (0, 0),
    8: (0, 3), 9: (0, 2), 10: (1, 3), 11: (1, 2),
    12: (2, 3), 13: (2, 2), 14: (3, 3), 15: (3, 2),
}


def vertical_uma16_positions(grid_m=0.042):
    return {
        ch: np.array([0.0, (col - 1.5) * grid_m, (1.5 - row) * grid_m], dtype=float)
        for ch, (row, col) in UMA16_LAYOUT.items()
    }


def make_uma16_doa(positions=None, **kwargs):
    from frequency_id.acoustics.doa_engine import ArrayDOA

    positions = vertical_uma16_positions() if positions is None else positions
    return ArrayDOA(
        "left",
        {ch: np.asarray(pos, dtype=float)[:2] for ch, pos in positions.items()},
        0.0,
        sample_rate=kwargs.pop("sample_rate", 44100),
        mount="vertical",
        mic_heights_local={ch: float(np.asarray(pos)[2]) for ch, pos in positions.items()},
        **kwargs,
    )


def plane_wave_block(
    positions,
    azimuth_deg,
    elevation_deg,
    frequency=1010.0,
    sample_rate=44100,
    n_samples=8820,
    snr_db=40.0,
    seed=1234,
    sound_speed=343.0,
    channel_delays_samples=None,
    phase_offsets_deg=None,
    channel_gains=None,
    envelope=None,
    interferer=None,
):
    azimuth = np.deg2rad(azimuth_deg)
    elevation = np.deg2rad(elevation_deg)
    direction = np.array([
        np.cos(elevation) * np.cos(azimuth),
        np.cos(elevation) * np.sin(azimuth),
        np.sin(elevation),
    ])
    t = np.arange(n_samples, dtype=float) / sample_rate
    rng = np.random.default_rng(seed)
    block = np.zeros((n_samples, 16), dtype=float)
    for ch, position in positions.items():
        pos = np.asarray(position, dtype=float)
        if pos.size == 2:
            pos = np.concatenate([pos, [0.0]])
        delay_samples = 0.0 if channel_delays_samples is None else channel_delays_samples.get(ch, 0.0)
        phase_deg = 0.0 if phase_offsets_deg is None else phase_offsets_deg.get(ch, 0.0)
        gain = 1.0 if channel_gains is None else channel_gains.get(ch, 1.0)
        delay = -np.dot(pos, direction) / sound_speed + delay_samples / sample_rate
        block[:, ch] = gain * np.sin(
            2.0 * np.pi * frequency * (t - delay) + np.deg2rad(phase_deg)
        )
    if envelope is not None:
        block *= np.asarray(envelope, dtype=float)[:, None]
    if interferer is not None:
        other = dict(interferer)
        other.setdefault("snr_db", None)
        other.setdefault("n_samples", n_samples)
        other.setdefault("sample_rate", sample_rate)
        other.setdefault("sound_speed", sound_speed)
        block += plane_wave_block(positions, **other)
    if snr_db is not None:
        noise = rng.normal(0.0, 10.0 ** (-snr_db / 20.0), block.shape)
        block += noise
    return block


def circular_angle_error(actual, expected):
    return abs((actual - expected + 180.0) % 360.0 - 180.0)


def write_pcm16_wav(path, samples, sample_rate=44100):
    samples = np.asarray(samples, dtype=float)
    if samples.ndim != 2:
        raise ValueError("WAV fixtures must have shape (frames, channels)")
    pcm = np.rint(np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as output:
        output.setnchannels(samples.shape[1])
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(pcm.tobytes())
    return Path(path)
