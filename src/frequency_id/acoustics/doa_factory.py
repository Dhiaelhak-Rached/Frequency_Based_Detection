"""Shared construction of calibrated array DOA engines."""

import logging

from frequency_id.acoustics.doa_engine import ArrayDOA

logger = logging.getLogger(__name__)


def create_doa_engine(
    geometry,
    array_id,
    *,
    sample_rate=None,
    freq_lo=200.0,
    freq_hi=16000.0,
    **overrides,
):
    """Build an ArrayDOA exclusively from RigGeometry calibration and explicit overrides."""
    if array_id not in geometry.arrays:
        raise ValueError(f"Unknown acoustic array: {array_id}")
    array = geometry.arrays[array_id]
    timeout_s = geometry.cfg.get("guidance", {}).get("observation_timeout_s", 0.6)
    parameters = {
        "array_id": array.array_id,
        "mic_positions_local": array.local_mics,
        "orientation_deg": array.orientation_deg,
        "sample_rate": geometry.sample_rate if sample_rate is None else sample_rate,
        "mount": array.mount,
        "mic_heights_local": array.mic_heights_local,
        "channel_delays": array.channel_delays,
        "phase_response": array.phase_response,
        "sound_speed": geometry.sound_speed,
        "tracker_timeout_s": timeout_s,
        "freq_lo": freq_lo,
        "freq_hi": freq_hi,
    }
    parameters.update(overrides)
    engine = ArrayDOA(**parameters)
    logger.info(
        "DOA %s: sound speed %.3f m/s (%s), %d delay-calibrated channels, %d phase-response channels",
        array_id,
        geometry.sound_speed,
        geometry.sound_speed_source,
        sum(value != 0.0 for value in array.channel_delays.values()),
        len(array.phase_response),
    )
    return engine
