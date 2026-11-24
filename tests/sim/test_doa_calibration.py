import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from frequency_id.acoustics.audio_manager import MultiDeviceAudioManager
from frequency_id.acoustics.doa_factory import create_doa_engine
from frequency_id.acoustics.geometry import RigGeometry
from sim.sensors.acoustic_dsp import AcousticWavefieldDOASensor
from sim.sensors.virtual_uma16 import VirtualUMA16Sensor


EXAMPLE_CONFIG = Path(__file__).resolve().parents[2] / "config" / "rig_config.example.json"


def config_with_sound_speed(tmp_path, sound_speed):
    config = json.loads(EXAMPLE_CONFIG.read_text())
    config["sound_speed"] = sound_speed
    path = tmp_path / f"rig-{sound_speed}.json"
    path.write_text(json.dumps(config))
    return str(path)


def test_virtual_sensor_factory_calibration_and_propagation_sound_speed(tmp_path):
    slow_path = config_with_sound_speed(tmp_path, 320.0)
    with patch("sim.sensors.virtual_uma16.create_doa_engine", wraps=create_doa_engine) as factory:
        slow = VirtualUMA16Sensor(config_path=slow_path)
    factory.assert_called_once_with(slow.geo, "left", freq_lo=800.0, freq_hi=1300.0)
    assert slow.doa_engine.sound_speed == 320.0
    fast = VirtualUMA16Sensor(config_path=config_with_sound_speed(tmp_path, 360.0))

    relative = np.array([4.0, 8.0, 3.0])
    slow_block = slow._synthesize_block(relative, now=2.0, target_active=True)
    fast_block = fast._synthesize_block(relative, now=2.0, target_active=True)
    assert not np.allclose(slow_block, fast_block)


def test_acoustic_wavefield_sensor_uses_factory_and_capture_timestamp():
    with patch("sim.sensors.acoustic_dsp.create_doa_engine", wraps=create_doa_engine) as factory:
        sensor = AcousticWavefieldDOASensor()
    factory.assert_called_once_with(sensor.geo, "left", sample_rate=44100)
    for index in range(3):
        sensor.sample(
            now=4.0 + 0.2 * index,
            drone_ned=np.array([0.0, 0.0, -5.0]),
            drone_rpy_deg=(0.0, 0.0, 0.0),
            target_ned=np.array([8.0, 4.0, -7.0]),
        )
    assert sensor.doa.tracker.timestamps[-1] == pytest.approx(4.4)


def test_virtual_sensor_passes_delayed_capture_timestamp_to_doa():
    sensor = VirtualUMA16Sensor()
    for index in range(3):
        sensor.sample(
            now=6.0 + 0.2 * index,
            drone_ned=np.zeros(3),
            drone_rpy_deg=(0.0, 0.0, 0.0),
            target_ned=np.array([8.0, 4.0, -7.0]),
        )
    assert sensor.doa_engine.tracker.timestamps[-1] == pytest.approx(6.4 - sensor.sensor.delay_s)


def test_audio_manager_simulation_uses_geometry_sound_speed(tmp_path):
    slow_geo = RigGeometry(config_with_sound_speed(tmp_path, 320.0))
    fast_geo = RigGeometry(config_with_sound_speed(tmp_path, 360.0))
    slow = MultiDeviceAudioManager(slow_geo, simulate=True)
    fast = MultiDeviceAudioManager(fast_geo, simulate=True)
    for manager in (slow, fast):
        manager.sim_drone_wave = None
        manager.sim_speed_rad_per_s = 0.0
        manager.enabled_arrays = {"left"}
        manager.set_simulation_target(x=2.0, y=4.0, freq_hz=1010.0, snr_db=40.0)
    np.random.seed(120)
    slow_block = slow.get_latest_blocks()["left"].samples
    np.random.seed(120)
    fast_block = fast.get_latest_blocks()["left"].samples
    assert not np.allclose(slow_block, fast_block)
