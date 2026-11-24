import json
import logging
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from frequency_id.acoustics.doa_factory import create_doa_engine
from frequency_id.acoustics.geometry import RigGeometry
from frequency_id.cli import cmd_cli, cmd_simulate, parse_args
from frequency_id.gui.app import TacticalGUI


EXAMPLE_CONFIG = Path(__file__).resolve().parents[3] / "config" / "rig_config.example.json"


def write_config(tmp_path, changes=None):
    config = json.loads(EXAMPLE_CONFIG.read_text())
    if changes is not None:
        changes(config)
    path = tmp_path / "rig.json"
    path.write_text(json.dumps(config))
    return path


def test_factory_uses_geometry_calibration_and_reports_effective_sound_speed(tmp_path, caplog):
    def changes(config):
        config["sound_speed"] = 320.0
        calibration = config["arrays"]["left"]["calibration"]
        calibration["channel_delays_samples"] = {"0": 2.5, "15": -1.0}
        calibration["local_mics_m"] = {"0": [0.001, -0.061, -0.062]}
        calibration["mic_heights_local_m"] = {"1": 0.061}
        calibration["phase_response"] = {
            "frequencies_hz": [500.0, 1500.0],
            "phase_deg_by_channel": {"0": [0.0, 20.0]},
        }

    geometry = RigGeometry(str(write_config(tmp_path, changes)))
    with caplog.at_level(logging.INFO, logger="frequency_id.acoustics.doa_factory"):
        engine = create_doa_engine(geometry, "left")

    assert engine.sound_speed == 320.0
    assert engine.delays[0] == 2.5
    assert engine.delays[-1] == -1.0
    np.testing.assert_allclose(engine._positions_3d[0], [0.001, -0.061, -0.062])
    assert engine.phase_response[0][1].tolist() == [0.0, 20.0]
    assert "320.000 m/s (configured)" in caplog.text


def test_temperature_derived_sound_speed_takes_precedence(tmp_path):
    def changes(config):
        config["sound_speed"] = 350.0
        config["temperature_c"] = 20.0

    geometry = RigGeometry(str(write_config(tmp_path, changes)))
    assert geometry.sound_speed == pytest.approx(331.3 + 0.606 * 20.0)
    assert geometry.sound_speed_source == "temperature_c"
    assert create_doa_engine(geometry, "left").sound_speed == geometry.sound_speed


def test_unmeasured_calibration_defaults_to_zero_and_disabled():
    geometry = RigGeometry(str(EXAMPLE_CONFIG))
    for array in geometry.arrays.values():
        assert set(array.channel_delays) == set(array.local_mics)
        assert all(value == 0.0 for value in array.channel_delays.values())
        assert array.phase_response == {}


@pytest.mark.parametrize(
    "calibration,match",
    [
        ({"channel_delays_samples": {"99": 1.0}}, "Unknown channel-delay channel"),
        ({"channel_delays_samples": {"0": float("nan")}}, "finite samples"),
        ({"local_mics_m": {"0": [0.0, float("inf")]}}, "coordinates must be finite"),
        ({"mic_heights_local_m": {"0": "bad"}}, "could not convert string"),
        (
            {"phase_response": {"frequencies_hz": [1000.0, 900.0], "phase_deg_by_channel": {}}},
            "strictly increasing",
        ),
        (
            {"phase_response": {"frequencies_hz": [900.0, 1100.0], "phase_deg_by_channel": {"0": [1.0]}}},
            "match frequencies_hz",
        ),
    ],
)
def test_invalid_calibration_is_rejected(tmp_path, calibration, match):
    def changes(config):
        config["arrays"]["left"]["calibration"] = calibration

    with pytest.raises(ValueError, match=match):
        RigGeometry(str(write_config(tmp_path, changes)))


@pytest.mark.parametrize("temperature", [float("nan"), -1000.0])
def test_invalid_temperature_is_rejected(tmp_path, temperature):
    def changes(config):
        config["temperature_c"] = temperature

    with pytest.raises(ValueError):
        RigGeometry(str(write_config(tmp_path, changes)))


def cli_args(*arguments):
    with patch.object(sys, "argv", ["main.py", *arguments]):
        return parse_args()


def test_headless_cli_constructs_through_shared_factory():
    args = cli_args("cli", "--simulate", "--no-classifier", "--no-vision")
    audio = MagicMock()
    audio.get_latest_blocks.side_effect = KeyboardInterrupt
    with (
        patch("frequency_id.cli.create_doa_engine") as factory,
        patch("frequency_id.cli.MultiDeviceAudioManager", return_value=audio),
        patch("frequency_id.cli.create_classifier", return_value=None),
        patch("frequency_id.cli.create_vision", return_value=None),
    ):
        cmd_cli(args)
    assert [call.args[1] for call in factory.call_args_list] == ["front", "left"]
    audio.stop.assert_called_once()


def test_offline_cli_simulation_constructs_through_shared_factory():
    args = cli_args("simulate")
    audio = MagicMock()
    block = SimpleNamespace(samples=np.zeros((8820, 16)), captured_at=1.0)
    audio.get_latest_blocks.return_value = {"front": block, "left": block}
    detector = MagicMock()
    detector.process.return_value = SimpleNamespace(peak_freq=1010.0)
    result = SimpleNamespace(global_angle_deg=0.0, prominence=1.0, stable=False)
    engine = MagicMock()
    engine.estimate.return_value = result
    triangulation = MagicMock(success=False)
    with (
        patch("frequency_id.cli.create_doa_engine", return_value=engine) as factory,
        patch("frequency_id.cli.MultiDeviceAudioManager", return_value=audio),
        patch("frequency_id.cli.FrequencyDetector", return_value=detector),
        patch("frequency_id.cli.Triangulator") as triangulator,
        patch("frequency_id.cli.create_classifier", return_value=None),
    ):
        triangulator.return_value.triangulate.return_value = triangulation
        cmd_simulate(args)
    assert [call.args[1] for call in factory.call_args_list] == ["front", "left"]
    assert engine.estimate.call_count == 30
    for call in engine.estimate.call_args_list:
        assert "captured_at" in call.kwargs


def test_gui_initialization_and_mode_change_use_shared_factory():
    geometry = RigGeometry(str(EXAMPLE_CONFIG))
    root = MagicMock()
    with (
        patch.object(TacticalGUI, "_build_ui", lambda self: None),
        patch("frequency_id.gui.app.MultiDeviceAudioManager", return_value=MagicMock()),
        patch("frequency_id.gui.app.create_doa_engine") as factory,
    ):
        factory.side_effect = lambda *args, **kwargs: MagicMock()
        app = TacticalGUI(root, geometry=geometry, simulate=True)
        assert [call.args[1] for call in factory.call_args_list] == ["front", "left"]

        app.var_front_only = MagicMock()
        app.var_front_only.get.return_value = False
        app.var_left_only = MagicMock()
        app.var_left_only.get.return_value = False
        app.var_uma16_front_only = MagicMock()
        app.var_uma16_front_only.get.return_value = True
        app._update_suppression_controls = MagicMock()
        app._clear_target_telemetry = MagicMock()
        app._reset_classifier = MagicMock()
        app._draw_tactical_map = MagicMock()
        app._draw_direction_3d = MagicMock()
        app.is_active = False
        factory.reset_mock()
        app.on_array_mode_change("uma16_front")

    factory.assert_called_once()
    assert factory.call_args.args[:2] == (geometry, "left")
    assert geometry.arrays["left"].orientation_deg == 0.0
