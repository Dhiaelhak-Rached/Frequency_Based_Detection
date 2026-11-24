"""Tactical GUI package: local dashboard, canvas widgets, simulation launcher, and Jetson tab."""

from .app import TacticalGUI
from .jetson_tab import JetsonRemoteTab
from .px4_launcher import (
    PX4FlightTestMixin,
    build_px4_test_command,
    parse_px4_test_line,
)
from .sim_launcher import SimulationLauncherMixin, build_simulation_command
from .widgets import (
    draw_direction_3d_canvas,
    draw_radar_canvas,
    draw_spectrum_canvas,
)

__all__ = [
    "JetsonRemoteTab",
    "PX4FlightTestMixin",
    "SimulationLauncherMixin",
    "TacticalGUI",
    "build_px4_test_command",
    "build_simulation_command",
    "draw_direction_3d_canvas",
    "draw_radar_canvas",
    "draw_spectrum_canvas",
    "parse_px4_test_line",
]
