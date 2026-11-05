#!/usr/bin/env python3
"""Rig geometry and spatial coordinate transformations for 3-array acoustic localization.

Coordinate System:
  * Origin (0, 0, 0): Center of front UMA-8 circular array.
  * +Y axis: Forward (Nose direction, 0 deg bearing).
  * +X axis: Right (+90 deg bearing).
  * -X axis: Left (270 / -90 deg bearing).
  * -Y axis: Rear (180 deg bearing).
  * Bearing theta: Clockwise from +Y (0 deg = Forward, 90 deg = Right).
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

SOUND_SPEED = 343.0  # m/s


@dataclass
class ArrayGeometry:
    array_id: str
    name: str
    array_type: str  # "uma8" or "uma16"
    center: np.ndarray  # shape (2,) [x, y] in meters
    orientation_deg: float  # rotation relative to forward +Y (deg)
    local_mics: Dict[int, np.ndarray] = field(default_factory=dict)  # ch -> [lx, ly] in local array frame
    mic_positions: Dict[int, np.ndarray] = field(default_factory=dict)  # ch -> [gx, gy] in global coords


class RigGeometry:
    """Unified 3-array coordinate system and mic placement manager."""

    def __init__(self, config_path: str = None):
        if config_path is None:
            config_path = str(Path(__file__).parent / "rig_config.json")
        self.config_path = config_path
        with open(config_path, "r") as f:
            self.cfg = json.load(f)

        self.sound_speed = float(self.cfg.get("sound_speed", 343.0))
        self.sample_rate = int(self.cfg.get("sample_rate", 44100))
        self.arrays: Dict[str, ArrayGeometry] = {}
        self._build_geometry()

    def _build_geometry(self):
        arr_cfgs = self.cfg.get("arrays", {})

        # 1. Front UMA-8
        front_cfg = arr_cfgs.get("front", {})
        c_front = np.array(front_cfg.get("center", [0.0, 0.0]), dtype=float)
        rot_front = float(front_cfg.get("orientation_deg", 0.0))
        radius = float(front_cfg.get("radius_m", 0.0855))
        mics_local_front = {}
        mics_global_front = {}

        # Default UMA-8 calibrated perimeter angles (degrees from reference axis)
        # ch0=center, ch4=10°, ch5=70°, ch6=130°, ch1=190°, ch2=250°, ch3=310°
        mics_local_front[0] = np.array([0.0, 0.0])
        mics_global_front[0] = c_front.copy()
        perimeter_map = {
            4: 10.0,
            5: 70.0,
            6: 130.0,
            1: 190.0,
            2: 250.0,
            3: 310.0,
        }
        for ch, deg in perimeter_map.items():
            ang = np.deg2rad(deg)
            # Local: lx along long/forward axis (0°), ly across (90°)
            lx = radius * np.cos(ang)
            ly = radius * np.sin(ang)
            mics_local_front[ch] = np.array([lx, ly])

            # Global coords
            ang_global = np.deg2rad(rot_front + deg)
            gx = c_front[0] + radius * np.sin(ang_global)
            gy = c_front[1] + radius * np.cos(ang_global)
            mics_global_front[ch] = np.array([gx, gy])

        self.arrays["front"] = ArrayGeometry(
            array_id="front",
            name=front_cfg.get("name", "UMA-8 Apex"),
            array_type="uma8",
            center=c_front,
            orientation_deg=rot_front,
            local_mics=mics_local_front,
            mic_positions=mics_global_front,
        )

        # 2. Left & Right UMA-16 arrays
        for side in ("left", "right"):
            if side not in arr_cfgs:
                continue
            cfg = arr_cfgs[side]
            c = np.array(cfg.get("center", [-0.075, -0.26] if side == "left" else [0.085, -0.285]), dtype=float)
            rot_deg = float(cfg.get("orientation_deg", -20.0 if side == "left" else 22.0))
            grid_m = float(cfg.get("grid_m", 0.042))
            rot_rad = np.deg2rad(rot_deg)

            # Axis vectors in global frame:
            # u_long points along the board from USB tail to apex
            # u_short points across the board to its right
            u_long = np.array([np.sin(rot_rad), np.cos(rot_rad)])
            u_short = np.array([np.cos(rot_rad), -np.sin(rot_rad)])

            mics_local = {}
            mics_global = {}
            for ch in range(16):
                col = ch % 8
                row = ch // 8
                lx = (col - 3.5) * grid_m   # along long axis (0 deg local)
                ly = (row - 0.5) * grid_m   # across short axis (90 deg local)
                mics_local[ch] = np.array([lx, ly])
                mics_global[ch] = c + lx * u_long + ly * u_short

            self.arrays[side] = ArrayGeometry(
                array_id=side,
                name=cfg.get("name", f"UMA-16 {side.capitalize()}"),
                array_type="uma16",
                center=c,
                orientation_deg=rot_deg,
                local_mics=mics_local,
                mic_positions=mics_global,
            )

    def local_to_global_bearing(self, array_id: str, local_bearing_deg: float) -> float:
        """Convert local array bearing angle to global rig bearing (0 deg = forward)."""
        if array_id not in self.arrays:
            return local_bearing_deg
        arr = self.arrays[array_id]
        return (local_bearing_deg + arr.orientation_deg) % 360.0

    def global_to_local_bearing(self, array_id: str, global_bearing_deg: float) -> float:
        """Convert global rig bearing to local array angle."""
        if array_id not in self.arrays:
            return global_bearing_deg
        arr = self.arrays[array_id]
        return (global_bearing_deg - arr.orientation_deg) % 360.0


def bearing_to_vector(bearing_deg: float) -> np.ndarray:
    """Convert bearing angle (0 deg = +Y, 90 deg = +X) to 2D unit vector."""
    rad = np.deg2rad(bearing_deg)
    return np.array([np.sin(rad), np.cos(rad)], dtype=float)


def vector_to_bearing(vec: np.ndarray) -> float:
    """Convert 2D vector [x, y] to bearing angle in [0, 360) deg."""
    rad = np.arctan2(vec[0], vec[1])
    return float(np.rad2deg(rad) % 360.0)
