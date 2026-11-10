#!/usr/bin/env python3
"""Rig geometry and spatial coordinate transformations for dual-array acoustic localization.

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
    orientation_deg: float  # flat rotation, or acoustic-face normal when vertical (deg)
    local_mics: Dict[int, np.ndarray] = field(default_factory=dict)  # ch -> [lx, ly] in local array frame
    mic_positions: Dict[int, np.ndarray] = field(default_factory=dict)  # ch -> [gx, gy] in global coords
    mic_heights_local: Dict[int, float] = field(default_factory=dict)
    mount: str = "flat"


class RigGeometry:
    """Unified dual-array coordinate system and mic placement manager."""

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
        front_mount = front_cfg.get("mount", "flat")
        if front_mount not in ("flat", "vertical"):
            raise ValueError(f"Unsupported UMA-8 mount: {front_mount!r}")
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
        rot_rad = np.deg2rad(rot_front)
        u_short = np.array([np.cos(rot_rad), -np.sin(rot_rad)])
        for ch, deg in perimeter_map.items():
            ang = np.deg2rad(deg)
            lx = radius * np.cos(ang)
            ly = radius * np.sin(ang)
            if front_mount == "vertical":
                mics_local_front[ch] = np.array([0.0, ly])
                mics_global_front[ch] = c_front + ly * u_short
            else:
                mics_local_front[ch] = np.array([lx, ly])
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
            mount=front_mount,
        )

        # 2. UMA-16 arrays. Channel N corresponds to PCB MIC N+1.
        # Rows run from the far end (0) to the USB end (3); columns run left to right.
        uma16_mic_layout = {
            0: (3, 1), 1: (3, 0), 2: (2, 1), 3: (2, 0),
            4: (1, 1), 5: (1, 0), 6: (0, 1), 7: (0, 0),
            8: (0, 3), 9: (0, 2), 10: (1, 3), 11: (1, 2),
            12: (2, 3), 13: (2, 2), 14: (3, 3), 15: (3, 2),
        }
        for side in ("left", "right"):
            if side not in arr_cfgs:
                continue
            cfg = arr_cfgs[side]
            default_xmos = [-0.295598291, -0.214118310] if side == "left" else [0.289401709, -0.214118310]
            xmos_center = np.array(cfg.get("xmos_center", cfg.get("center", default_xmos)), dtype=float)
            rot_deg = float(cfg.get("orientation_deg", -90.0 if side == "left" else 90.0))
            grid_m = float(cfg.get("grid_m", 0.042))
            mount = cfg.get("mount", "flat")
            if mount not in ("flat", "vertical"):
                raise ValueError(f"Unsupported UMA-16 mount: {mount!r}")
            center_offset = np.array(cfg.get("mic_center_offset_local_m", [0.0, 0.0]), dtype=float)
            active_channels = set(cfg.get("active_channels", range(16)))
            rot_rad = np.deg2rad(rot_deg)

            # Flat: u_long runs USB end -> far end. Vertical: u_long is the acoustic-face normal.
            # u_short points to the viewer's right on the acoustic face; column 0 is viewer-left.
            u_long = np.array([np.sin(rot_rad), np.cos(rot_rad)])
            u_short = np.array([np.cos(rot_rad), -np.sin(rot_rad)])
            acoustic_center = xmos_center + center_offset[0] * u_long + center_offset[1] * u_short

            mics_local = {}
            mics_global = {}
            mic_heights_local = {}
            for ch, (row, col) in uma16_mic_layout.items():
                if ch not in active_channels:
                    continue
                column_offset = (col - 1.5) * grid_m
                if mount == "vertical":
                    grid_local = np.array([0.0, column_offset])
                    mic_heights_local[ch] = (1.5 - row) * grid_m
                else:
                    grid_local = np.array([(1.5 - row) * grid_m, column_offset])
                mics_local[ch] = grid_local - center_offset
                mics_global[ch] = xmos_center + grid_local[0] * u_long + grid_local[1] * u_short

            self.arrays[side] = ArrayGeometry(
                array_id=side,
                name=cfg.get("name", f"UMA-16 {side.capitalize()}"),
                array_type="uma16",
                center=acoustic_center,
                orientation_deg=rot_deg,
                local_mics=mics_local,
                mic_positions=mics_global,
                mic_heights_local=mic_heights_local,
                mount=mount,
            )

    def set_uma16_front_only(self, enabled: bool):
        """Place the left UMA-16 at the forward rig origin, or restore its normal position."""
        arr = self.arrays.get("left")
        if arr is None:
            raise ValueError("UMA-16 front-only mode requires the left UMA-16 configuration")

        original = getattr(self, "_uma16_front_original", None)
        if original is None:
            original = (
                arr.center.copy(),
                arr.orientation_deg,
                arr.name,
                {ch: pos.copy() for ch, pos in arr.mic_positions.items()},
            )
            self._uma16_front_original = original

        if enabled:
            arr.center = np.zeros(2, dtype=float)
            arr.orientation_deg = 0.0
            arr.name = "UMA-16 Front"
            arr.mic_positions = {
                ch: np.array([pos[1], pos[0]], dtype=float)
                for ch, pos in arr.local_mics.items()
            }
        else:
            center, orientation, name, mic_positions = original
            arr.center = center.copy()
            arr.orientation_deg = orientation
            arr.name = name
            arr.mic_positions = {ch: pos.copy() for ch, pos in mic_positions.items()}

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
