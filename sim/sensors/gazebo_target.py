"""Configurable 3D acoustic target trajectories and Gazebo Garden entity bridge."""

from __future__ import annotations

from dataclasses import dataclass
import math
import subprocess
import threading
import time
from typing import Callable, Optional, Tuple

import numpy as np


@dataclass
class TargetSample:
    north_m: float
    east_m: float
    altitude_m: float
    active: bool = True

    @property
    def ned(self) -> np.ndarray:
        return np.array([self.north_m, self.east_m, -self.altitude_m], dtype=float)


TargetTrajectoryFn = Callable[[float], TargetSample]


def static_target(north_m: float, east_m: float, altitude_m: float) -> TargetTrajectoryFn:
    def _fn(_t: float) -> TargetSample:
        return TargetSample(north_m=north_m, east_m=east_m, altitude_m=altitude_m, active=True)
    return _fn


def moving_target(
    start_north: float,
    start_east: float,
    start_alt: float,
    vel_north: float = 0.0,
    vel_east: float = 0.0,
    vel_alt: float = 0.0,
    move_after_s: float = 0.0,
) -> TargetTrajectoryFn:
    def _fn(t: float) -> TargetSample:
        dt = max(0.0, t - move_after_s)
        return TargetSample(
            north_m=start_north + vel_north * dt,
            east_m=start_east + vel_east * dt,
            altitude_m=start_alt + vel_alt * dt,
            active=True,
        )
    return _fn


def step_change_target(
    first: Tuple[float, float, float],
    second: Tuple[float, float, float],
    switch_at_s: float,
) -> TargetTrajectoryFn:
    def _fn(t: float) -> TargetSample:
        n, e, a = first if t < switch_at_s else second
        return TargetSample(north_m=n, east_m=e, altitude_m=a, active=True)
    return _fn


def dropout_window_target(
    base_fn: TargetTrajectoryFn,
    off_start_s: float,
    off_end_s: float,
) -> TargetTrajectoryFn:
    def _fn(t: float) -> TargetSample:
        s = base_fn(t)
        active = not (off_start_s <= t < off_end_s)
        return TargetSample(s.north_m, s.east_m, s.altitude_m, active=active)
    return _fn


class GazeboTargetVisual:
    """Spawns and updates a visible 3D target beacon in Gazebo Garden (/world/default)."""

    def __init__(self, world_name: str = "default", model_name: str = "acoustic_target"):
        self.world_name = world_name
        self.model_name = model_name
        self.spawned = False
        self._lock = threading.Lock()
        self._latest_pose: Optional[Tuple[float, float, float]] = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def spawn(self, sample: TargetSample) -> bool:
        x_enu = sample.east_m
        y_enu = sample.north_m
        z_enu = sample.altitude_m
        sdf = (
            '<?xml version="1.0"?>'
            '<sdf version="1.9">'
            f'<model name="{self.model_name}">'
            '<static>true</static>'
            f'<pose>{x_enu:.3f} {y_enu:.3f} {z_enu:.3f} 0 0 0</pose>'
            '<link name="body">'
            '<visual name="sphere_vis">'
            '<geometry><sphere><radius>0.35</radius></sphere></geometry>'
            '<material>'
            '<ambient>1.0 0.35 0.0 1</ambient>'
            '<diffuse>1.0 0.45 0.0 1</diffuse>'
            '<emissive>0.8 0.25 0.0 1</emissive>'
            '</material>'
            '</visual>'
            '</link>'
            '</model>'
            '</sdf>'
        )
        req = f'sdf: {sdf!r}'.replace("'", '"')
        # Pass sdf as double-quoted protobuf text with escaped internal quotes
        escaped_sdf = sdf.replace('"', '\\"')
        req = f'sdf: "{escaped_sdf}"'
        try:
            proc = subprocess.run(
                [
                    "gz", "service",
                    "-s", f"/world/{self.world_name}/create",
                    "--reqtype", "gz.msgs.EntityFactory",
                    "--reptype", "gz.msgs.Boolean",
                    "--timeout", "1500",
                    "--req", req,
                ],
                capture_output=True,
                text=True,
                timeout=3.0,
            )
            self.spawned = proc.returncode == 0
        except Exception:
            self.spawned = False

        if self.spawned and self._thread is None:
            self._stop.clear()
            self._thread = threading.Thread(target=self._pose_worker, daemon=True)
            self._thread.start()
        return self.spawned

    def update_pose(self, sample: TargetSample):
        with self._lock:
            self._latest_pose = (sample.east_m, sample.north_m, sample.altitude_m)

    def _pose_worker(self):
        while not self._stop.is_set():
            pose = None
            with self._lock:
                pose = self._latest_pose
                self._latest_pose = None
            if pose is not None and self.spawned:
                x_e, y_n, z_u = pose
                req = (
                    f'name: "{self.model_name}", '
                    f'position: {{x: {x_e:.3f}, y: {y_n:.3f}, z: {z_u:.3f}}}'
                )
                try:
                    subprocess.run(
                        [
                            "gz", "service",
                            "-s", f"/world/{self.world_name}/set_pose",
                            "--reqtype", "gz.msgs.Pose",
                            "--reptype", "gz.msgs.Boolean",
                            "--timeout", "500",
                            "--req", req,
                        ],
                        capture_output=True,
                        timeout=1.5,
                    )
                except Exception:
                    pass
            time.sleep(0.2)

    def close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
