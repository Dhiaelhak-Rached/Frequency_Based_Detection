"""PX4 SITL + Gazebo Garden + MAVSDK-Python harness (simulation-only, never connects to hardware)."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
from typing import Callable, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import numpy as np

from frequency_id.control.guidance import (
    AcousticObservation,
    FlightState,
    GuidanceCommand,
    GuidanceConfig,
    GuidanceController,
    GuidanceState,
)
from frequency_id.control.px4 import PX4FlightMonitor, validate_observation_metadata
from frequency_id.paths import (
    PROJECT_ROOT,
    REPORTS_DIR,
    get_default_rig_config_path,
    get_px4_dir,
)
from sim.sensors.gazebo_target import GazeboTargetVisual, TargetSample, TargetTrajectoryFn
from sim.sensors.virtual_uma16 import AcousticSensorConfig, VirtualUMA16Sensor
from sim.harness.visualizer import Live3DGuidanceVisualizer, SimRecord


PX4_DIR = get_px4_dir()
REPO_ROOT = PROJECT_ROOT


def assert_sitl_endpoint(address: str):
    """Safety guard: refuse anything other than localhost/wildcard UDP/TCP SITL endpoints."""
    addr = address.strip()
    if addr.startswith("serial://") or "/dev/" in addr:
        raise RuntimeError(f"REFUSING hardware serial endpoint {address!r} in simulation harness")
    parsed = urlparse(addr)
    host = (parsed.hostname or "").lower()
    if host not in ("", "127.0.0.1", "localhost", "0.0.0.0"):
        raise RuntimeError(f"REFUSING non-local endpoint {address!r} in simulation harness")


def load_sim_guidance_config(overrides: Optional[Dict] = None) -> Tuple[GuidanceConfig, str, float]:
    with open(get_default_rig_config_path(), "r") as f:
        raw = json.load(f).get("guidance", {})
    merged = dict(raw)
    if overrides:
        merged.update(overrides)
    mount_id = str(merged.get("mount_calibration_id", "uma16-front-body-v1"))
    rate_hz = float(merged.get("control_rate_hz", 20.0))
    return GuidanceConfig.from_mapping(merged), mount_id, rate_hz


def write_sim_rig_config(dest: Path, overrides: Optional[Dict] = None) -> Path:
    """Create a simulation-only rig_config.json copy with guidance.enabled=true."""
    with open(get_default_rig_config_path(), "r") as f:
        cfg = json.load(f)
    guidance_cfg = dict(cfg.get("guidance", {}))
    guidance_cfg["enabled"] = True
    if overrides:
        guidance_cfg.update(overrides)
    cfg["guidance"] = guidance_cfg
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(cfg, indent=2))
    return dest


def build_sitl_environment(headless: bool) -> Dict[str, str]:
    env = os.environ.copy()
    if headless:
        return env

    qt_plugins = Path("/usr/lib") / f"{os.uname().machine}-linux-gnu" / "qt5" / "plugins"
    if qt_plugins.is_dir():
        env["QT_QPA_PLATFORM"] = "xcb"
        env["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(qt_plugins / "platforms")
        env["QT_PLUGIN_PATH"] = str(qt_plugins)

    library_path = env.get("LD_LIBRARY_PATH")
    if library_path:
        clean_parts = [
            part for part in library_path.split(":")
            if part and "site-packages/cv2" not in part
        ]
        if clean_parts:
            env["LD_LIBRARY_PATH"] = ":".join(clean_parts)
        else:
            env.pop("LD_LIBRARY_PATH", None)
    return env


def apply_gui_environment():
    env = build_sitl_environment(headless=False)
    for key in ("QT_QPA_PLATFORM", "QT_QPA_PLATFORM_PLUGIN_PATH", "QT_PLUGIN_PATH", "LD_LIBRARY_PATH"):
        if key in env:
            os.environ[key] = env[key]
        else:
            os.environ.pop(key, None)


class PX4SitlProcess:
    """Launches and cleanly tears down PX4 SITL + Gazebo Garden (gz_x500)."""

    def __init__(
        self,
        px4_dir: Path = PX4_DIR,
        headless: bool = True,
        log_path: Optional[Path] = None,
    ):
        self.px4_dir = px4_dir
        self.headless = headless
        self.log_path = log_path or (REPORTS_DIR / "px4_sitl.log")
        self.proc: Optional[subprocess.Popen] = None
        self._log_file = None

    def start(self, startup_wait_s: float = 12.0):
        px4_bin = self.px4_dir / "build" / "px4_sitl_default" / "bin" / "px4"
        if not px4_bin.exists():
            raise FileNotFoundError(f"PX4 SITL binary not built at {px4_bin}")
        self.kill_stale()
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_file = open(self.log_path, "w")

        env = build_sitl_environment(self.headless)
        env["PX4_SYS_AUTOSTART"] = "4001"
        env["PX4_SIM_MODEL"] = "gz_x500"
        if self.headless:
            env["HEADLESS"] = "1"
        else:
            env.pop("HEADLESS", None)

        # Clean PX4 EEPROM parameters so each session starts deterministically
        rootfs = self.px4_dir / "build" / "px4_sitl_default" / "rootfs"
        eeprom = rootfs / "eeprom"
        if eeprom.exists():
            shutil.rmtree(eeprom, ignore_errors=True)

        self.proc = subprocess.Popen(
            ["make", "px4_sitl", "gz_x500"],
            cwd=str(self.px4_dir),
            env=env,
            stdin=subprocess.PIPE,
            stdout=self._log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            text=True,
        )
        # Allow Gazebo + PX4 EKF2 initial lockstep startup
        deadline = time.monotonic() + startup_wait_s
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"PX4 SITL exited early with code {self.proc.returncode}. See {self.log_path}")
            time.sleep(0.5)

    @staticmethod
    def kill_stale():
        for pattern in (
            "[p]x4_sitl_default/bin/px4",
            "[g]z sim",
            "[r]uby.*gz",
            "[m]avsdk/bin/mavsdk_server",
        ):
            subprocess.run(["pkill", "-9", "-f", pattern], capture_output=True)
        time.sleep(0.5)

    def stop(self):
        if self.proc is not None:
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
                self.proc.wait(timeout=5.0)
            except Exception:
                try:
                    os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
                except Exception:
                    pass
            self.proc = None
        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None
        self.kill_stale()


@dataclass(frozen=True)
class PX4InstanceConfig:
    instance_id: int
    pose_enu: str = "0,0,0,0,0,0"
    udp_port: int = 14540
    mavsdk_port: int = 50051

    @property
    def connect_address(self) -> str:
        return f"udp://:{self.udp_port}"


class MultiPX4SitlCluster:
    """Launches two PX4 SITL + gz_x500 instances inside a single Gazebo Garden world."""

    def __init__(
        self,
        instances: Tuple[PX4InstanceConfig, ...] = (
            PX4InstanceConfig(instance_id=0, pose_enu="0,0,0,0,0,0", udp_port=14540, mavsdk_port=50051),
            PX4InstanceConfig(instance_id=1, pose_enu="0,8,0,0,0,0", udp_port=14541, mavsdk_port=50052),
        ),
        px4_dir: Path = PX4_DIR,
        headless: bool = False,
        log_dir: Optional[Path] = None,
    ):
        self.instances = instances
        self.px4_dir = px4_dir
        self.headless = headless
        self.log_dir = log_dir or (REPO_ROOT / "sim" / "reports")
        self._procs: List[subprocess.Popen] = []
        self._log_files: List = []

    def start(self, primary_wait_s: float = 12.0, secondary_wait_s: float = 8.0):
        px4_bin = self.px4_dir / "build" / "px4_sitl_default" / "bin" / "px4"
        if not px4_bin.exists():
            raise FileNotFoundError(f"PX4 SITL binary not built at {px4_bin}")
        PX4SitlProcess.kill_stale()
        self.log_dir.mkdir(parents=True, exist_ok=True)

        rootfs = self.px4_dir / "build" / "px4_sitl_default" / "rootfs"
        for inst in self.instances:
            eeprom_dir = rootfs / (str(inst.instance_id) if inst.instance_id > 0 else "") / "eeprom"
            if eeprom_dir.exists():
                shutil.rmtree(eeprom_dir, ignore_errors=True)

        for idx, inst in enumerate(self.instances):
            env = build_sitl_environment(self.headless)
            env["PX4_SYS_AUTOSTART"] = "4001"
            env["PX4_SIM_MODEL"] = "gz_x500"
            env["PX4_GZ_MODEL_POSE"] = inst.pose_enu
            if self.headless:
                env["HEADLESS"] = "1"
            else:
                env.pop("HEADLESS", None)

            log_file = open(self.log_dir / f"px4_sitl_{inst.instance_id}.log", "w")
            self._log_files.append(log_file)

            if idx == 0:
                cmd = ["make", "px4_sitl", "gz_x500"]
            else:
                env["PX4_GZ_STANDALONE"] = "1"
                gz_env = self.px4_dir / "build" / "px4_sitl_default" / "rootfs" / "gz_env.sh"
                etc_dir = self.px4_dir / "build" / "px4_sitl_default" / "etc"
                workdir = rootfs / str(inst.instance_id)
                workdir.mkdir(parents=True, exist_ok=True)
                cmd = [
                    "bash",
                    "-lc",
                    f"source '{gz_env}' 2>/dev/null || true; "
                    f"exec '{px4_bin}' -i {inst.instance_id} -d '{etc_dir}' -w '{workdir}'",
                ]

            proc = subprocess.Popen(
                cmd,
                cwd=str(self.px4_dir),
                env=env,
                stdin=subprocess.PIPE,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                text=True,
            )
            self._procs.append(proc)

            wait_s = primary_wait_s if idx == 0 else secondary_wait_s
            deadline = time.monotonic() + wait_s
            while time.monotonic() < deadline:
                if proc.poll() is not None:
                    raise RuntimeError(
                        f"PX4 SITL instance {inst.instance_id} exited early with code {proc.returncode}"
                    )
                time.sleep(0.5)

    def stop(self):
        for proc in self._procs:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                proc.wait(timeout=4.0)
            except Exception:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except Exception:
                    pass
        self._procs.clear()
        for f in self._log_files:
            try:
                f.close()
            except Exception:
                pass
        self._log_files.clear()
        PX4SitlProcess.kill_stale()


class AttitudeWatcher:
    """Simulator-only reader of drone world NED position, velocity, and Euler attitude."""

    def __init__(self, drone):
        self.drone = drone
        self.north_m = 0.0
        self.east_m = 0.0
        self.down_m = 0.0
        self.vx_m_s = 0.0
        self.vy_m_s = 0.0
        self.vz_m_s = 0.0  # Upward climb velocity = -vz_ned
        self.roll_deg = 0.0
        self.pitch_deg = 0.0
        self.yaw_deg = 0.0
        self._tasks: List[asyncio.Task] = []

    def start(self):
        self._tasks = [
            asyncio.create_task(self._watch_pv()),
            asyncio.create_task(self._watch_att()),
        ]

    async def stop(self):
        for t in self._tasks:
            t.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks = []

    async def _watch_pv(self):
        async for pv in self.drone.telemetry.position_velocity_ned():
            self.north_m = float(pv.position.north_m)
            self.east_m = float(pv.position.east_m)
            self.down_m = float(pv.position.down_m)
            self.vx_m_s = float(pv.velocity.north_m_s)
            self.vy_m_s = float(pv.velocity.east_m_s)
            self.vz_m_s = -float(pv.velocity.down_m_s)

    async def _watch_att(self):
        async for att in self.drone.telemetry.attitude_euler():
            self.roll_deg = float(att.roll_deg)
            self.pitch_deg = float(att.pitch_deg)
            self.yaw_deg = float(att.yaw_deg)


async def connect_sitl_drone(
    address: str = "udp://:14540",
    timeout_s: float = 30.0,
    mavsdk_port: int = 50051,
):
    assert_sitl_endpoint(address)
    from mavsdk import System

    drone = System(port=mavsdk_port)
    await drone.connect(system_address=address)

    async def _wait_conn():
        async for state in drone.core.connection_state():
            if state.is_connected:
                return

    await asyncio.wait_for(_wait_conn(), timeout=timeout_s)

    # Wait until PX4 EKF2 reports local/global position & armable
    async def _wait_health():
        async for health in drone.telemetry.health():
            if health.is_global_position_ok and health.is_home_position_ok and health.is_armable:
                return

    await asyncio.wait_for(_wait_health(), timeout=timeout_s)
    return drone


def set_gazebo_camera_pose(
    pos_enu: Tuple[float, float, float] = (-7.0, -4.0, 6.5),
    look_at_enu: Tuple[float, float, float] = (0.0, 4.0, 3.0),
) -> bool:
    """Move Gazebo Garden GUI camera to frame both drones clearly."""
    dx = look_at_enu[0] - pos_enu[0]
    dy = look_at_enu[1] - pos_enu[1]
    dz = look_at_enu[2] - pos_enu[2]
    yaw = math.atan2(dy, dx)
    pitch = -math.atan2(dz, math.hypot(dx, dy))
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    qw = cp * cy
    qx = -sp * sy
    qy = sp * cy
    qz = cp * sy
    req = (
        f"pose: {{position: {{x: {pos_enu[0]:.2f}, y: {pos_enu[1]:.2f}, z: {pos_enu[2]:.2f}}}, "
        f"orientation: {{x: {qx:.5f}, y: {qy:.5f}, z: {qz:.5f}, w: {qw:.5f}}}}}"
    )
    try:
        proc = subprocess.run(
            [
                "gz", "service",
                "-s", "/gui/move_to/pose",
                "--reqtype", "gz.msgs.GUICamera",
                "--reptype", "gz.msgs.Boolean",
                "--timeout", "1500",
                "--req", req,
            ],
            capture_output=True,
            text=True,
            timeout=2.5,
        )
        return proc.returncode == 0
    except Exception:
        return False


DEFAULT_FOLLOW_OFFSET_XYZ: Tuple[float, float, float] = (-1.8, 0.0, 0.55)


def set_gazebo_follow_offset(
    offset_xyz: Tuple[float, float, float] = DEFAULT_FOLLOW_OFFSET_XYZ,
) -> bool:
    """Update the Gazebo Garden /gui/follow/offset relative to the active follow target."""
    offset_req = f"x: {offset_xyz[0]:.2f}, y: {offset_xyz[1]:.2f}, z: {offset_xyz[2]:.2f}"
    try:
        offset_proc = subprocess.run(
            [
                "gz", "service",
                "-s", "/gui/follow/offset",
                "--reqtype", "gz.msgs.Vector3d",
                "--reptype", "gz.msgs.Boolean",
                "--timeout", "1500",
                "--req", offset_req,
            ],
            capture_output=True,
            text=True,
            timeout=2.5,
        )
        return offset_proc.returncode == 0 and "false" not in offset_proc.stdout.lower()
    except Exception:
        return False


def set_gazebo_follow_target(
    model_name: str = "x500_0",
    offset_xyz: Tuple[float, float, float] = DEFAULT_FOLLOW_OFFSET_XYZ,
) -> bool:
    """Attach Gazebo Garden GUI camera to follow a model in close GTA-style 3rd-person view."""
    try:
        follow_proc = subprocess.run(
            [
                "gz", "service",
                "-s", "/gui/follow",
                "--reqtype", "gz.msgs.StringMsg",
                "--reptype", "gz.msgs.Boolean",
                "--timeout", "1500",
                "--req", f'data: "{model_name}"',
            ],
            capture_output=True,
            text=True,
            timeout=2.5,
        )
        if follow_proc.returncode != 0 or "false" in follow_proc.stdout.lower():
            return False

        # Wait briefly for CameraTracking::PreRender to attach followNode before applying offset
        time.sleep(0.15)
        return set_gazebo_follow_offset(offset_xyz)
    except Exception:
        return False


def _visual_material_xml(rgba: str) -> str:
    return (
        "<material>"
        f"<ambient>{rgba}</ambient>"
        f"<diffuse>{rgba}</diffuse>"
        f"<emissive>{rgba}</emissive>"
        "</material>"
    )


def spawn_gazebo_motion_reference_arena(
    world_name: str = "default",
    hover_alt_m: float = 10.0,
) -> bool:
    """Spawn a collision-free visual 3D grid of pillars and altitude markers for motion parallax."""
    visuals: List[str] = [
        (
            '<visual name="takeoff_pad">'
            "<pose>0 0 0.02 0 0 0</pose>"
            "<geometry><cylinder><radius>0.8</radius><length>0.04</length></cylinder></geometry>"
            f"{_visual_material_xml('0.95 0.95 0.95 1')}"
            "</visual>"
        )
    ]

    grid_coords = (-12, -8, -4, 0, 4, 8, 12)
    for gx in grid_coords:
        for gy in grid_coords:
            if gx == 0 and gy == 0:
                continue
            if gx == 0 and gy > 0:
                rgba = "0.1 0.95 0.3 1"
            elif gx > 0 and gy == 0:
                rgba = "0.1 0.75 1.0 1"
            elif gx == 0 and gy < 0:
                rgba = "1.0 0.5 0.1 1"
            elif gx < 0 and gy == 0:
                rgba = "0.95 0.2 0.85 1"
            else:
                rgba = "0.95 0.85 0.2 1"

            pillar_rgba = rgba if (gx == 0 or gy == 0) else "0.35 0.5 0.7 1"
            tag = f"{gx}_{gy}".replace("-", "m")
            visuals.append(
                f'<visual name="pillar_{tag}">'
                f"<pose>{gx} {gy} 9.0 0 0 0</pose>"
                "<geometry><cylinder><radius>0.08</radius><length>18.0</length></cylinder></geometry>"
                f"{_visual_material_xml(pillar_rgba)}"
                "</visual>"
            )
            for alt_idx, (alt_z, radius) in enumerate(
                ((5.0, 0.18), (float(hover_alt_m), 0.28), (15.0, 0.18))
            ):
                visuals.append(
                    f'<visual name="marker_{tag}_{alt_idx}">'
                    f"<pose>{gx} {gy} {alt_z:.2f} 0 0 0</pose>"
                    f"<geometry><sphere><radius>{radius:.2f}</radius></sphere></geometry>"
                    f"{_visual_material_xml(rgba)}"
                    "</visual>"
                )

    sdf = (
        '<?xml version="1.0"?>'
        '<sdf version="1.9">'
        '<model name="motion_reference_arena">'
        "<static>true</static>"
        f'<link name="arena">{"".join(visuals)}</link>'
        "</model>"
        "</sdf>"
    )
    escaped_sdf = sdf.replace('"', '\\"')
    try:
        proc = subprocess.run(
            [
                "gz", "service",
                "-s", f"/world/{world_name}/create",
                "--reqtype", "gz.msgs.EntityFactory",
                "--reptype", "gz.msgs.Boolean",
                "--timeout", "1500",
                "--req", f'sdf: "{escaped_sdf}"',
            ],
            capture_output=True,
            text=True,
            timeout=3.0,
        )
        return proc.returncode == 0 and "false" not in proc.stdout.lower()
    except Exception:
        return False


async def arm_and_hover_sitl(
    drone,
    att: AttitudeWatcher,
    hover_alt_m: float = 5.0,
    quick: bool = False,
):
    """Ensure simulated drone is armed, airborne, and hovering near (N=0, E=0, Alt=hover_alt_m, Yaw=0)."""
    from mavsdk.offboard import PositionNedYaw, VelocityBodyYawspeed

    is_armed = False
    async for armed in drone.telemetry.armed():
        is_armed = bool(armed)
        break

    in_air = False
    async for ia in drone.telemetry.in_air():
        in_air = bool(ia)
        break

    if not is_armed or not in_air:
        await drone.action.set_takeoff_altitude(hover_alt_m)
        if not is_armed:
            arm_deadline = time.monotonic() + 15.0
            while True:
                try:
                    await drone.action.arm()
                    break
                except Exception:
                    if time.monotonic() >= arm_deadline:
                        raise
                    await asyncio.sleep(1.0)
        await drone.action.takeoff()
        takeoff_timeout = max(12.0, hover_alt_m * 1.5 + 4.0) if quick else max(25.0, hover_alt_m * 2.0 + 5.0)
        deadline = time.monotonic() + takeoff_timeout
        while time.monotonic() < deadline:
            alt_now = -att.down_m
            if quick:
                if alt_now >= max(2.2, hover_alt_m - 0.6):
                    break
            elif abs(alt_now - hover_alt_m) <= 0.4 and abs(att.vz_m_s) < 0.25:
                break
            await asyncio.sleep(0.1)

    # Reposition smoothly to (0, 0, -hover_alt_m, yaw=0) via Offboard PositionNedYaw
    await drone.offboard.set_position_ned(PositionNedYaw(0.0, 0.0, -hover_alt_m, 0.0))
    try:
        await drone.offboard.start()
    except Exception:
        # Retry once if already in offboard or mode transition settling
        await asyncio.sleep(0.3)
        await drone.offboard.set_position_ned(PositionNedYaw(0.0, 0.0, -hover_alt_m, 0.0))
        await drone.offboard.start()

    deadline = time.monotonic() + (4.0 if quick else 35.0)
    settled = False
    while time.monotonic() < deadline:
        await drone.offboard.set_position_ned(PositionNedYaw(0.0, 0.0, -hover_alt_m, 0.0))
        pos_err = math.hypot(att.north_m, att.east_m)
        alt_err = abs((-att.down_m) - hover_alt_m)
        yaw_err = abs((att.yaw_deg + 180.0) % 360.0 - 180.0)
        if quick:
            if alt_err < 0.5 and abs(att.vz_m_s) < 0.45:
                settled = True
                break
        elif pos_err < 0.35 and alt_err < 0.25 and yaw_err < 3.0 and abs(att.vz_m_s) < 0.15:
            settled = True
            break
        await asyncio.sleep(0.05)
    if not settled:
        print(
            f"[WARN] arm_and_hover_sitl({hover_alt_m}) timeout: "
            f"pos_err={math.hypot(att.north_m, att.east_m):.2f}, alt={-att.down_m:.2f}, yaw={att.yaw_deg:.1f}, vz={att.vz_m_s:.2f}",
            flush=True,
        )


@dataclass
class ScenarioResult:
    name: str
    passed: bool
    summary: str
    records: List[SimRecord]
    plot_path: Optional[Path] = None
    metrics: Dict = None


async def run_sitl_guidance_scenario(
    name: str,
    drone,
    att: AttitudeWatcher,
    target_fn: Optional[TargetTrajectoryFn] = None,
    sensor_config: Optional[AcousticSensorConfig] = None,
    guidance_overrides: Optional[Dict] = None,
    hover_alt_m: float = 5.0,
    duration_s: float = 28.0,
    gz_visual: Optional[GazeboTargetVisual] = None,
    live_viz: bool = False,
    plot_path: Optional[Path] = None,
    fault_hook: Optional[Callable[[float, int, VirtualUMA16Sensor, PX4FlightMonitor, object], None]] = None,
    command_hook: Optional[Callable[[float, object], None]] = None,
    stop_on_abort: bool = True,
    observation_provider: Optional[
        Callable[[float, np.ndarray, Tuple[float, float, float]], Tuple[Optional[Dict], Dict]]
    ] = None,
) -> ScenarioResult:
    """Execute one full PX4 SITL + Gazebo + UMA-16 (virtual or real) + GuidanceController scenario."""
    from mavsdk.offboard import PositionNedYaw, VelocityBodyYawspeed

    if observation_provider is None and target_fn is None:
        raise ValueError("Either target_fn or observation_provider must be provided")

    config, mount_id, control_rate_hz = load_sim_guidance_config(guidance_overrides)
    await arm_and_hover_sitl(drone, att, hover_alt_m=hover_alt_m)

    # Fresh PX4FlightMonitor (reuses px4_controller.py directly!)
    monitor = PX4FlightMonitor(drone, config)
    monitor_tasks = [asyncio.create_task(t) for t in monitor.tasks()]

    sensor = (
        VirtualUMA16Sensor(sensor_config=sensor_config or AcousticSensorConfig(mount_calibration_id=mount_id))
        if observation_provider is None
        else None
    )
    guidance = GuidanceController(config)
    viz = Live3DGuidanceVisualizer(title=name, live=live_viz)

    records: List[SimRecord] = []

    try:
        # Keep holding PositionNedYaw until monitor telemetry streams are ready (especially 0.5 Hz battery)
        deadline = time.monotonic() + 5.0
        while not monitor.ready(time.monotonic()):
            if time.monotonic() >= deadline:
                raise TimeoutError("PX4FlightMonitor telemetry timeout")
            await drone.offboard.set_position_ned(PositionNedYaw(0.0, 0.0, -hover_alt_m, 0.0))
            await asyncio.sleep(0.05)

        await drone.offboard.set_velocity_body(VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0))
        t0 = time.monotonic()

        # Anchor relative altitude & start position
        monitor.start_position = monitor.local_position
        if target_fn is not None and gz_visual is not None:
            initial_target = target_fn(0.0)
            if not gz_visual.spawned:
                gz_visual.spawn(initial_target)
            else:
                gz_visual.update_pose(initial_target)

        guidance.enable()
        period_s = 1.0 / control_rate_hz
        acoustic_period_s = 0.2  # 5 Hz UMA-16 audio block cadence (matches audio_manager block_duration=0.2)
        last_audio_t = -1.0
        latest_payload: Optional[Dict] = None
        latest_diag: Dict = {}

        step_idx = 0
        while True:
            now = time.monotonic()
            elapsed = now - t0
            if elapsed >= duration_s:
                break

            tgt = target_fn(elapsed) if target_fn is not None else None
            if tgt is not None and gz_visual is not None:
                gz_visual.update_pose(tgt)

            if fault_hook is not None:
                res = fault_hook(elapsed, step_idx, sensor, monitor, drone)
                if asyncio.iscoroutine(res):
                    await res

            drone_ned = np.array([att.north_m, att.east_m, att.down_m], dtype=float)
            drone_rpy = (att.roll_deg, att.pitch_deg, att.yaw_deg)

            if observation_provider is not None:
                latest_payload, latest_diag = observation_provider(now, drone_ned, drone_rpy)
            elif last_audio_t < 0.0 or (elapsed - last_audio_t) >= acoustic_period_s:
                # Produce new virtual UMA-16 200 ms block at 5 Hz
                latest_payload, latest_diag = sensor.sample(
                    now=now,
                    drone_ned=drone_ned,
                    drone_rpy_deg=drone_rpy,
                    target_ned=tgt.ned,
                    target_active=tgt.active,
                )
                last_audio_t = elapsed

            flight = monitor.flight_state(now)
            if not flight.offboard_active:
                command = guidance.abort("offboard_lost_or_manual_override")
            elif not flight.telemetry_fresh:
                command = guidance.abort("flight_telemetry_stale")
            else:
                if latest_payload is not None:
                    validate_observation_metadata(latest_payload, mount_id)
                obs = (
                    AcousticObservation.from_mapping(latest_payload)
                    if latest_payload is not None
                    else None
                )
                command = guidance.update(now, obs, flight)

            if command.state != GuidanceState.ABORTED and flight.offboard_active:
                await drone.offboard.set_velocity_body(
                    VelocityBodyYawspeed(
                        command.forward_m_s,
                        command.right_m_s,
                        command.down_m_s,
                        command.yaw_rate_deg_s,
                    )
                )

            if command_hook is not None:
                res = command_hook(elapsed, command)
                if asyncio.iscoroutine(res):
                    await res

            records.append(
                SimRecord(
                    t_s=elapsed,
                    drone_n=att.north_m,
                    drone_e=att.east_m,
                    drone_alt=-att.down_m,
                    drone_yaw_deg=att.yaw_deg,
                    drone_vz_m_s=att.vz_m_s,
                    target_n=None if tgt is None else tgt.north_m,
                    target_e=None if tgt is None else tgt.east_m,
                    target_alt=None if tgt is None else tgt.altitude_m,
                    target_active=False if tgt is None else tgt.active,
                    ideal_az_deg=None if "ideal_azimuth_deg" not in latest_diag else float(latest_diag["ideal_azimuth_deg"]),
                    ideal_el_deg=None if "ideal_elevation_deg" not in latest_diag else float(latest_diag["ideal_elevation_deg"]),
                    true_range_m=None if "true_range_m" not in latest_diag else float(latest_diag["true_range_m"]),
                    meas_az_deg=None if latest_payload is None else latest_payload.get("azimuth_deg"),
                    meas_el_deg=None if latest_payload is None else latest_payload.get("elevation_deg"),
                    az_prominence=0.0 if latest_payload is None else float(latest_payload.get("azimuth_prominence", 0.0)),
                    el_prominence=0.0 if latest_payload is None else float(latest_payload.get("elevation_prominence", 0.0)),
                    state=command.state.value,
                    reason=command.reason,
                    cmd_forward_m_s=command.forward_m_s,
                    cmd_down_m_s=command.down_m_s,
                    cmd_yaw_rate_deg_s=command.yaw_rate_deg_s,
                    offboard_active=flight.offboard_active,
                    telemetry_fresh=flight.telemetry_fresh,
                    geofence_ok=flight.geofence_ok,
                )
            )

            if live_viz:
                viz.render(records, force=False)

            if stop_on_abort and command.state == GuidanceState.ABORTED:
                break

            step_idx += 1
            sleep_for = period_s - (time.monotonic() - now)
            if sleep_for > 0.0:
                await asyncio.sleep(sleep_for)

        viz.render(records, force=True)
        if plot_path is not None:
            viz.save(plot_path)
    finally:
        viz.close()
        for t in monitor_tasks:
            t.cancel()
        if monitor_tasks:
            await asyncio.gather(*monitor_tasks, return_exceptions=True)

    return ScenarioResult(
        name=name,
        passed=True,
        summary="",
        records=records,
        plot_path=plot_path,
        metrics={},
    )
