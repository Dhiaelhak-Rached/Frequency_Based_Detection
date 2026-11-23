"""Real-time 3D visualization, telemetry plotting, and Markdown report generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
import numpy as np


@dataclass
class SimRecord:
    t_s: float
    drone_n: float
    drone_e: float
    drone_alt: float
    drone_yaw_deg: float
    drone_vz_m_s: float
    target_n: Optional[float] = None
    target_e: Optional[float] = None
    target_alt: Optional[float] = None
    target_active: bool = False
    ideal_az_deg: Optional[float] = None
    ideal_el_deg: Optional[float] = None
    true_range_m: Optional[float] = None
    meas_az_deg: Optional[float] = None
    meas_el_deg: Optional[float] = None
    az_prominence: float = 0.0
    el_prominence: float = 0.0
    state: str = "DISABLED"
    reason: str = ""
    cmd_forward_m_s: float = 0.0
    cmd_down_m_s: float = 0.0
    cmd_yaw_rate_deg_s: float = 0.0
    offboard_active: bool = False
    telemetry_fresh: bool = False
    geofence_ok: bool = False


def format_desk_alignment_cue(latest: SimRecord) -> str:
    """Human-readable cue for manually rotating/tilting the desk UMA-16 during HIL."""
    if latest.state == "ABORTED":
        return f"ABORTED ({latest.reason})"
    if latest.state in ("DISABLED", "HOVER_SEARCH", "TARGET_LOST_HOLD"):
        return "NO ACOUSTIC LOCK: Play or aim sound toward UMA-16"

    az = latest.meas_az_deg
    el = latest.meas_el_deg
    if latest.state == "ALIGN_YAW" and az is not None:
        wrapped_az = (az + 180.0) % 360.0 - 180.0
        direction = "RIGHT" if wrapped_az > 0.0 else "LEFT"
        return f"DESK CUE: Turn UMA-16 {direction} ({abs(wrapped_az):.1f} deg to center)"
    if latest.state == "ALIGN_VERTICAL" and el is not None:
        direction = "UP" if el > 0.0 else "DOWN"
        return f"DESK CUE: Tilt UMA-16 {direction} ({abs(el):.1f} deg to level)"
    if latest.state == "FOLLOW":
        return "ALIGNED -> FOLLOWING (keep sound centered)"
    return f"STATE: {latest.state}"


STATE_COLORS = {
    "DISABLED": "#6e7681",
    "HOVER_SEARCH": "#58a6ff",
    "ALIGN_YAW": "#d29922",
    "ALIGN_VERTICAL": "#a371f7",
    "FOLLOW": "#3fb950",
    "TARGET_LOST_HOLD": "#f0883e",
    "ABORTED": "#f85149",
}


class Live3DGuidanceVisualizer:
    """Real-time 3D + telemetry dashboard (GUI when live=True, offscreen save otherwise)."""

    def __init__(self, title: str, live: bool = False):
        self.title = title
        self.live = live
        if not live:
            matplotlib.use("Agg", force=True)
        import mpl_toolkits
        local_mpl_tk = Path(matplotlib.__file__).resolve().parent.parent / "mpl_toolkits"
        if local_mpl_tk.is_dir() and str(local_mpl_tk) not in mpl_toolkits.__path__:
            mpl_toolkits.__path__.insert(0, str(local_mpl_tk))
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        self.plt = plt
        if live:
            plt.ion()
        self.fig = plt.figure(figsize=(15, 9), facecolor="#0d1117")
        self.fig.suptitle(title, color="#e6edf3", fontsize=13, fontweight="bold")
        self.ax3d = self.fig.add_subplot(2, 2, 1, projection="3d", facecolor="#161b22")
        self.ax_alt = self.fig.add_subplot(2, 2, 2, facecolor="#161b22")
        self.ax_ang = self.fig.add_subplot(2, 2, 3, facecolor="#161b22")
        self.ax_vel = self.fig.add_subplot(2, 2, 4, facecolor="#161b22")
        self._last_draw = 0.0

    def render(self, records: List[SimRecord], force: bool = False):
        if not records:
            return
        now = records[-1].t_s
        if not force and self.live and (now - self._last_draw) < 0.25:
            return
        self._last_draw = now

        ts = np.array([r.t_s for r in records])
        dn = np.array([r.drone_n for r in records])
        de = np.array([r.drone_e for r in records])
        da = np.array([r.drone_alt for r in records])
        has_target = any(
            r.target_n is not None and r.target_e is not None and r.target_alt is not None
            for r in records
        )
        tn = np.array([r.target_n if r.target_n is not None else np.nan for r in records], dtype=float)
        te = np.array([r.target_e if r.target_e is not None else np.nan for r in records], dtype=float)
        ta = np.array([r.target_alt if r.target_alt is not None else np.nan for r in records], dtype=float)

        latest = records[-1]
        if not has_target:
            self.fig.suptitle(
                f"{self.title}  |  {format_desk_alignment_cue(latest)}",
                color=STATE_COLORS.get(latest.state, "#e6edf3"),
                fontsize=12,
                fontweight="bold",
            )
        for ax in (self.ax3d, self.ax_alt, self.ax_ang, self.ax_vel):
            ax.clear()
            ax.tick_params(colors="#8b949e", labelsize=8)

        # 1. 3D Trajectory
        self.ax3d.set_facecolor("#161b22")
        self.ax3d.plot(de, dn, da, color="#58a6ff", linewidth=2.0, label="Drone Path")
        if has_target:
            self.ax3d.plot(te, tn, ta, color="#f0883e", linestyle="--", linewidth=1.5, label="Target Path")
        self.ax3d.scatter([de[-1]], [dn[-1]], [da[-1]], color="#3fb950", s=70, label="Drone")
        if has_target and np.isfinite(te[-1]) and np.isfinite(tn[-1]) and np.isfinite(ta[-1]):
            self.ax3d.scatter([te[-1]], [tn[-1]], [ta[-1]], color="#ff4444", s=80, marker="^", label="Target")
        # Nose vector
        yaw_r = np.deg2rad(latest.drone_yaw_deg)
        self.ax3d.quiver(
            de[-1], dn[-1], da[-1],
            1.5 * np.sin(yaw_r), 1.5 * np.cos(yaw_r), 0.0,
            color="#3fb950", linewidth=2.0,
        )
        self.ax3d.set_xlabel("East (m)", color="#8b949e", fontsize=8)
        self.ax3d.set_ylabel("North (m)", color="#8b949e", fontsize=8)
        self.ax3d.set_zlabel("Altitude (m)", color="#8b949e", fontsize=8)
        self.ax3d.set_title(
            f"3D Pose | State: {latest.state} ({latest.reason})",
            color=STATE_COLORS.get(latest.state, "#e6edf3"),
            fontsize=10,
        )
        self.ax3d.legend(loc="upper left", fontsize=7, facecolor="#21262d", labelcolor="#e6edf3")

        # 2. Altitude & Remaining Error
        self.ax_alt.set_facecolor("#161b22")
        self.ax_alt.plot(ts, da, color="#58a6ff", linewidth=2.0, label="Drone Alt (m)")
        if has_target:
            self.ax_alt.plot(ts, ta, color="#f0883e", linestyle="--", linewidth=2.0, label="Target Alt (m)")
            self.ax_alt.plot(ts, ta - da, color="#a371f7", linewidth=1.2, label="ΔAlt (Target - Drone)")
        self.ax_alt.axhline(0.0, color="#30363d", linewidth=0.8)
        if has_target and latest.target_alt is not None:
            alt_title = (
                f"Altitude Tracking | Drone={latest.drone_alt:.2f}m Target={latest.target_alt:.2f}m "
                f"Δ={latest.target_alt - latest.drone_alt:+.2f}m"
            )
        else:
            alt_title = f"Altitude Tracking | Drone={latest.drone_alt:.2f}m (Live UMA-16)"
        self.ax_alt.set_title(
            alt_title,
            color="#e6edf3",
            fontsize=9,
        )
        self.ax_alt.set_xlabel("Time (s)", color="#8b949e", fontsize=8)
        self.ax_alt.set_ylabel("Meters", color="#8b949e", fontsize=8)
        self.ax_alt.grid(color="#212b36", linewidth=0.6)
        self.ax_alt.legend(loc="best", fontsize=7, facecolor="#21262d", labelcolor="#e6edf3")

        # 3. Acoustic Azimuth & Elevation
        self.ax_ang.set_facecolor("#161b22")
        maz = np.array([
            ((r.meas_az_deg + 180.0) % 360.0 - 180.0) if r.meas_az_deg is not None else np.nan
            for r in records
        ])
        mel = np.array([r.meas_el_deg if r.meas_el_deg is not None else np.nan for r in records])
        has_ideal = any(r.ideal_az_deg is not None and r.ideal_el_deg is not None for r in records)
        self.ax_ang.plot(ts, maz, color="#3fb950", linewidth=1.8, label="Meas Azimuth (°)")
        self.ax_ang.plot(ts, mel, color="#a371f7", linewidth=1.8, label="Meas Elevation (°)")
        if has_ideal:
            iaz = np.array([r.ideal_az_deg if r.ideal_az_deg is not None else np.nan for r in records], dtype=float)
            iel = np.array([r.ideal_el_deg if r.ideal_el_deg is not None else np.nan for r in records], dtype=float)
            self.ax_ang.plot(ts, iaz, color="#3fb950", linestyle=":", alpha=0.6, label="Ideal Az (°)")
            self.ax_ang.plot(ts, iel, color="#a371f7", linestyle=":", alpha=0.6, label="Ideal El (°)")
        self.ax_ang.axhspan(-4.0, 4.0, color="#3fb950", alpha=0.08)
        self.ax_ang.axhline(0.0, color="#30363d", linewidth=0.8)
        range_text = (
            f"Range={latest.true_range_m:.2f}m"
            if latest.true_range_m is not None
            else f"Prom=(Az:{latest.az_prominence:.2f}, El:{latest.el_prominence:.2f})"
        )
        self.ax_ang.set_title(
            f"Acoustic Angles | Az={maz[-1]:.1f}° El={mel[-1]:.1f}° | {range_text}",
            color="#e6edf3",
            fontsize=9,
        )
        self.ax_ang.set_xlabel("Time (s)", color="#8b949e", fontsize=8)
        self.ax_ang.set_ylabel("Degrees", color="#8b949e", fontsize=8)
        self.ax_ang.grid(color="#212b36", linewidth=0.6)
        self.ax_ang.legend(loc="best", fontsize=7, facecolor="#21262d", labelcolor="#e6edf3")

        # 4. Commands, Vertical Speed, & Safety
        self.ax_vel.set_facecolor("#161b22")
        self.ax_vel.plot(ts, [r.cmd_forward_m_s for r in records], color="#3fb950", linewidth=1.8, label="Cmd Forward (m/s)")
        self.ax_vel.plot(ts, [-r.cmd_down_m_s for r in records], color="#58a6ff", linewidth=1.8, label="Cmd Climb (-down m/s)")
        self.ax_vel.plot(ts, [r.drone_vz_m_s for r in records], color="#e3b341", linestyle="--", linewidth=1.2, label="Actual Climb (m/s)")
        safety = (
            f"OFFBOARD={latest.offboard_active} GEO={latest.geofence_ok} "
            f"FRESH={latest.telemetry_fresh}"
        )
        self.ax_vel.set_title(f"Velocities & Safety | {safety}", color="#e6edf3", fontsize=9)
        self.ax_vel.set_xlabel("Time (s)", color="#8b949e", fontsize=8)
        self.ax_vel.set_ylabel("m/s", color="#8b949e", fontsize=8)
        self.ax_vel.grid(color="#212b36", linewidth=0.6)
        self.ax_vel.legend(loc="best", fontsize=7, facecolor="#21262d", labelcolor="#e6edf3")

        self.fig.tight_layout(rect=[0, 0, 1, 0.95])
        if self.live:
            self.plt.pause(0.001)

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.fig.savefig(path, dpi=150, facecolor=self.fig.get_facecolor())

    def close(self):
        self.plt.close(self.fig)
