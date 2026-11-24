"""Pure Tkinter canvas rendering helpers and palette constants for the Tactical GUI."""

import time
import tkinter as tk
from typing import Dict, List, Optional, Tuple

import numpy as np

from frequency_id.acoustics.direction_suppression import DirectionSuppression
from frequency_id.acoustics.geometry import RigGeometry

# Color Palette
BG = "#161b22"
PANEL = "#21262d"
BORDER = "#30363d"
FG = "#e6edf3"
MUTED = "#8b949e"
ACCENT = "#58a6ff"
WARN = "#f85149"
OK = "#3fb950"
DRONE_RED = "#ff4444"
UNKNOWN_AMBER = "#e3b341"
RAY_FRONT = "#3fb950"  # Green
RAY_LEFT = "#58a6ff"   # Blue
RAY_RIGHT = "#d29922"  # Orange
SUPPRESSION_MAGENTA = "#ff66c4"
GRID_COLOR = "#212b36"



def draw_spectrum_canvas(
    c: tk.Canvas,
    freqs,
    mags,
    band_low: float,
    band_high: float,
    threshold_db: float,
    peak_freq: Optional[float] = None,
):
    """Render the narrowband FFT spectrum, threshold, and peak marker onto a canvas."""
    c.delete("all")
    w, h = c.winfo_width(), c.winfo_height()
    if w < 10:
        w, h = 400, 300

    max_f = max(4000.0, float(band_high) * 1.25)
    if mags is None or freqs is None or len(mags) == 0 or len(freqs) == 0:
        return

    x_lo = (band_low / max_f) * w
    x_hi = (band_high / max_f) * w
    c.create_rectangle(x_lo, 0, x_hi, h, fill="#1c2430", width=0)
    c.create_line(x_lo, 0, x_lo, h, fill=ACCENT, dash=(2, 2))
    c.create_line(x_hi, 0, x_hi, h, fill=ACCENT, dash=(2, 2))

    db_floor = -100.0
    db_ceil = 0.0
    step = max(1, len(freqs) // 180)
    pts = []
    for i in range(0, len(freqs), step):
        if freqs[i] > max_f:
            break
        px = (freqs[i] / max_f) * w
        db = max(db_floor, min(db_ceil, mags[i]))
        py = h - ((db - db_floor) / (db_ceil - db_floor)) * (h - 20) - 10
        pts.extend([px, py])

    if len(pts) >= 4:
        c.create_line(pts, fill=FG, width=1.5)

    py_thresh = h - ((threshold_db - db_floor) / (db_ceil - db_floor)) * (h - 20) - 10
    c.create_line(0, py_thresh, w, py_thresh, fill=WARN, dash=(4, 4), width=1)
    c.create_text(w - 65, py_thresh - 8, text=f"Thresh {threshold_db:.0f}dB", fill=WARN, font=("Monospace", 8))

    if peak_freq is not None and peak_freq <= max_f:
        px_peak = (peak_freq / max_f) * w
        c.create_line(px_peak, 0, px_peak, h, fill=WARN, width=2)
        c.create_text(px_peak + 4, 18, text=f"{peak_freq:.0f} Hz", fill=WARN, anchor="w", font=("Monospace", 9, "bold"))

    tick_interval = 1000.0 if max_f <= 7000.0 else 2000.0
    for f_tick in np.arange(tick_interval, max_f, tick_interval):
        tx = (f_tick / max_f) * w
        c.create_line(tx, h - 5, tx, h, fill=MUTED)
        c.create_text(tx, h - 12, text=f"{f_tick/1000:g}k", fill=MUTED, font=("Monospace", 7))


def draw_radar_canvas(
    c: tk.Canvas,
    geo: RigGeometry,
    map_scale: float,
    selected_ids: List[str],
    suppression_enabled: bool = False,
    suppression_bearing: float = 0.0,
    suppression_tolerance: float = 10.0,
    single_bearing: Optional[float] = None,
    single_array_id: Optional[str] = None,
    single_label: Optional[str] = None,
    tri_x: Optional[float] = None,
    tri_y: Optional[float] = None,
    tri_label: Optional[str] = None,
    drone_confirmed: bool = False,
    camera_bearing: Optional[float] = None,
    camera_elevation: Optional[float] = None,
):
    """Render the 2D top-down tactical spatial map onto a canvas."""
    c.delete("all")
    w, h = c.winfo_width(), c.winfo_height()
    if w < 10:
        w, h = 680, 360

    ox = w / 2.0
    oy = h - 100

    for r_m in (1.0, 2.0, 3.0, 4.0, 5.0, 6.0):
        r_px = r_m * map_scale
        c.create_oval(ox - r_px, oy - r_px, ox + r_px, oy + r_px, outline=GRID_COLOR, dash=(2, 4))
        c.create_text(ox + 8, oy - r_px + 8, text=f"{r_m:g}m", fill=MUTED, font=("Monospace", 7))

    c.create_line(ox, 10, ox, h - 10, fill=GRID_COLOR)
    c.create_line(10, oy, w - 10, oy, fill=GRID_COLOR)
    c.create_text(ox + 12, 20, text="FORWARD (+Y)", fill=MUTED, font=("Monospace", 8))
    c.create_text(w - 60, oy - 10, text="+X (RIGHT)", fill=MUTED, font=("Monospace", 8))

    def to_canvas(x_m, y_m):
        return ox + x_m * map_scale, oy - y_m * map_scale

    for arr_id, arr in geo.arrays.items():
        if arr_id not in selected_ids:
            continue
        cx, cy = to_canvas(arr.center[0], arr.center[1])
        col = RAY_FRONT if arr_id == "front" else (RAY_LEFT if arr_id == "left" else RAY_RIGHT)
        if arr.array_type == "uma8" and arr.mount != "vertical":
            c.create_oval(cx - 7, cy - 7, cx + 7, cy + 7, fill=col, outline="#ffffff")
            c.create_text(cx, cy + 12, text="UMA-8", fill=col, font=("Helvetica", 7, "bold"))
        elif arr.mount == "vertical":
            rad = np.deg2rad(arr.orientation_deg)
            column_axis = np.array([np.cos(rad), -np.sin(rad)])
            half_width = 0.063
            p0 = to_canvas(*(arr.center - half_width * column_axis))
            p1 = to_canvas(*(arr.center + half_width * column_axis))
            c.create_line(p0[0], p0[1], p1[0], p1[1], fill=col, width=5)
            c.create_text(cx, cy + 12, text=f"{arr.name} ({len(arr.mic_positions)} mics)", fill=col, font=("Helvetica", 7))
        else:
            c.create_rectangle(cx - 10, cy - 10, cx + 10, cy + 10, fill=col, outline="#ffffff")
            c.create_text(cx, cy + 16, text=f"{arr.name} ({len(arr.mic_positions)} mics)", fill=col, font=("Helvetica", 7))

    if suppression_enabled and "left" in geo.arrays:
        origin = to_canvas(*geo.arrays["left"].center)
        for boundary in (suppression_bearing - suppression_tolerance, suppression_bearing + suppression_tolerance):
            angle = np.deg2rad(boundary)
            endpoint = (
                origin[0] + 1.8 * map_scale * np.sin(angle),
                origin[1] - 1.8 * map_scale * np.cos(angle),
            )
            c.create_line(*origin, *endpoint, fill=SUPPRESSION_MAGENTA, width=1, dash=(3, 5))
        angle = np.deg2rad(suppression_bearing)
        endpoint = (
            origin[0] + 2.0 * map_scale * np.sin(angle),
            origin[1] - 2.0 * map_scale * np.cos(angle),
        )
        c.create_line(*origin, *endpoint, fill=SUPPRESSION_MAGENTA, width=3, dash=(8, 4))
        c.create_text(
            endpoint[0],
            endpoint[1] - 10,
            text=f"SUPPRESSED {suppression_bearing:.0f}° ±{suppression_tolerance:.0f}°",
            fill=SUPPRESSION_MAGENTA,
            font=("Helvetica", 9, "bold"),
        )

    if single_bearing is not None and single_array_id in geo.arrays:
        arr = geo.arrays[single_array_id]
        ray_color = RAY_FRONT if single_array_id == "front" else RAY_LEFT
        angle = np.deg2rad(single_bearing)
        origin = to_canvas(*arr.center)
        endpoint = to_canvas(arr.center[0] + 2.0 * np.sin(angle), arr.center[1] + 2.0 * np.cos(angle))
        c.create_line(origin[0], origin[1], endpoint[0], endpoint[1], fill=ray_color, width=3, arrow="last")
        c.create_text(
            origin[0] + 10,
            origin[1] - 16,
            text=f"{single_label or 'ARRAY'} DIRECTION ONLY",
            fill=ray_color,
            anchor="w",
            font=("Helvetica", 8, "bold"),
        )
    elif tri_x is not None and tri_y is not None:
        tx, ty = to_canvas(tri_x, tri_y)
        origin = to_canvas(0.0, 0.0)
        color = DRONE_RED if drone_confirmed else UNKNOWN_AMBER
        radius = (8.0 + 2.0 * np.sin(time.monotonic() * 6.0)) if drone_confirmed else 6.0
        c.create_line(origin[0], origin[1], tx, ty, fill=color, width=3)
        c.create_oval(tx - radius, ty - radius, tx + radius, ty + radius, fill=color, outline="#ffffff", width=2)
        c.create_text(
            tx + 12,
            ty - 8,
            text=f"{tri_label or 'TARGET'} ({tri_x:+.1f}m, {tri_y:+.1f}m)",
            fill=color,
            anchor="w",
            font=("Helvetica", 9, "bold"),
        )

    if camera_bearing is not None:
        angle = np.deg2rad(camera_bearing)
        origin = to_canvas(0.0, 0.0)
        endpoint = to_canvas(4.0 * np.sin(angle), 4.0 * np.cos(angle))
        c.create_line(origin[0], origin[1], endpoint[0], endpoint[1], fill="#c678dd", width=2, dash=(8, 4), arrow="last")
        el_text = f"EL {camera_elevation:+.1f}°, " if camera_elevation is not None else ""
        c.create_text(
            endpoint[0] + 8,
            endpoint[1],
            text=f"CAMERA {camera_bearing:.1f}° ({el_text}DIRECTION ONLY)",
            fill="#c678dd",
            anchor="w",
            font=("Helvetica", 8, "bold"),
        )


def draw_direction_3d_canvas(
    c: tk.Canvas,
    direction_vector: Optional[Tuple[float, float, float]],
    bearing_deg: Optional[float],
    elevation_deg: Optional[float],
    title: str = "3D DIRECTION",
    blocked_message: Optional[str] = None,
    empty_title: str = "WAITING",
    empty_subtitle: str = "NO 3D VECTOR",
    empty_note: str = "No direction arrow",
    arrow_color: str = DRONE_RED,
    default_size: Tuple[int, int] = (680, 260),
):
    """Render an isometric 3D unit-vector arrow and bearing/elevation readout on a canvas."""
    c.delete("all")
    w, h = c.winfo_width(), c.winfo_height()
    if w < 10:
        w, h = default_size
    c.create_text(16, 12, text=title, fill=MUTED, anchor="nw", font=("Helvetica", 9, "bold"))
    if blocked_message is not None:
        c.create_text(w / 2, h / 2, text=blocked_message, fill=MUTED, font=("Helvetica", 13, "bold"))
        return

    cx, cy = w * 0.33, h * 0.64
    scale = min(w * 0.34, h * 0.42)

    def project(vector):
        x, y, z = vector
        return cx + scale * (x - 0.65 * y), cy + scale * (0.28 * x + 0.28 * y - z)

    plane = [
        project((-1.0, -1.0, 0.0)),
        project((1.0, -1.0, 0.0)),
        project((1.0, 1.0, 0.0)),
        project((-1.0, 1.0, 0.0)),
    ]
    c.create_polygon(plane, fill="#101c26", outline="#344754", width=1)
    for step in (-0.5, 0.0, 0.5):
        c.create_line(*project((step, -1.0, 0.0)), *project((step, 1.0, 0.0)), fill="#243642")
        c.create_line(*project((-1.0, step, 0.0)), *project((1.0, step, 0.0)), fill="#243642")

    origin = project((0.0, 0.0, 0.0))
    axes = (
        ((1.08, 0.0, 0.0), "RIGHT", "#f85149", "w"),
        ((0.0, 1.08, 0.0), "FORWARD", "#3fb950", "e"),
        ((0.0, 0.0, 1.08), "UP", "#58a6ff", "n"),
    )
    for vector, label, color, anchor in axes:
        endpoint = project(vector)
        c.create_line(origin[0], origin[1], endpoint[0], endpoint[1], fill=color, width=2, arrow="last")
        c.create_text(endpoint[0], endpoint[1], text=label, fill=color, anchor=anchor, font=("Helvetica", 9, "bold"))

    status_x = w * 0.76
    if direction_vector is None or bearing_deg is None or elevation_deg is None:
        c.create_text(status_x, h * 0.38, text=empty_title, fill=UNKNOWN_AMBER, font=("Helvetica", 20, "bold"))
        c.create_text(status_x, h * 0.58, text=empty_subtitle, fill=FG, font=("Helvetica", 11, "bold"))
        c.create_text(status_x, h * 0.72, text=empty_note, fill=MUTED, font=("Helvetica", 10))
        return

    endpoint = project(direction_vector)
    ground_endpoint = project((direction_vector[0], direction_vector[1], 0.0))
    c.create_line(origin[0], origin[1], ground_endpoint[0], ground_endpoint[1], fill=UNKNOWN_AMBER, width=2)
    c.create_line(ground_endpoint[0], ground_endpoint[1], endpoint[0], endpoint[1], fill=UNKNOWN_AMBER, width=2, dash=(5, 4))
    c.create_line(origin[0], origin[1], endpoint[0], endpoint[1], fill="#2d1f3a" if arrow_color == "#c678dd" else "#7a1f1f", width=10)
    c.create_line(origin[0], origin[1], endpoint[0], endpoint[1], fill=arrow_color, width=6, arrow="last", arrowshape=(18, 22, 8))
    c.create_oval(origin[0] - 5, origin[1] - 5, origin[0] + 5, origin[1] + 5, fill=FG, outline="")

    vertical = "ABOVE" if elevation_deg > 2.0 else "BELOW" if elevation_deg < -2.0 else "LEVEL"
    bearing = bearing_deg % 360.0
    directions = (
        "FORWARD", "FORWARD RIGHT", "RIGHT", "BACK RIGHT",
        "BACK", "BACK LEFT", "LEFT", "FORWARD LEFT",
    )
    horizontal = directions[int((bearing + 22.5) // 45.0) % 8]
    c.create_text(status_x, max(38, h * 0.20), text=vertical, fill=arrow_color, font=("Helvetica", 20, "bold"))
    c.create_text(status_x, max(72, h * 0.38), text=f"{elevation_deg:+.1f}°", fill=FG, font=("Monospace", 20, "bold"))
    c.create_text(status_x, max(98, h * 0.50), text="ELEVATION", fill=MUTED, font=("Helvetica", 8, "bold"))
    c.create_text(status_x, max(128, h * 0.64), text=horizontal, fill=ACCENT, font=("Helvetica", 14, "bold"), tags=("direction-value",))
    c.create_text(status_x, max(154, h * 0.76), text=f"AZIMUTH {bearing:.1f}°", fill=FG, font=("Monospace", 10, "bold"))
    c.create_text(status_x, max(178, h * 0.87), text="Arrow points to the source", fill=MUTED, font=("Helvetica", 8))
