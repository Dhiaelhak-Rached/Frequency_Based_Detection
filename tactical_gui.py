#!/usr/bin/env python3
"""2D Tactical Radar GUI for frequency-locked dual-array acoustic triangulation.

Displays real-time FFT spectrum, live drone telemetry, and a top-down 2D
tactical map plotting the physical rig and one validated bearing to the
triangulated (X, Y) target position.
"""

import json
from pathlib import Path
import tkinter as tk
from tkinter import ttk
from typing import Dict

import numpy as np

from geometry import RigGeometry
from detector import FrequencyDetector
from doa_engine import ArrayDOA
from triangulation import Triangulator
from audio_manager import MultiDeviceAudioManager

# Color Palette
BG = "#161b22"
PANEL = "#21262d"
BORDER = "#30363d"
FG = "#e6edf3"
MUTED = "#8b949e"
ACCENT = "#58a6ff"
WARN = "#f85149"
OK = "#3fb950"
RAY_FRONT = "#3fb950"  # Green
RAY_LEFT = "#58a6ff"   # Blue
RAY_RIGHT = "#d29922"  # Orange
GRID_COLOR = "#212b36"


class TacticalGUI:
    def __init__(self, root: tk.Tk, geometry: RigGeometry, simulate: bool = False):
        self.root = root
        self.geo = geometry
        self.simulate = simulate

        root.title("Acoustic Drone Tactical Radar — Dual-Array Triangulation")
        root.configure(bg=BG)
        root.geometry("1180x760")
        root.minsize(950, 650)
        root.protocol("WM_DELETE_WINDOW", self.on_close)

        # Core Engines
        self.detector = FrequencyDetector(
            sample_rate=geometry.sample_rate,
            block_duration=0.2,
            average_blocks=4,
            threshold_dbfs=-65.0,
        )
        self.doa_engines: Dict[str, ArrayDOA] = {}
        for arr_id, arr in self.geo.arrays.items():
            self.doa_engines[arr_id] = ArrayDOA(
                array_id=arr_id,
                mic_positions_local=arr.local_mics,
                orientation_deg=arr.orientation_deg,
                sample_rate=geometry.sample_rate,
                mount=arr.mount,
                freq_lo=200.0,
                freq_hi=16000.0,
            )
        self.triangulator = Triangulator(geometry=geometry, min_prominence=1.4)
        self.audio_mgr = MultiDeviceAudioManager(
            geometry=geometry,
            sample_rate=geometry.sample_rate,
            block_duration=0.2,
            simulate=simulate,
        )

        self.is_active = False
        self.map_scale = 90.0  # pixels per meter
        self.map_center_offset_y = 220  # offset origin downwards on canvas

        self._build_ui()
        self.root.after(100, self._update_loop)

    def _build_ui(self):
        # 1. Top Header & Telemetry Bar
        top_frame = tk.Frame(self.root, bg=PANEL, highlightbackground=BORDER, highlightthickness=1)
        top_frame.pack(fill="x", padx=10, pady=(10, 6))

        self.status_lbl = tk.Label(
            top_frame,
            text="STANDBY" if not self.simulate else "SIMULATION ACTIVE",
            font=("Helvetica", 14, "bold"),
            bg="#2d333b",
            fg=OK if self.simulate else MUTED,
            padx=16,
            pady=8,
        )
        self.status_lbl.pack(side="left", padx=8, pady=8)

        # Telemetry metrics
        self.lbl_freq = self._create_metric(top_frame, "DETECTED FREQ", "---.- Hz")
        self.lbl_level = self._create_metric(top_frame, "SIGNAL LEVEL", "---.- dBFS")
        self.lbl_range = self._create_metric(top_frame, "RANGE", "-.-- m")
        self.lbl_bearing = self._create_metric(top_frame, "BEARING", "---.-°")
        self.lbl_coords = self._create_metric(top_frame, "COORDINATES (X, Y)", "X: -.--  Y: -.--")

        # 2. Main Visual Body (Spectrum on Left, Radar Map on Right)
        body = tk.Frame(self.root, bg=BG)
        body.pack(fill="both", expand=True, padx=10, pady=4)

        # Left: Spectrum Canvas
        spec_frame = tk.LabelFrame(body, text=" Real-Time Spectrum (Narrowband Filter) ", bg=BG, fg=FG,
                                   font=("Helvetica", 10, "bold"), highlightbackground=BORDER)
        spec_frame.pack(side="left", fill="both", expand=False, padx=(0, 6))
        self.canvas_spec = tk.Canvas(spec_frame, width=380, height=480, bg="#0d1117", highlightthickness=0)
        self.canvas_spec.pack(fill="both", expand=True, padx=4, pady=4)

        # Right: 2D Tactical Radar Canvas
        radar_frame = tk.LabelFrame(body, text=" 2D Tactical Spatial Map (Top-Down Triangulation) ", bg=BG, fg=FG,
                                    font=("Helvetica", 10, "bold"), highlightbackground=BORDER)
        radar_frame.pack(side="right", fill="both", expand=True)
        self.canvas_radar = tk.Canvas(radar_frame, width=680, height=480, bg="#0d1117", highlightthickness=0)
        self.canvas_radar.pack(fill="both", expand=True, padx=4, pady=4)

        # 3. Bottom Controls Panel
        ctrl = tk.Frame(self.root, bg=PANEL, highlightbackground=BORDER, highlightthickness=1)
        ctrl.pack(fill="x", padx=10, pady=(6, 10))

        # Top row of controls: Audio Device Mapping
        dev_row = tk.Frame(ctrl, bg=PANEL)
        dev_row.pack(fill="x", padx=8, pady=(6, 2))

        tk.Label(dev_row, text="Mic Device IDs:", font=("Helvetica", 9, "bold"), bg=PANEL, fg=ACCENT).pack(side="left", padx=(0, 8))

        tk.Label(dev_row, text="Front (UMA-8):", bg=PANEL, fg=FG).pack(side="left", padx=(4, 2))
        default_front = str(self.geo.cfg.get("arrays", {}).get("front", {}).get("device_index", 4))
        self.var_dev_front = tk.StringVar(value=default_front)
        tk.Entry(dev_row, textvariable=self.var_dev_front, width=4, bg="#0d1117", fg=FG, insertbackground=FG).pack(side="left")

        tk.Label(dev_row, text="Left (UMA-16):", bg=PANEL, fg=FG).pack(side="left", padx=(10, 2))
        default_left = str(self.geo.cfg.get("arrays", {}).get("left", {}).get("device_index", 1))
        self.var_dev_left = tk.StringVar(value=default_left)
        tk.Entry(dev_row, textvariable=self.var_dev_left, width=4, bg="#0d1117", fg=FG, insertbackground=FG).pack(side="left")


        # Bottom row of controls: Filter settings & Action buttons
        act_row = tk.Frame(ctrl, bg=PANEL)
        act_row.pack(fill="x", padx=8, pady=(4, 6))

        self.btn_toggle = tk.Button(
            act_row, text="START LISTENING", font=("Helvetica", 11, "bold"),
            bg="#238636", fg="#ffffff", activebackground="#2ea043",
            padx=14, pady=4, command=self.on_toggle_listening, relief="flat",
        )
        self.btn_toggle.pack(side="left", padx=(0, 10))

        # Band Low
        tk.Label(act_row, text="Band Low (Hz):", font=("Helvetica", 9, "bold"), bg=PANEL, fg=ACCENT).pack(side="left", padx=(4, 2))
        self.var_low = tk.StringVar(value=str(self.geo.cfg["default_detection"]["target_low_hz"]))
        entry_low = tk.Entry(act_row, textvariable=self.var_low, width=7, font=("Monospace", 9, "bold"), bg="#0d1117", fg=FG, insertbackground=FG)
        entry_low.pack(side="left")
        entry_low.bind("<Return>", lambda e: self.on_apply_band())

        # Band High
        tk.Label(act_row, text="Band High (Hz):", font=("Helvetica", 9, "bold"), bg=PANEL, fg=ACCENT).pack(side="left", padx=(10, 2))
        self.var_high = tk.StringVar(value=str(self.geo.cfg["default_detection"]["target_high_hz"]))
        entry_high = tk.Entry(act_row, textvariable=self.var_high, width=7, font=("Monospace", 9, "bold"), bg="#0d1117", fg=FG, insertbackground=FG)
        entry_high.pack(side="left")
        entry_high.bind("<Return>", lambda e: self.on_apply_band())

        # Apply Band button
        btn_apply = tk.Button(
            act_row, text="Apply Band", font=("Helvetica", 9, "bold"),
            bg="#21262d", fg=ACCENT, activebackground="#30363d", activeforeground=FG,
            padx=6, pady=2, command=self.on_apply_band, relief="solid", bd=1
        )
        btn_apply.pack(side="left", padx=(6, 12))

        # Threshold
        tk.Label(act_row, text="Thresh (dBFS):", bg=PANEL, fg=FG).pack(side="left", padx=(4, 2))
        self.var_thresh = tk.StringVar(value=str(self.geo.cfg["default_detection"]["threshold_dbfs"]))
        tk.Entry(act_row, textvariable=self.var_thresh, width=5, bg="#0d1117", fg=FG, insertbackground=FG).pack(side="left")

        # Averaging
        tk.Label(act_row, text="Avg:", bg=PANEL, fg=FG).pack(side="left", padx=(8, 2))
        self.var_avg = tk.StringVar(value=str(self.geo.cfg["default_detection"]["average_blocks"]))
        tk.Entry(act_row, textvariable=self.var_avg, width=3, bg="#0d1117", fg=FG, insertbackground=FG).pack(side="left")

        # Simulation Toggle
        self.var_sim = tk.BooleanVar(value=self.simulate)
        chk_sim = tk.Checkbutton(
            act_row, text="Simulation Mode", variable=self.var_sim, bg=PANEL, fg=FG,
            selectcolor="#0d1117", activebackground=PANEL, activeforeground=FG,
            command=self.on_toggle_sim,
        )
        chk_sim.pack(side="left", padx=(14, 8))

        self.var_front_only = tk.BooleanVar(value=False)
        chk_front = tk.Checkbutton(
            act_row, text="UMA-8 Only", variable=self.var_front_only, bg=PANEL, fg=FG,
            selectcolor="#0d1117", activebackground=PANEL, activeforeground=FG,
            command=lambda: self.on_array_mode_change("front"),
        )
        chk_front.pack(side="left", padx=(4, 8))

        self.var_left_only = tk.BooleanVar(value=False)
        chk_left = tk.Checkbutton(
            act_row, text="UMA-16 Only", variable=self.var_left_only, bg=PANEL, fg=FG,
            selectcolor="#0d1117", activebackground=PANEL, activeforeground=FG,
            command=lambda: self.on_array_mode_change("left"),
        )
        chk_left.pack(side="left", padx=(4, 8))

        # Scale slider
        tk.Label(act_row, text="Zoom:", bg=PANEL, fg=FG).pack(side="right", padx=(4, 2))
        self.slider_zoom = tk.Scale(
            act_row, from_=30, to=200, orient="horizontal", showvalue=0, bg=PANEL,
            troughcolor="#0d1117", highlightthickness=0, length=80, command=self.on_zoom_change
        )
        self.slider_zoom.set(int(self.map_scale))
        self.slider_zoom.pack(side="right", padx=4)

    def _create_metric(self, parent: tk.Frame, title: str, init_val: str) -> tk.Label:
        box = tk.Frame(parent, bg=PANEL)
        box.pack(side="left", padx=12, pady=4)
        tk.Label(box, text=title, font=("Helvetica", 8), bg=PANEL, fg=MUTED).pack(anchor="w")
        val_lbl = tk.Label(box, text=init_val, font=("Monospace", 12, "bold"), bg=PANEL, fg=FG)
        val_lbl.pack(anchor="w")
        return val_lbl

    def on_zoom_change(self, val):
        self.map_scale = float(val)

    def on_toggle_sim(self):
        self.simulate = self.var_sim.get()
        self.audio_mgr.simulate = self.simulate
        if self.simulate:
            self.status_lbl.configure(text="SIMULATION ACTIVE", fg=OK)
        else:
            self.status_lbl.configure(text="STANDBY" if not self.is_active else "LISTENING", fg=FG)

    def _device_map(self):
        return {
            "front": self.var_dev_front.get().strip(),
            "left": self.var_dev_left.get().strip(),
        }

    def _selected_array_ids(self):
        if self.var_front_only.get():
            return ["front"]
        if self.var_left_only.get():
            return ["left"]
        return list(self.geo.arrays)

    def on_array_mode_change(self, selected):
        if selected == "front" and self.var_front_only.get():
            self.var_left_only.set(False)
        elif selected == "left" and self.var_left_only.get():
            self.var_front_only.set(False)
        self._clear_target_telemetry()
        for doa_eng in self.doa_engines.values():
            doa_eng.reset()
        self.audio_mgr.enabled_arrays = set(self._selected_array_ids())
        if self.is_active:
            self.audio_mgr.stop()
            self.audio_mgr.start(device_map=self._device_map(), array_ids=self._selected_array_ids())
            self.status_lbl.configure(text="SEARCHING...", fg=ACCENT)

    def on_toggle_listening(self):
        if not self.is_active:
            try:
                low = float(self.var_low.get())
                high = float(self.var_high.get())
                thresh = float(self.var_thresh.get())
                avg = int(self.var_avg.get())
            except ValueError:
                return

            self.detector.set_parameters(threshold_dbfs=thresh, average_blocks=avg)
            self.save_config()
            self.audio_mgr.start(device_map=self._device_map(), array_ids=self._selected_array_ids())
            self.is_active = True
            self.btn_toggle.configure(text="STOP", bg=WARN, activebackground="#da3633")
            self.status_lbl.configure(text="SEARCHING...", fg=ACCENT)
        else:
            self.audio_mgr.stop()
            self.is_active = False
            self.btn_toggle.configure(text="START LISTENING", bg="#238636", activebackground="#2ea043")
            self.status_lbl.configure(text="STANDBY", fg=MUTED)

    def on_apply_band(self):
        try:
            low = float(self.var_low.get())
            high = float(self.var_high.get())
            thresh = float(self.var_thresh.get())
            avg = int(self.var_avg.get())
            self.detector.set_parameters(threshold_dbfs=thresh, average_blocks=avg)
            self.save_config()
            self.status_lbl.configure(text=f"BAND UPDATED: {low:g}-{high:g} Hz", fg=ACCENT)
        except ValueError:
            pass

    def save_config(self):
        try:
            self.geo.cfg["default_detection"]["target_low_hz"] = float(self.var_low.get())
            self.geo.cfg["default_detection"]["target_high_hz"] = float(self.var_high.get())
            self.geo.cfg["default_detection"]["threshold_dbfs"] = float(self.var_thresh.get())
            self.geo.cfg["default_detection"]["average_blocks"] = int(self.var_avg.get())
            if "front" in self.geo.cfg.get("arrays", {}):
                self.geo.cfg["arrays"]["front"]["device_index"] = int(self.var_dev_front.get().strip())
            if "left" in self.geo.cfg.get("arrays", {}):
                self.geo.cfg["arrays"]["left"]["device_index"] = int(self.var_dev_left.get().strip())
            with open(self.geo.config_path, "w") as f:
                json.dump(self.geo.cfg, f, indent=2)
        except Exception:
            pass

    def _update_loop(self):
        if self.is_active or self.simulate:
            self._process_frame()
        self.root.after(100, self._update_loop)

    def _process_frame(self):
        try:
            low = float(self.var_low.get())
            high = float(self.var_high.get())
        except ValueError:
            low, high = 900.0, 1100.0

        blocks = self.audio_mgr.get_latest_blocks()
        if not blocks:
            return

        # 1. Frequency Detection (on front array ch0 or first available block)
        det_block = blocks.get("front", next(iter(blocks.values())))
        det_ch = det_block[:, 0]
        det_res = self.detector.process(det_ch, low, high)

        # 2. Narrowband DOA for each selected array
        bearings = {}
        prominences = {}
        doa_errors = {}
        doa_results = {}
        selected_ids = self._selected_array_ids()
        for arr_id in selected_ids:
            if arr_id in blocks:
                doa_eng = self.doa_engines[arr_id]
                target_f = det_res.peak_freq if det_res.detected else None
                res = doa_eng.estimate(blocks[arr_id], target_freq=target_f, band=(low, high))
                doa_results[arr_id] = res
                bearings[arr_id] = res.global_angle_deg
                prominences[arr_id] = res.prominence
                doa_errors[arr_id] = res.error

        single_array_id = selected_ids[0] if len(selected_ids) == 1 else None
        single_doa = doa_results.get(single_array_id) if single_array_id else None
        if single_doa is not None or not det_res.detected:
            tri_res = self.triangulator.triangulate({}, {})
        else:
            tri_res = self.triangulator.triangulate(
                bearings,
                prominences,
                stable={arr_id: res.stable for arr_id, res in doa_results.items()},
            )

        # 3. Update Telemetry & UI
        self._update_telemetry(det_res, tri_res, doa_errors, single_doa, single_array_id)
        self._draw_spectrum(det_res)
        self._draw_tactical_map(tri_res, single_doa if det_res.detected else None, single_array_id)

    def _clear_target_telemetry(self):
        self.lbl_range.configure(text="-.-- m", fg=MUTED)
        self.lbl_bearing.configure(text="---.-°", fg=MUTED)
        self.lbl_coords.configure(text="X: -.--  Y: -.--", fg=MUTED)

    def _update_telemetry(self, det: FrequencyDetector, tri: Triangulator, doa_errors=None, single_doa=None, single_array_id=None):
        if det.peak_freq is not None:
            self.lbl_freq.configure(text=f"{det.peak_freq:6.1f} Hz", fg=WARN if det.detected else FG)
        else:
            self.lbl_freq.configure(text="---.- Hz", fg=FG)

        if det.peak_db is not None:
            self.lbl_level.configure(text=f"{det.peak_db:5.1f} dBFS", fg=WARN if det.detected else FG)

        if single_doa is not None:
            array_label = "UMA-8" if single_array_id == "front" else "UMA-16"
            if det.detected and single_doa.global_angle_deg is not None and (single_doa.prominence >= 1.4 or single_doa.stable):
                self.status_lbl.configure(text=f"{array_label} ONLY / BEARING LOCK", fg=OK, bg="#1a2733")
                self.lbl_range.configure(text="-.-- m", fg=MUTED)
                self.lbl_bearing.configure(text=f"{single_doa.global_angle_deg:5.1f}°", fg=OK)
                self.lbl_coords.configure(text="DIRECTION ONLY", fg=MUTED)
            elif det.detected:
                if single_doa.error == "spatial_aliasing":
                    max_safe_freq = self.doa_engines[single_array_id].max_direction_freq_hz
                    status = f"FREQ OK / DOA BLOCKED ABOVE {max_safe_freq:.0f} Hz"
                else:
                    status = f"FREQ DETECTED (WAITING {array_label} BEARING)"
                self.status_lbl.configure(text=status, fg=WARN, bg="#2b2010")
                self._clear_target_telemetry()
            else:
                self.status_lbl.configure(text=f"{array_label} ONLY / SCANNING", fg=ACCENT, bg="#1a2733")
                self._clear_target_telemetry()
            return

        if det.detected:
            if tri.success and tri.x is not None and tri.y is not None:
                self.status_lbl.configure(text="TARGET LOCKED", fg=WARN, bg="#3b1d1d")
                self.lbl_range.configure(text=f"{tri.range_m:4.2f} m", fg=OK)
                self.lbl_bearing.configure(text=f"{tri.bearing_deg:5.1f}°", fg=OK)
                self.lbl_coords.configure(text=f"X:{tri.x:+4.2f}m  Y:{tri.y:+4.2f}m", fg=OK)
            else:
                if doa_errors and all(err == "spatial_aliasing" for err in doa_errors.values()):
                    max_safe_freq = min(self.doa_engines[arr_id].max_direction_freq_hz for arr_id in doa_errors)
                    status = f"FREQ OK / DOA BLOCKED ABOVE {max_safe_freq:.0f} Hz"
                else:
                    status = "FREQ DETECTED (WAITING VALID TRIANGULATION)"
                self.status_lbl.configure(text=status, fg=WARN, bg="#2b2010")
                self._clear_target_telemetry()
        else:
            self.status_lbl.configure(text="SCANNING", fg=ACCENT, bg="#1a2733")
            self._clear_target_telemetry()

    def _draw_spectrum(self, det):
        c = self.canvas_spec
        c.delete("all")
        w, h = c.winfo_width(), c.winfo_height()
        if w < 10:
            w, h = 380, 480

        # Dynamic frequency scale based on user's target band
        max_f = max(4000.0, float(det.band[1]) * 1.25)
        mags = det.mags
        freqs = det.freqs
        if mags is None or len(mags) == 0:
            return

        # Target band highlight
        x_lo = (det.band[0] / max_f) * w
        x_hi = (det.band[1] / max_f) * w
        c.create_rectangle(x_lo, 0, x_hi, h, fill="#1c2430", width=0)
        c.create_line(x_lo, 0, x_lo, h, fill=ACCENT, dash=(2, 2))
        c.create_line(x_hi, 0, x_hi, h, fill=ACCENT, dash=(2, 2))

        # Spectrum line
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

        # Threshold line
        py_thresh = h - ((det.threshold_db - db_floor) / (db_ceil - db_floor)) * (h - 20) - 10
        c.create_line(0, py_thresh, w, py_thresh, fill=WARN, dash=(4, 4), width=1)
        c.create_text(w - 65, py_thresh - 8, text=f"Thresh {det.threshold_db:.0f}dB", fill=WARN, font=("Monospace", 8))

        # Peak marker
        if det.peak_freq is not None and det.peak_freq <= max_f:
            px_peak = (det.peak_freq / max_f) * w
            c.create_line(px_peak, 0, px_peak, h, fill=WARN, width=2)
            c.create_text(px_peak + 4, 18, text=f"{det.peak_freq:.0f} Hz", fill=WARN, anchor="w", font=("Monospace", 9, "bold"))

        # Axis ticks
        tick_interval = 1000.0 if max_f <= 7000.0 else 2000.0
        for f_tick in np.arange(tick_interval, max_f, tick_interval):
            tx = (f_tick / max_f) * w
            c.create_line(tx, h - 5, tx, h, fill=MUTED)
            c.create_text(tx, h - 12, text=f"{f_tick/1000:g}k", fill=MUTED, font=("Monospace", 7))

    def _draw_tactical_map(self, tri, single_doa=None, single_array_id=None):
        c = self.canvas_radar
        c.delete("all")
        w, h = c.winfo_width(), c.winfo_height()
        if w < 10:
            w, h = 680, 480

        # Origin on canvas (rig position)
        ox = w / 2.0
        oy = h - 100

        # Radar Grid Rings
        for r_m in (1.0, 2.0, 3.0, 4.0, 5.0, 6.0):
            r_px = r_m * self.map_scale
            c.create_oval(ox - r_px, oy - r_px, ox + r_px, oy + r_px, outline=GRID_COLOR, dash=(2, 4))
            c.create_text(ox + 8, oy - r_px + 8, text=f"{r_m:g}m", fill=MUTED, font=("Monospace", 7))

        # Coordinate axes
        c.create_line(ox, 10, ox, h - 10, fill=GRID_COLOR)
        c.create_line(10, oy, w - 10, oy, fill=GRID_COLOR)
        c.create_text(ox + 12, 20, text="FORWARD (+Y)", fill=MUTED, font=("Monospace", 8))
        c.create_text(w - 60, oy - 10, text="+X (RIGHT)", fill=MUTED, font=("Monospace", 8))

        # Helper to convert metric (x, y) to canvas (px, py)
        def to_canvas(x_m, y_m):
            return ox + x_m * self.map_scale, oy - y_m * self.map_scale

        # Draw the selected array geometry
        for arr_id, arr in self.geo.arrays.items():
            if arr_id not in self._selected_array_ids():
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

        if single_doa is not None and single_doa.global_angle_deg is not None and (single_doa.prominence >= 1.4 or single_doa.stable):
            arr = self.geo.arrays[single_array_id]
            ray_color = RAY_FRONT if single_array_id == "front" else RAY_LEFT
            array_label = "UMA-8" if single_array_id == "front" else "UMA-16"
            angle = np.deg2rad(single_doa.global_angle_deg)
            origin = to_canvas(*arr.center)
            endpoint = to_canvas(
                arr.center[0] + 2.0 * np.sin(angle),
                arr.center[1] + 2.0 * np.cos(angle),
            )
            c.create_line(origin[0], origin[1], endpoint[0], endpoint[1], fill=ray_color, width=3, arrow="last")
            c.create_text(origin[0] + 10, origin[1] - 16, text=f"{array_label} DIRECTION ONLY",
                          fill=ray_color, anchor="w", font=("Helvetica", 8, "bold"))
        elif tri.success and tri.x is not None and tri.y is not None:
            tx, ty = to_canvas(tri.x, tri.y)
            origin = to_canvas(0.0, 0.0)
            c.create_line(origin[0], origin[1], tx, ty, fill=WARN, width=3)
            c.create_oval(tx - 6, ty - 6, tx + 6, ty + 6, fill=WARN, outline="#ffffff", width=2)
            c.create_text(tx + 12, ty - 8, text=f"TARGET ({tri.x:+.1f}m, {tri.y:+.1f}m)",
                          fill=WARN, anchor="w", font=("Helvetica", 9, "bold"))

    def on_close(self):
        self.audio_mgr.stop()
        self.root.destroy()
