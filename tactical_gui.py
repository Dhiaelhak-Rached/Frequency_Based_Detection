#!/usr/bin/env python3
"""Tactical GUI for acoustic tracking and independent YOLO camera detection.

Displays the real-time FFT spectrum, selectable-camera annotated video, drone
telemetry, and the top-down acoustic bearing or triangulated target position.
"""

from dataclasses import replace
import base64
import io
import json
import os
import queue
import re
import shlex
import threading
import time
import tkinter as tk
from tkinter import scrolledtext, ttk
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageTk

import jetson_ctl

from geometry import RigGeometry
from vision_detector import discover_cameras, normalize_camera_source, validate_horizontal_fov
from detector import DetectionResult, FrequencyDetector
from direction_suppression import DirectionSuppression, displayable_doa, select_front_only_target
from doa_engine import ArrayDOA
from triangulation import TriangulationResult, Triangulator
from audio_manager import MultiDeviceAudioManager
from target_state_machine import StateMachineConfig, TargetStateMachine, TrackingMode

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


class TacticalGUI:
    def __init__(
        self,
        root: tk.Tk,
        geometry: RigGeometry,
        simulate: bool = False,
        classifier=None,
        require_drone: bool = False,
        vision=None,
        suppression=None,
    ):
        self.root = root
        self.geo = geometry
        self.simulate = simulate
        self.classifier = classifier
        self.vision = vision
        self.suppression = suppression or DirectionSuppression.from_config(
            geometry.cfg.get("uma16_front_only", {}).get("direction_suppression", {})
        )
        self.vision_result = None
        self.camera_photo = None
        self.classification = None
        self.signal_detected = False
        self.classifier_hop = max(
            0.05,
            float(geometry.cfg.get("classifier", {}).get("hop_seconds", 0.2)),
        )
        self.last_classification = 0.0
        self.require_drone = require_drone
        self.state_machine = TargetStateMachine(
            StateMachineConfig(require_drone_for_audio=require_drone)
        )
        self.fused_state = None

        root.title("Acoustic + Camera Drone Tactical Radar")
        root.configure(bg=BG)
        root.geometry("1360x1020")
        root.minsize(1100, 940)
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
            mic_heights_local=arr.mic_heights_local,
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
        if self.vision is not None and self.var_vision_enabled.get():
            self.vision.start()
        self.root.after(100, self._update_loop)

    def _build_ui(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TNotebook", background=BG, borderwidth=0)
        style.configure(
            "TNotebook.Tab",
            background=PANEL,
            foreground=MUTED,
            padding=[14, 6],
            font=("Helvetica", 10, "bold"),
        )
        style.map(
            "TNotebook.Tab",
            background=[("selected", "#1f6feb")],
            foreground=[("selected", "#ffffff")],
        )

        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True)

        self.local_tab = tk.Frame(self.notebook, bg=BG)
        self.notebook.add(self.local_tab, text=" Tactical Radar (Local) ")

        # 1. Top Header & Telemetry Bar
        top_frame = tk.Frame(self.local_tab, bg=PANEL, highlightbackground=BORDER, highlightthickness=1)
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
        self.lbl_classification = self._create_metric(
            top_frame,
            "CLASSIFICATION",
            "STANDBY" if self.classifier is not None else "OFF",
        )
        self.lbl_vision = self._create_metric(
            top_frame,
            "CAMERA",
            "CONNECTING" if self.vision is not None else "OFF",
        )
        self.lbl_tracker_mode = self._create_metric(top_frame, "ACTIVE TRACKER", "SEARCHING")
        self.lbl_range = self._create_metric(top_frame, "RANGE", "-.-- m")
        self.lbl_bearing = self._create_metric(top_frame, "BEARING", "---.-°")
        self.lbl_elevation = self._create_metric(top_frame, "ELEVATION", "---.-°")
        self.lbl_coords = self._create_metric(top_frame, "COORDINATES (X, Y)", "X: -.--  Y: -.--")

        # 2. Main Visual Body (Spectrum on Left, Radar Map on Right)
        body = tk.Frame(self.local_tab, bg=BG)
        body.pack(fill="both", expand=True, padx=10, pady=4)

        left_visual = tk.Frame(body, bg=BG)
        left_visual.pack(side="left", fill="both", expand=False, padx=(0, 6))
        spec_frame = tk.LabelFrame(left_visual, text=" Real-Time Spectrum (Narrowband Filter) ", bg=BG, fg=FG,
                                   font=("Helvetica", 10, "bold"), highlightbackground=BORDER)
        spec_frame.pack(side="top", fill="both", expand=True)
        self.canvas_spec = tk.Canvas(spec_frame, width=400, height=300, bg="#0d1117", highlightthickness=0)
        self.canvas_spec.pack(fill="both", expand=True, padx=4, pady=4)

        camera_frame = tk.LabelFrame(left_visual, text=" YOLO Camera Detection ", bg=BG, fg=FG,
                                     font=("Helvetica", 10, "bold"), highlightbackground=BORDER)
        camera_frame.pack(side="bottom", fill="both", expand=False, pady=(6, 0))
        self.camera_video_lbl = tk.Label(
            camera_frame,
            text="NO CAMERA FRAME",
            width=48,
            height=11,
            bg="#0d1117",
            fg=MUTED,
        )
        self.camera_video_lbl.pack(fill="both", expand=True, padx=4, pady=4)

        right_visual = tk.Frame(body, bg=BG)
        right_visual.pack(side="right", fill="both", expand=True)
        radar_frame = tk.LabelFrame(right_visual, text=" 2D Tactical Spatial Map (Top-Down Triangulation) ", bg=BG, fg=FG,
                                    font=("Helvetica", 10, "bold"), highlightbackground=BORDER)
        radar_frame.pack(side="top", fill="both", expand=True)
        self.canvas_radar = tk.Canvas(radar_frame, width=680, height=360, bg="#0d1117", highlightthickness=0)
        self.canvas_radar.pack(fill="both", expand=True, padx=4, pady=4)

        direction_frame = tk.LabelFrame(right_visual, text=" SOUND DIRECTION — CLEAR 3D VIEW ", bg=BG, fg=FG,
                                        font=("Helvetica", 10, "bold"), highlightbackground=BORDER)
        direction_frame.pack(side="bottom", fill="x", pady=(6, 0))
        self.canvas_direction = tk.Canvas(direction_frame, width=680, height=260, bg="#0d1117", highlightthickness=0)
        self.canvas_direction.pack(fill="both", expand=True, padx=4, pady=4)

        # 3. Actions and organized settings
        ctrl = tk.Frame(self.local_tab, bg=PANEL, highlightbackground=BORDER, highlightthickness=1)
        ctrl.pack(fill="x", padx=10, pady=(6, 10))

        action_row = tk.Frame(ctrl, bg=PANEL)
        action_row.pack(fill="x", padx=8, pady=(6, 4))
        self.btn_toggle = tk.Button(
            action_row, text="START LISTENING", font=("Helvetica", 11, "bold"),
            bg="#238636", fg="#ffffff", activebackground="#2ea043",
            padx=14, pady=4, command=self.on_toggle_listening, relief="flat",
        )
        self.btn_toggle.pack(side="left", padx=(4, 12))
        self.classifier_status_lbl = tk.Label(
            action_row,
            text="CLASSIFIER ACTIVE" if self.classifier is not None else "CLASSIFIER OFF",
            font=("Helvetica", 9, "bold"),
            bg=PANEL,
            fg=OK if self.classifier is not None else MUTED,
        )
        self.classifier_status_lbl.pack(side="left", padx=4)
        tk.Label(action_row, text="Map Zoom:", bg=PANEL, fg=FG).pack(side="right", padx=(4, 2))
        self.slider_zoom = tk.Scale(
            action_row, from_=30, to=200, orient="horizontal", showvalue=True, bg=PANEL,
            fg=FG, troughcolor="#0d1117", highlightthickness=0, length=110,
            command=self.on_zoom_change,
        )
        self.slider_zoom.set(int(self.map_scale))
        self.slider_zoom.pack(side="right", padx=4)

        settings = tk.LabelFrame(
            ctrl,
            text=" Settings ",
            bg=PANEL,
            fg=ACCENT,
            font=("Helvetica", 10, "bold"),
            highlightbackground=BORDER,
        )
        settings.pack(fill="x", padx=8, pady=(2, 8))
        settings.columnconfigure(0, weight=1)
        settings.columnconfigure(1, weight=2)
        settings.columnconfigure(2, weight=1)

        audio_settings = tk.LabelFrame(settings, text=" Audio & Detection ", bg=PANEL, fg=FG, padx=6, pady=4)
        audio_settings.grid(row=0, column=0, sticky="nsew", padx=(4, 6), pady=4)
        mic_row = tk.Frame(audio_settings, bg=PANEL)
        mic_row.pack(fill="x", pady=2)
        tk.Label(mic_row, text="UMA-8 ID:", bg=PANEL, fg=FG).pack(side="left")
        default_front = str(self.geo.cfg.get("arrays", {}).get("front", {}).get("device_index", 4))
        self.var_dev_front = tk.StringVar(value=default_front)
        tk.Entry(mic_row, textvariable=self.var_dev_front, width=4, bg="#0d1117", fg=FG, insertbackground=FG).pack(side="left", padx=(4, 12))
        tk.Label(mic_row, text="UMA-16 ID:", bg=PANEL, fg=FG).pack(side="left")
        default_left = str(self.geo.cfg.get("arrays", {}).get("left", {}).get("device_index", 1))
        self.var_dev_left = tk.StringVar(value=default_left)
        tk.Entry(mic_row, textvariable=self.var_dev_left, width=4, bg="#0d1117", fg=FG, insertbackground=FG).pack(side="left", padx=4)

        band_row = tk.Frame(audio_settings, bg=PANEL)
        band_row.pack(fill="x", pady=2)
        tk.Label(band_row, text="Band Low:", bg=PANEL, fg=FG).pack(side="left")
        self.var_low = tk.StringVar(value=str(self.geo.cfg["default_detection"]["target_low_hz"]))
        entry_low = tk.Entry(band_row, textvariable=self.var_low, width=7, font=("Monospace", 9, "bold"), bg="#0d1117", fg=FG, insertbackground=FG)
        entry_low.pack(side="left", padx=(4, 8))
        entry_low.bind("<Return>", lambda event: self.on_apply_band())
        tk.Label(band_row, text="High:", bg=PANEL, fg=FG).pack(side="left")
        self.var_high = tk.StringVar(value=str(self.geo.cfg["default_detection"]["target_high_hz"]))
        entry_high = tk.Entry(band_row, textvariable=self.var_high, width=7, font=("Monospace", 9, "bold"), bg="#0d1117", fg=FG, insertbackground=FG)
        entry_high.pack(side="left", padx=4)
        entry_high.bind("<Return>", lambda event: self.on_apply_band())

        quality_row = tk.Frame(audio_settings, bg=PANEL)
        quality_row.pack(fill="x", pady=2)
        tk.Label(quality_row, text="Threshold:", bg=PANEL, fg=FG).pack(side="left")
        self.var_thresh = tk.StringVar(value=str(self.geo.cfg["default_detection"]["threshold_dbfs"]))
        tk.Entry(quality_row, textvariable=self.var_thresh, width=6, bg="#0d1117", fg=FG, insertbackground=FG).pack(side="left", padx=(4, 8))
        tk.Label(quality_row, text="Avg:", bg=PANEL, fg=FG).pack(side="left")
        self.var_avg = tk.StringVar(value=str(self.geo.cfg["default_detection"]["average_blocks"]))
        tk.Entry(quality_row, textvariable=self.var_avg, width=4, bg="#0d1117", fg=FG, insertbackground=FG).pack(side="left", padx=4)
        tk.Button(
            quality_row, text="Apply Audio", font=("Helvetica", 9, "bold"),
            bg="#21262d", fg=ACCENT, activebackground="#30363d", activeforeground=FG,
            padx=6, pady=2, command=self.on_apply_band, relief="flat",
        ).pack(side="right", padx=4)

        suppression_row = tk.Frame(audio_settings, bg=PANEL)
        suppression_row.pack(fill="x", pady=(4, 2))
        self.var_suppress_enabled = tk.BooleanVar(value=self.suppression.enabled)
        self.suppress_enable_button = tk.Checkbutton(
            suppression_row,
            text="Suppress Fixed Direction",
            variable=self.var_suppress_enabled,
            bg=PANEL,
            fg=FG,
            selectcolor="#0d1117",
            activebackground=PANEL,
            activeforeground=FG,
            command=self.on_apply_suppression,
        )
        self.suppress_enable_button.pack(side="left")

        suppression_values = tk.Frame(audio_settings, bg=PANEL)
        suppression_values.pack(fill="x", pady=2)
        tk.Label(suppression_values, text="Bearing:", bg=PANEL, fg=FG).pack(side="left")
        self.var_suppress_bearing = tk.StringVar(value=f"{self.suppression.bearing_deg:g}")
        self.suppress_bearing_entry = tk.Entry(
            suppression_values,
            textvariable=self.var_suppress_bearing,
            width=6,
            bg="#0d1117",
            fg=FG,
            insertbackground=FG,
        )
        self.suppress_bearing_entry.pack(side="left", padx=(4, 8))
        self.suppress_bearing_entry.bind("<Return>", lambda event: self.on_apply_suppression())
        tk.Label(suppression_values, text="±°:", bg=PANEL, fg=FG).pack(side="left")
        self.var_suppress_tolerance = tk.StringVar(value=f"{self.suppression.tolerance_deg:g}")
        self.suppress_tolerance_entry = tk.Entry(
            suppression_values,
            textvariable=self.var_suppress_tolerance,
            width=5,
            bg="#0d1117",
            fg=FG,
            insertbackground=FG,
        )
        self.suppress_tolerance_entry.pack(side="left", padx=4)
        self.suppress_tolerance_entry.bind("<Return>", lambda event: self.on_apply_suppression())
        self.suppress_apply_button = tk.Button(
            suppression_values,
            text="Apply",
            bg="#21262d",
            fg=ACCENT,
            relief="flat",
            command=self.on_apply_suppression,
        )
        self.suppress_apply_button.pack(side="right", padx=4)

        camera_settings = tk.LabelFrame(settings, text=" Camera ", bg=PANEL, fg=FG, padx=6, pady=4)
        camera_settings.grid(row=0, column=1, sticky="nsew", padx=6, pady=4)
        camera_action_row = tk.Frame(camera_settings, bg=PANEL)
        camera_action_row.pack(fill="x", pady=2)
        self.var_vision_enabled = tk.BooleanVar(value=self.vision is not None)
        tk.Checkbutton(
            camera_action_row,
            text="Enable Camera Detection",
            variable=self.var_vision_enabled,
            bg=PANEL,
            fg=FG,
            selectcolor="#0d1117",
            activebackground=PANEL,
            activeforeground=FG,
            command=self.on_toggle_vision,
            state="normal" if self.vision is not None else "disabled",
        ).pack(side="left")
        tk.Button(
            camera_action_row,
            text="Refresh Cameras",
            command=self.on_refresh_cameras,
            bg="#21262d",
            fg=ACCENT,
            relief="flat",
            state="normal" if self.vision is not None else "disabled",
        ).pack(side="right", padx=4)

        camera_source_row = tk.Frame(camera_settings, bg=PANEL)
        camera_source_row.pack(fill="x", pady=2)
        tk.Label(camera_source_row, text="Source:", bg=PANEL, fg=FG).pack(side="left")
        self.var_camera = tk.StringVar(value=str(self.vision.camera) if self.vision is not None else "")
        self.camera_combo = ttk.Combobox(
            camera_source_row,
            textvariable=self.var_camera,
            values=self._camera_sources(),
            width=42,
            state="normal" if self.vision is not None else "disabled",
        )
        self.camera_combo.pack(side="left", fill="x", expand=True, padx=4)
        self.camera_combo.bind("<<ComboboxSelected>>", self.on_camera_selected)
        self.camera_combo.bind("<Return>", self.on_camera_selected)

        fov_row = tk.Frame(camera_settings, bg=PANEL)
        fov_row.pack(fill="x", pady=2)
        tk.Label(fov_row, text="Horizontal FOV (°):", bg=PANEL, fg=FG).pack(side="left")
        initial_fov = self.vision.horizontal_fov_deg if self.vision is not None else None
        self.var_camera_fov = tk.StringVar(value="" if initial_fov is None else f"{initial_fov:g}")
        self.camera_fov_entry = tk.Entry(
            fov_row,
            textvariable=self.var_camera_fov,
            width=7,
            bg="#0d1117",
            fg=FG,
            insertbackground=FG,
            state="normal" if self.vision is not None else "disabled",
        )
        self.camera_fov_entry.pack(side="left", padx=4)
        self.camera_fov_entry.bind("<Return>", lambda event: self.on_apply_fov())
        tk.Button(
            fov_row,
            text="Apply FOV",
            command=self.on_apply_fov,
            bg="#21262d",
            fg=ACCENT,
            relief="flat",
            state="normal" if self.vision is not None else "disabled",
        ).pack(side="right", padx=4)

        mode_settings = tk.LabelFrame(settings, text=" Modes & Lock ", bg=PANEL, fg=FG, padx=8, pady=4)
        mode_settings.grid(row=0, column=2, sticky="nsew", padx=(6, 4), pady=4)
        self.var_sim = tk.BooleanVar(value=self.simulate)
        tk.Checkbutton(
            mode_settings, text="Simulation Mode", variable=self.var_sim, bg=PANEL, fg=FG,
            selectcolor="#0d1117", activebackground=PANEL, activeforeground=FG,
            command=self.on_toggle_sim,
        ).pack(anchor="w")
        self.var_front_only = tk.BooleanVar(value=False)
        tk.Checkbutton(
            mode_settings, text="UMA-8 Only", variable=self.var_front_only, bg=PANEL, fg=FG,
            selectcolor="#0d1117", activebackground=PANEL, activeforeground=FG,
            command=lambda: self.on_array_mode_change("front"),
        ).pack(anchor="w")
        self.var_left_only = tk.BooleanVar(value=False)
        tk.Checkbutton(
            mode_settings, text="UMA-16 Only", variable=self.var_left_only, bg=PANEL, fg=FG,
            selectcolor="#0d1117", activebackground=PANEL, activeforeground=FG,
            command=lambda: self.on_array_mode_change("left"),
        ).pack(anchor="w")
        self.var_uma16_front_only = tk.BooleanVar(value=False)
        tk.Checkbutton(
            mode_settings,
            text="UMA-16 Front Only (3D)",
            variable=self.var_uma16_front_only,
            bg=PANEL,
            fg=FG,
            selectcolor="#0d1117",
            activebackground=PANEL,
            activeforeground=FG,
            command=lambda: self.on_array_mode_change("uma16_front"),
        ).pack(anchor="w")
        self.var_require_drone = tk.BooleanVar(value=self.require_drone)
        tk.Checkbutton(
            mode_settings,
            text="Require Drone ID for Lock",
            variable=self.var_require_drone,
            bg=PANEL,
            fg=FG,
            selectcolor="#0d1117",
            activebackground=PANEL,
            activeforeground=FG,
        ).pack(anchor="w", pady=(4, 0))
        self._update_suppression_controls()

        self.jetson_tab_frame = tk.Frame(self.notebook, bg=BG)
        self.notebook.add(self.jetson_tab_frame, text=" Jetson Remote ")
        self.jetson_remote = JetsonRemoteTab(self.jetson_tab_frame, self.root, self.geo)
        self.notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)

    def _on_tab_changed(self, _event=None):
        if self.notebook.index(self.notebook.select()) == 1:
            self.jetson_remote.on_tab_selected()
        else:
            self.jetson_remote.on_tab_deselected()

    def _create_metric(self, parent: tk.Frame, title: str, init_val: str) -> tk.Label:
        box = tk.Frame(parent, bg=PANEL)
        box.pack(side="left", padx=12, pady=4)
        tk.Label(box, text=title, font=("Helvetica", 8), bg=PANEL, fg=MUTED).pack(anchor="w")
        val_lbl = tk.Label(box, text=init_val, font=("Monospace", 12, "bold"), bg=PANEL, fg=FG)
        val_lbl.pack(anchor="w")
        return val_lbl

    def _camera_sources(self):
        try:
            sources = [device.source for device in discover_cameras()]
        except Exception:
            sources = []
        current = str(self.vision.camera) if self.vision is not None else ""
        if current and current not in sources:
            sources.insert(0, current)
        return sources

    def on_refresh_cameras(self):
        self.camera_combo.configure(values=self._camera_sources())

    def on_camera_selected(self, event=None):
        if self.vision is None:
            return
        source = self.var_camera.get().strip()
        if not source:
            return
        if source != str(self.vision.camera):
            self.vision.set_camera(source)
            key = str(normalize_camera_source(source))
            saved_fov = self.geo.cfg.get("vision", {}).get("camera_fovs", {}).get(key)
            self.vision.set_horizontal_fov(saved_fov)
            self.var_camera_fov.set("" if saved_fov is None else f"{float(saved_fov):g}")
            self.vision_result = None
        self.save_config()
        self._update_vision()

    def on_apply_fov(self):
        if self.vision is None:
            return
        try:
            fov = validate_horizontal_fov(self.var_camera_fov.get())
        except (TypeError, ValueError):
            self.lbl_vision.configure(text="INVALID FOV", fg=WARN)
            return
        self.vision.set_horizontal_fov(fov)
        key = str(normalize_camera_source(self.vision.camera))
        vision_cfg = self.geo.cfg.setdefault("vision", {})
        vision_cfg.setdefault("camera_fovs", {})[key] = fov
        vision_cfg["horizontal_fov_deg"] = fov
        self.save_config()
        self._update_vision()

    def on_toggle_vision(self):
        if self.vision is None:
            return
        if self.var_vision_enabled.get():
            self.on_camera_selected()
            self.vision.start()
        else:
            self.vision.stop()
            self.vision_result = None
        self._update_vision()

    def _update_vision(self):
        if self.vision is None or not self.var_vision_enabled.get():
            self.lbl_vision.configure(text="OFF", fg=MUTED)
            self.camera_photo = None
            self.camera_video_lbl.configure(image="", text="CAMERA OFF")
            return

        result = self.vision.get_result()
        self.vision_result = result
        if result.error:
            tracker_error = result.error.startswith("Tracker error:")
            self.lbl_vision.configure(text="TRACKER OFFLINE" if tracker_error else "OFFLINE", fg=WARN)
        elif result.timestamp == 0.0:
            self.lbl_vision.configure(text="CONNECTING", fg=MUTED)
        elif result.target is not None and result.direction is not None:
            tracked = result.output_source == "tracker"
            label = "DRONE TRACKING" if tracked else f"DRONE {result.best_confidence:.0%}"
            self.lbl_vision.configure(
                text=f"{label} @ {result.direction.global_bearing_deg:.1f}°",
                fg=DRONE_RED,
            )
        elif result.target is not None:
            tracked = result.output_source == "tracker"
            label = "DRONE TRACKING" if tracked else f"DRONE {result.best_confidence:.0%}"
            self.lbl_vision.configure(text=f"{label} NO FOV", fg=DRONE_RED)
        elif result.detections:
            self.lbl_vision.configure(text=f"ACQUIRING {result.best_confidence:.0%}", fg=UNKNOWN_AMBER)
        elif result.tracking_state == "search":
            self.lbl_vision.configure(text="SEARCHING", fg=UNKNOWN_AMBER)
        else:
            self.lbl_vision.configure(text="CLEAR", fg=OK)

        if result.frame is None:
            self.camera_photo = None
            text = result.error or "CONNECTING TO CAMERA..."
            self.camera_video_lbl.configure(image="", text=text, width=48, height=11)
            return

        rgb = np.ascontiguousarray(result.frame[:, :, ::-1])
        image = Image.fromarray(rgb)
        width = max(320, self.camera_video_lbl.winfo_width())
        image.thumbnail((width, 240))
        self.camera_photo = ImageTk.PhotoImage(image=image)
        self.camera_video_lbl.configure(
            image=self.camera_photo,
            text="",
            width=image.width,
            height=image.height,
        )

    def on_zoom_change(self, val):
        self.map_scale = float(val)

    def _reset_classifier(self):
        if self.classifier is not None:
            self.classifier.reset()
        self.classification = None
        self.signal_detected = False
        self.last_classification = 0.0

    def _update_suppression_controls(self):
        state = "normal" if self.var_uma16_front_only.get() else "disabled"
        for widget in (
            self.suppress_enable_button,
            self.suppress_bearing_entry,
            self.suppress_tolerance_entry,
            self.suppress_apply_button,
        ):
            widget.configure(state=state)

    def _configured_suppression(self):
        return DirectionSuppression(
            enabled=self.var_suppress_enabled.get(),
            bearing_deg=float(self.var_suppress_bearing.get()),
            tolerance_deg=float(self.var_suppress_tolerance.get()),
            max_peak_candidates=self.suppression.max_peak_candidates,
        )

    def _current_suppression(self):
        try:
            configured = self._configured_suppression()
        except ValueError:
            configured = self.suppression
        return replace(
            configured,
            enabled=configured.enabled and self.var_uma16_front_only.get(),
        )

    def on_apply_suppression(self):
        if not self.var_uma16_front_only.get():
            return
        try:
            self.suppression = self._configured_suppression()
        except ValueError as exc:
            self.status_lbl.configure(text=f"INVALID SUPPRESSION: {exc}", fg=WARN, bg="#2b2010")
            return
        self.doa_engines["left"].reset()
        self.save_config()
        self.status_lbl.configure(
            text=(
                f"SUPPRESSION {self.suppression.bearing_deg:.1f}° ± {self.suppression.tolerance_deg:.1f}°"
                if self.suppression.enabled
                else "FIXED DIRECTION SUPPRESSION OFF"
            ),
            fg=ACCENT,
            bg="#1a2733",
        )

    def on_toggle_sim(self):
        self.simulate = self.var_sim.get()
        self.audio_mgr.simulate = self.simulate
        self._reset_classifier()
        self._update_classification_metric()
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
        if self.var_left_only.get() or self.var_uma16_front_only.get():
            return ["left"]
        return list(self.geo.arrays)

    def on_array_mode_change(self, selected):
        if selected == "front" and self.var_front_only.get():
            self.var_left_only.set(False)
            self.var_uma16_front_only.set(False)
        elif selected == "left" and self.var_left_only.get():
            self.var_front_only.set(False)
            self.var_uma16_front_only.set(False)
        elif selected == "uma16_front" and self.var_uma16_front_only.get():
            self.var_front_only.set(False)
            self.var_left_only.set(False)

        self._update_suppression_controls()
        self.geo.set_uma16_front_only(self.var_uma16_front_only.get())
        arr = self.geo.arrays["left"]
        self.doa_engines["left"] = ArrayDOA(
            array_id="left",
            mic_positions_local=arr.local_mics,
            orientation_deg=arr.orientation_deg,
            sample_rate=self.geo.sample_rate,
            mount=arr.mount,
            mic_heights_local=arr.mic_heights_local,
            freq_lo=200.0,
            freq_hi=16000.0,
        )
        self._clear_target_telemetry()
        self._reset_classifier()
        for doa_eng in self.doa_engines.values():
            doa_eng.reset()
        selected_ids = self._selected_array_ids()
        single_id = selected_ids[0] if len(selected_ids) == 1 else None
        empty_tri = TriangulationResult(False, None, None, None, None, 0.0, 0, {})
        self._draw_tactical_map(empty_tri, None, single_id)
        self._draw_direction_3d(None, single_id)
        self.audio_mgr.enabled_arrays = set(selected_ids)
        if self.is_active:
            self.audio_mgr.stop()
            self.audio_mgr.start(device_map=self._device_map(), array_ids=selected_ids)
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
            self._reset_classifier()
            self.save_config()
            self.audio_mgr.start(device_map=self._device_map(), array_ids=self._selected_array_ids())
            self.is_active = True
            self._update_classification_metric()
            self.btn_toggle.configure(text="STOP", bg=WARN, activebackground="#da3633")
            self.status_lbl.configure(text="SEARCHING...", fg=ACCENT)
        else:
            self.audio_mgr.stop()
            self.is_active = False
            self._reset_classifier()
            self._update_classification_metric()
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
            if self.vision is not None:
                self.geo.cfg.setdefault("vision", {})["camera"] = self.var_camera.get().strip()
            suppression = self._configured_suppression()
            self.geo.cfg.setdefault("uma16_front_only", {})["direction_suppression"] = {
                "enabled": suppression.enabled,
                "bearing_deg": suppression.bearing_deg,
                "tolerance_deg": suppression.tolerance_deg,
                "max_peak_candidates": suppression.max_peak_candidates,
            }
            with open(self.geo.config_path, "w") as f:
                json.dump(self.geo.cfg, f, indent=2)
        except Exception:
            pass

    def _update_loop(self):
        self._update_vision()
        if self.is_active or self.simulate:
            self._process_frame()
        else:
            self._update_fused_state()
        self.root.after(100, self._update_loop)

    def _process_frame(self):
        try:
            low = float(self.var_low.get())
            high = float(self.var_high.get())
        except ValueError:
            low, high = 900.0, 1100.0

        blocks = self.audio_mgr.get_latest_blocks()
        if not blocks:
            self._reset_classifier()
            self._update_classification_metric()
            self._clear_target_telemetry()
            self.status_lbl.configure(text="WAITING FOR AUDIO INPUT", fg=WARN, bg="#2b2010")
            return

        # 1. Frequency Detection (on front array ch0 or first available block)
        det_block = blocks.get("front", next(iter(blocks.values())))
        det_ch = det_block[:, 0]
        det_res = self.detector.process(det_ch, low, high)
        front_doa = None
        if self.var_uma16_front_only.get():
            selection = select_front_only_target(
                det_res,
                blocks["left"],
                self.doa_engines["left"],
                (low, high),
                self._current_suppression(),
            )
            det_res = selection.detection
            if selection.suppressed:
                self._show_suppressed_target(det_res)
                return
            front_doa = selection.doa
        self.signal_detected = det_res.detected

        now = time.monotonic()
        if not det_res.detected:
            self._reset_classifier()
        elif self.classifier is not None and now - self.last_classification >= self.classifier_hop:
            self.classification = self.classifier.classify(det_ch)
            self.last_classification = now
        self._update_classification_metric()

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
                res = front_doa if arr_id == "left" and front_doa is not None else doa_eng.estimate(
                    blocks[arr_id],
                    target_freq=target_f,
                    band=(low, high),
                )
                doa_results[arr_id] = res
                bearings[arr_id] = res.global_angle_deg
                prominences[arr_id] = res.prominence
                doa_errors[arr_id] = res.error

        single_array_id = selected_ids[0] if len(selected_ids) == 1 else None
        single_doa = doa_results.get(single_array_id) if single_array_id else None
        if det_res.detected and self.var_uma16_front_only.get() and not displayable_doa(
            single_doa,
            self._current_suppression(),
        ):
            hidden = replace(
                det_res,
                detected=False,
                peak_freq=None,
                peak_db=None,
                peak_candidates=(),
            )
            self._show_suppressed_target(hidden)
            return
        if single_doa is not None or not det_res.detected:
            tri_res = TriangulationResult(False, None, None, None, None, 0.0, 0, {})
        else:
            tri_res = self.triangulator.triangulate(
                bearings,
                prominences,
                stable={arr_id: res.stable for arr_id, res in doa_results.items()},
            )

        # 3. Update Telemetry & UI
        self._update_fused_state(
            detection=det_res,
            single_doa=single_doa,
            triangulation=tri_res,
        )
        self._update_telemetry(det_res, tri_res, doa_errors, single_doa, single_array_id)
        self._draw_spectrum(det_res)
        self._draw_tactical_map(tri_res, single_doa if det_res.detected else None, single_array_id)
        self._draw_direction_3d(single_doa, single_array_id)

    def _update_fused_state(
        self,
        detection=None,
        single_doa=None,
        triangulation=None,
        suppressed: bool = False,
    ):
        self.state_machine.config = StateMachineConfig(
            vision_hold_s=self.state_machine.config.vision_hold_s,
            require_drone_for_audio=bool(self.var_require_drone.get()),
            min_acoustic_prominence=self.state_machine.config.min_acoustic_prominence,
        )
        fused = self.state_machine.from_pipeline_outputs(
            now=time.monotonic(),
            vision=self.vision_result,
            detection=detection,
            single_doa=single_doa,
            triangulation=triangulation,
            classification=self.classification,
            suppressed=suppressed,
        )
        self.fused_state = fused
        if fused.mode == TrackingMode.VISION_TRACKING:
            bearing = f" {fused.bearing_deg:.1f}°" if fused.bearing_deg is not None else ""
            self.lbl_tracker_mode.configure(text=f"VISION{bearing}", fg="#c678dd")
        elif fused.mode == TrackingMode.VISION_HOLD:
            self.lbl_tracker_mode.configure(text="VISION HOLD", fg=WARN)
        elif fused.mode == TrackingMode.ACOUSTIC_TRACKING:
            bearing = f" {fused.bearing_deg:.1f}°" if fused.bearing_deg is not None else ""
            self.lbl_tracker_mode.configure(text=f"SOUND{bearing}", fg=ACCENT)
        elif fused.vision_uncalibrated:
            self.lbl_tracker_mode.configure(text="VISION UNCALIB", fg=WARN)
        else:
            self.lbl_tracker_mode.configure(text="SEARCHING", fg=MUTED)

    def _show_suppressed_target(self, detection):
        self._reset_classifier()
        self._update_classification_metric()
        self._update_fused_state(detection=detection, suppressed=True)
        self._clear_target_telemetry()
        self.lbl_freq.configure(text="---.- Hz", fg=MUTED)
        self.lbl_level.configure(text="---.- dBFS", fg=MUTED)
        self.status_lbl.configure(text="FIXED DIRECTION SUPPRESSED", fg=MUTED, bg="#1a2733")
        empty_tri = TriangulationResult(False, None, None, None, None, 0.0, 0, {})
        self._draw_spectrum(detection)
        self._draw_tactical_map(empty_tri, None, "left")
        self._draw_direction_3d(None, "left")

    def _update_classification_metric(self):
        result = self.classification
        if self.classifier is None:
            self.lbl_classification.configure(text="OFF", fg=MUTED)
        elif not self.is_active and not self.simulate:
            self.lbl_classification.configure(text="STANDBY", fg=MUTED)
        elif not self.signal_detected:
            self.lbl_classification.configure(text="NO TARGET SIGNAL", fg=MUTED)
        elif result is None or not result.is_ready:
            self.lbl_classification.configure(text="BUFFERING", fg=MUTED)
        elif result.is_drone:
            self.lbl_classification.configure(
                text=f"DRONE ({result.probability:.0%}) [{result.decision_votes}/{result.vote_window}]",
                fg=DRONE_RED,
            )
        elif result.probability >= result.threshold:
            self.lbl_classification.configure(
                text=f"CANDIDATE ({result.probability:.0%}) [{result.decision_votes}/{result.vote_window}]",
                fg=UNKNOWN_AMBER,
            )
        else:
            self.lbl_classification.configure(
                text=f"CLEAR ({result.probability:.0%}) [{result.decision_votes}/{result.vote_window}]",
                fg=OK,
            )

    def _drone_confirmed(self):
        return (
            self.signal_detected
            and self.classification is not None
            and self.classification.is_drone
        )

    def _clear_target_telemetry(self):
        self.lbl_range.configure(text="-.-- m", fg=MUTED)
        self.lbl_bearing.configure(text="---.-°", fg=MUTED)
        self.lbl_elevation.configure(text="---.-°", fg=MUTED)
        self.lbl_coords.configure(text="X: -.--  Y: -.--", fg=MUTED)

    def _update_telemetry(
        self,
        det: DetectionResult,
        tri: TriangulationResult,
        doa_errors=None,
        single_doa=None,
        single_array_id=None,
    ):
        if det.peak_freq is not None:
            self.lbl_freq.configure(text=f"{det.peak_freq:6.1f} Hz", fg=WARN if det.detected else FG)
        else:
            self.lbl_freq.configure(text="---.- Hz", fg=FG)

        if det.peak_db is not None:
            self.lbl_level.configure(text=f"{det.peak_db:5.1f} dBFS", fg=WARN if det.detected else FG)

        drone_confirmed = self._drone_confirmed()
        require_drone = self.var_require_drone.get()
        if single_doa is not None:
            if single_array_id == "front":
                array_label = "UMA-8"
            elif self.var_uma16_front_only.get():
                array_label = "UMA-16 FRONT"
            else:
                array_label = "UMA-16"
            if single_array_id == "left":
                if single_doa.elevation_deg is None:
                    self.lbl_elevation.configure(text="UNCERTAIN", fg=MUTED)
                else:
                    elevation = single_doa.elevation_deg
                    vertical = "ABOVE" if elevation > 2.0 else "BELOW" if elevation < -2.0 else "LEVEL"
                    self.lbl_elevation.configure(text=f"{elevation:+.1f}° {vertical}", fg=ACCENT)
            else:
                self.lbl_elevation.configure(text="N/A", fg=MUTED)
            if det.detected and single_doa.global_angle_deg is not None and (single_doa.prominence >= 1.4 or single_doa.stable):
                if require_drone and not drone_confirmed:
                    status = f"{array_label} BEARING / DRONE ID REQUIRED"
                    color = UNKNOWN_AMBER
                elif drone_confirmed:
                    status = f"{array_label} BEARING LOCK - DRONE CONFIRMED"
                    color = DRONE_RED
                else:
                    status = f"{array_label} ONLY / BEARING LOCK"
                    color = OK
                self.status_lbl.configure(text=status, fg=color, bg="#1a2733")
                self.lbl_range.configure(text="-.-- m", fg=MUTED)
                self.lbl_bearing.configure(text=f"{single_doa.global_angle_deg:5.1f}°", fg=color)
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

        self.lbl_elevation.configure(text="N/A", fg=MUTED)
        if det.detected:
            if tri.success and tri.x is not None and tri.y is not None:
                if require_drone and not drone_confirmed:
                    status = "ACOUSTIC SOURCE / DRONE ID REQUIRED"
                elif drone_confirmed:
                    status = "TARGET LOCKED - DRONE CONFIRMED"
                else:
                    status = "TARGET LOCKED - UNKNOWN/NON-DRONE"
                color = DRONE_RED if drone_confirmed else UNKNOWN_AMBER
                self.status_lbl.configure(text=status, fg=color, bg="#3b1d1d" if drone_confirmed else "#2b2010")
                self.lbl_range.configure(text=f"{tri.range_m:4.2f} m", fg=color)
                self.lbl_bearing.configure(text=f"{tri.bearing_deg:5.1f}°", fg=color)
                self.lbl_coords.configure(text=f"X:{tri.x:+4.2f}m  Y:{tri.y:+4.2f}m", fg=color)
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
        draw_spectrum_canvas(
            self.canvas_spec,
            freqs=det.freqs,
            mags=det.mags,
            band_low=det.band[0],
            band_high=det.band[1],
            threshold_db=det.threshold_db,
            peak_freq=det.peak_freq,
        )

    def _draw_direction_3d(self, result, array_id):
        if array_id != "left":
            draw_direction_3d_canvas(
                self.canvas_direction,
                direction_vector=None,
                bearing_deg=None,
                elevation_deg=None,
                title="SOUND SOURCE DIRECTION",
                blocked_message="SELECT UMA-16 ONLY OR UMA-16 FRONT ONLY",
            )
            return
        draw_direction_3d_canvas(
            self.canvas_direction,
            direction_vector=result.direction_vector if result is not None else None,
            bearing_deg=result.global_angle_deg if result is not None else None,
            elevation_deg=result.elevation_deg if result is not None else None,
            title="SOUND SOURCE DIRECTION",
            empty_title="WAITING",
            empty_subtitle="ELEVATION UNCERTAIN",
            empty_note="No direction arrow",
        )

    def _draw_tactical_map(self, tri, single_doa=None, single_array_id=None):
        single_bearing = None
        single_label = None
        if single_doa is not None and single_doa.global_angle_deg is not None and (single_doa.prominence >= 1.4 or single_doa.stable):
            single_bearing = single_doa.global_angle_deg
            if single_array_id == "front":
                single_label = "UMA-8"
            elif self.var_uma16_front_only.get():
                single_label = "UMA-16 FRONT"
            else:
                single_label = "UMA-16"

        tri_x = tri.x if tri.success and tri.x is not None and tri.y is not None else None
        tri_y = tri.y if tri_x is not None else None
        tri_label = None
        drone_confirmed = self._drone_confirmed()
        if tri_x is not None:
            if drone_confirmed:
                tri_label = f"DRONE ({self.classification.probability:.0%})"
            elif self.classification is not None and self.classification.is_ready:
                tri_label = f"TARGET ({self.classification.probability:.0%})"
            else:
                tri_label = "TARGET (UNKNOWN)"

        visual = self.vision_result
        cam_b = None
        cam_el = None
        if visual is not None and visual.target is not None and visual.direction is not None:
            cam_b = visual.direction.global_bearing_deg
            cam_el = visual.direction.elevation_deg

        suppression = self._current_suppression()
        draw_radar_canvas(
            self.canvas_radar,
            geo=self.geo,
            map_scale=self.map_scale,
            selected_ids=self._selected_array_ids(),
            suppression_enabled=suppression.enabled,
            suppression_bearing=suppression.bearing_deg,
            suppression_tolerance=suppression.tolerance_deg,
            single_bearing=single_bearing,
            single_array_id=single_array_id,
            single_label=single_label,
            tri_x=tri_x,
            tri_y=tri_y,
            tri_label=tri_label,
            drone_confirmed=drone_confirmed,
            camera_bearing=cam_b,
            camera_elevation=cam_el,
        )

    def on_close(self):
        self.jetson_remote.stop_if_running()
        self.audio_mgr.stop()
        if self.vision is not None:
            self.vision.stop()
        self.root.destroy()


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


class JetsonRemoteTab:
    """Embedded Jetson Remote tab mirroring all Local Tactical Radar features over SSH."""

    def __init__(self, parent: tk.Frame, root: tk.Tk, geo: RigGeometry):
        self.parent = parent
        self.root = root
        self.geo = RigGeometry(geo.config_path)
        self._ui_queue: queue.Queue = queue.Queue()
        self._cli_stop_requested = False
        self._cli_running = False
        self._restart_after_stop = False
        self._preview_running = False
        self._preview_stop_requested = False
        self._preview_restart = False
        self._tab_active = False
        self._initialized = False
        self.map_scale = 90.0
        self.camera_photo = None
        self._log_window = None
        self._log_widget = None
        self._log_history: List[str] = []
        self._audio_devices: List[Dict] = []
        self._camera_fovs: Dict[str, float] = {}

        self._build_ui()
        self._update_suppression_controls()
        self._reset_live_telemetry("STANDBY")
        self.root.after(50, self._drain_queue)

    def ensure_initialized(self):
        self.on_tab_selected()

    def on_tab_selected(self):
        self._tab_active = True
        if not self._initialized:
            self._initialized = True
            self.on_refresh_all()
        else:
            self._start_preview_if_needed()

    def on_tab_deselected(self):
        self._tab_active = False
        self._stop_preview()

    def _build_ui(self):
        # 0. SSH Connection & Quick Actions Bar
        ssh_bar = tk.Frame(self.parent, bg=PANEL, highlightbackground=BORDER, highlightthickness=1)
        ssh_bar.pack(fill="x", padx=10, pady=(8, 4))

        self.var_host = tk.StringVar(value=os.environ.get("JETSON_HOST", jetson_ctl.DEFAULT_HOST))
        self.var_port = tk.StringVar(value=os.environ.get("JETSON_PORT", str(jetson_ctl.DEFAULT_PORT)))
        self.var_user = tk.StringVar(value=os.environ.get("JETSON_USER", jetson_ctl.DEFAULT_USER))
        self.var_dir = tk.StringVar(value=os.environ.get("JETSON_DIR", jetson_ctl.DEFAULT_REMOTE_DIR))
        self.var_pass = tk.StringVar(
            value=os.environ.get("JETSON_PASS") or os.environ.get("SSHPASS") or "meltem123"
        )
        self.var_stream_video = tk.BooleanVar(value=True)

        for label, var, width, show in [
            ("Host:", self.var_host, 14, ""),
            ("Port:", self.var_port, 5, ""),
            ("User:", self.var_user, 8, ""),
            ("Dir:", self.var_dir, 22, ""),
            ("Pass:", self.var_pass, 10, "*"),
        ]:
            tk.Label(ssh_bar, text=label, bg=PANEL, fg=MUTED, font=("Helvetica", 8)).pack(side="left", padx=(4, 1))
            tk.Entry(ssh_bar, textvariable=var, width=width, show=show, bg="#0d1117", fg=FG, insertbackground=FG).pack(side="left", padx=(0, 5), pady=4)

        for text, cmd, bg in [
            ("Connect / Refresh All", self.on_refresh_all, "#1f6feb"),
            ("Sync Code → Jetson", self.on_sync, "#238636"),
            ("Run Tests", self.on_run_tests, "#30363d"),
            ("Console Log", self.on_open_console, "#30363d"),
        ]:
            tk.Button(
                ssh_bar, text=text, command=cmd, bg=bg, fg="#ffffff",
                font=("Helvetica", 8, "bold"), relief="flat", padx=8, pady=3,
            ).pack(side="left", padx=2, pady=4)

        tk.Checkbutton(
            ssh_bar, text="Stream Video", variable=self.var_stream_video,
            bg=PANEL, fg=FG, selectcolor="#0d1117", activebackground=PANEL, activeforeground=FG,
            font=("Helvetica", 8), command=self.on_toggle_stream_video,
        ).pack(side="left", padx=6)

        self.conn_status = tk.Label(
            ssh_bar, text="DISCONNECTED", bg="#2d333b", fg=MUTED,
            font=("Helvetica", 8, "bold"), padx=8, pady=3,
        )
        self.conn_status.pack(side="right", padx=6)

        # 1. Top Header & Telemetry Bar (identical to Local Tactical Radar)
        top_frame = tk.Frame(self.parent, bg=PANEL, highlightbackground=BORDER, highlightthickness=1)
        top_frame.pack(fill="x", padx=10, pady=(2, 6))

        self.status_lbl = tk.Label(
            top_frame, text="STANDBY", font=("Helvetica", 14, "bold"),
            bg="#2d333b", fg=MUTED, padx=16, pady=8,
        )
        self.status_lbl.pack(side="left", padx=8, pady=8)

        self.lbl_freq = self._create_metric(top_frame, "DETECTED FREQ", "---.- Hz")
        self.lbl_level = self._create_metric(top_frame, "SIGNAL LEVEL", "---.- dBFS")
        self.lbl_classification = self._create_metric(top_frame, "CLASSIFICATION", "STANDBY")
        self.lbl_vision = self._create_metric(top_frame, "CAMERA", "STANDBY")
        self.lbl_tracker_mode = self._create_metric(top_frame, "ACTIVE TRACKER", "SEARCHING")
        self.lbl_range = self._create_metric(top_frame, "RANGE", "-.-- m")
        self.lbl_bearing = self._create_metric(top_frame, "BEARING", "---.-°")
        self.lbl_elevation = self._create_metric(top_frame, "ELEVATION", "---.-°")
        self.lbl_coords = self._create_metric(top_frame, "COORDINATES (X, Y)", "X: -.--  Y: -.--")

        # 2. Main Visual Body (Spectrum + Camera on Left, 2D Radar + 3D Direction on Right)
        body = tk.Frame(self.parent, bg=BG)
        body.pack(fill="both", expand=True, padx=10, pady=4)

        left_visual = tk.Frame(body, bg=BG)
        left_visual.pack(side="left", fill="both", expand=False, padx=(0, 6))
        spec_frame = tk.LabelFrame(
            left_visual, text=" Real-Time Spectrum (Narrowband Filter) ",
            bg=BG, fg=FG, font=("Helvetica", 10, "bold"), highlightbackground=BORDER,
        )
        spec_frame.pack(side="top", fill="both", expand=True)
        self.canvas_spec = tk.Canvas(spec_frame, width=400, height=280, bg="#0d1117", highlightthickness=0)
        self.canvas_spec.pack(fill="both", expand=True, padx=4, pady=4)

        camera_frame = tk.LabelFrame(
            left_visual, text=" YOLO Camera Detection ",
            bg=BG, fg=FG, font=("Helvetica", 10, "bold"), highlightbackground=BORDER,
        )
        camera_frame.pack(side="bottom", fill="both", expand=False, pady=(6, 0))
        self.camera_video_lbl = tk.Label(
            camera_frame, text="NO CAMERA FRAME", width=48, height=11, bg="#0d1117", fg=MUTED,
        )
        self.camera_video_lbl.pack(fill="both", expand=True, padx=4, pady=4)

        right_visual = tk.Frame(body, bg=BG)
        right_visual.pack(side="right", fill="both", expand=True)
        radar_frame = tk.LabelFrame(
            right_visual, text=" 2D Tactical Spatial Map (Top-Down Triangulation) ",
            bg=BG, fg=FG, font=("Helvetica", 10, "bold"), highlightbackground=BORDER,
        )
        radar_frame.pack(side="top", fill="both", expand=True)
        self.canvas_radar = tk.Canvas(radar_frame, width=680, height=340, bg="#0d1117", highlightthickness=0)
        self.canvas_radar.pack(fill="both", expand=True, padx=4, pady=4)

        direction_frame = tk.LabelFrame(
            right_visual, text=" SOUND DIRECTION — CLEAR 3D VIEW ",
            bg=BG, fg=FG, font=("Helvetica", 10, "bold"), highlightbackground=BORDER,
        )
        direction_frame.pack(side="bottom", fill="x", pady=(6, 0))
        self.canvas_direction = tk.Canvas(direction_frame, width=680, height=240, bg="#0d1117", highlightthickness=0)
        self.canvas_direction.pack(fill="both", expand=True, padx=4, pady=4)

        # 3. Actions and Organized Settings (identical to Local)
        ctrl = tk.Frame(self.parent, bg=PANEL, highlightbackground=BORDER, highlightthickness=1)
        ctrl.pack(fill="x", padx=10, pady=(4, 8))

        action_row = tk.Frame(ctrl, bg=PANEL)
        action_row.pack(fill="x", padx=8, pady=(6, 4))
        self.btn_toggle = tk.Button(
            action_row, text="START LISTENING", font=("Helvetica", 11, "bold"),
            bg="#238636", fg="#ffffff", activebackground="#2ea043",
            padx=14, pady=4, command=self.on_toggle_listening, relief="flat",
        )
        self.btn_toggle.pack(side="left", padx=(4, 12))

        self.var_classifier = tk.BooleanVar(value=True)
        tk.Checkbutton(
            action_row, text="CLASSIFIER ACTIVE", variable=self.var_classifier,
            font=("Helvetica", 9, "bold"), bg=PANEL, fg=OK, selectcolor="#0d1117",
            activebackground=PANEL, activeforeground=OK, command=self.on_apply_audio,
        ).pack(side="left", padx=4)

        self.lbl_last_log = tk.Label(action_row, text="Ready.", bg=PANEL, fg=MUTED, font=("Monospace", 9))
        self.lbl_last_log.pack(side="left", padx=12)

        tk.Label(action_row, text="Map Zoom:", bg=PANEL, fg=FG).pack(side="right", padx=(4, 2))
        self.slider_zoom = tk.Scale(
            action_row, from_=30, to=200, orient="horizontal", showvalue=True, bg=PANEL,
            fg=FG, troughcolor="#0d1117", highlightthickness=0, length=110,
            command=self.on_zoom_change,
        )
        self.slider_zoom.set(int(self.map_scale))
        self.slider_zoom.pack(side="right", padx=4)

        settings = tk.LabelFrame(
            ctrl, text=" Settings ", bg=PANEL, fg=ACCENT,
            font=("Helvetica", 10, "bold"), highlightbackground=BORDER,
        )
        settings.pack(fill="x", padx=8, pady=(2, 6))
        settings.columnconfigure(0, weight=1)
        settings.columnconfigure(1, weight=2)
        settings.columnconfigure(2, weight=1)

        # Audio & Detection (matches Local)
        audio_settings = tk.LabelFrame(settings, text=" Audio & Detection ", bg=PANEL, fg=FG, padx=6, pady=4)
        audio_settings.grid(row=0, column=0, sticky="nsew", padx=(4, 6), pady=4)
        mic_row = tk.Frame(audio_settings, bg=PANEL)
        mic_row.pack(fill="x", pady=2)
        tk.Label(mic_row, text="UMA-8 ID:", bg=PANEL, fg=FG).pack(side="left")
        self.var_dev_front = tk.StringVar(value="4")
        self.combo_dev_front = ttk.Combobox(mic_row, textvariable=self.var_dev_front, width=14)
        self.combo_dev_front.pack(side="left", padx=(4, 8))
        tk.Label(mic_row, text="UMA-16 ID:", bg=PANEL, fg=FG).pack(side="left")
        self.var_dev_left = tk.StringVar(value="0")
        self.combo_dev_left = ttk.Combobox(mic_row, textvariable=self.var_dev_left, width=14)
        self.combo_dev_left.pack(side="left", padx=4)
        tk.Button(
            mic_row, text="Refresh Mics", command=self.on_refresh_all,
            bg="#21262d", fg=ACCENT, relief="flat", padx=4,
        ).pack(side="right", padx=2)

        band_row = tk.Frame(audio_settings, bg=PANEL)
        band_row.pack(fill="x", pady=2)
        tk.Label(band_row, text="Band Low:", bg=PANEL, fg=FG).pack(side="left")
        self.var_low = tk.StringVar(value="200.0")
        entry_low = tk.Entry(band_row, textvariable=self.var_low, width=7, font=("Monospace", 9, "bold"), bg="#0d1117", fg=FG, insertbackground=FG)
        entry_low.pack(side="left", padx=(4, 8))
        entry_low.bind("<Return>", lambda _e: self.on_apply_audio())
        tk.Label(band_row, text="High:", bg=PANEL, fg=FG).pack(side="left")
        self.var_high = tk.StringVar(value="5000.0")
        entry_high = tk.Entry(band_row, textvariable=self.var_high, width=7, font=("Monospace", 9, "bold"), bg="#0d1117", fg=FG, insertbackground=FG)
        entry_high.pack(side="left", padx=4)
        entry_high.bind("<Return>", lambda _e: self.on_apply_audio())

        quality_row = tk.Frame(audio_settings, bg=PANEL)
        quality_row.pack(fill="x", pady=2)
        tk.Label(quality_row, text="Threshold:", bg=PANEL, fg=FG).pack(side="left")
        self.var_threshold = tk.StringVar(value="-70.0")
        tk.Entry(quality_row, textvariable=self.var_threshold, width=6, bg="#0d1117", fg=FG, insertbackground=FG).pack(side="left", padx=(4, 8))
        tk.Label(quality_row, text="Avg:", bg=PANEL, fg=FG).pack(side="left")
        self.var_avg = tk.StringVar(value="4")
        tk.Entry(quality_row, textvariable=self.var_avg, width=4, bg="#0d1117", fg=FG, insertbackground=FG).pack(side="left", padx=4)
        tk.Button(
            quality_row, text="Apply Audio", font=("Helvetica", 9, "bold"),
            bg="#21262d", fg=ACCENT, activebackground="#30363d", activeforeground=FG,
            padx=6, pady=2, command=self.on_apply_audio, relief="flat",
        ).pack(side="right", padx=4)

        suppression_row = tk.Frame(audio_settings, bg=PANEL)
        suppression_row.pack(fill="x", pady=(4, 2))
        self.var_sup_enabled = tk.BooleanVar(value=False)
        self.suppress_enable_button = tk.Checkbutton(
            suppression_row, text="Suppress Fixed Direction", variable=self.var_sup_enabled,
            bg=PANEL, fg=FG, selectcolor="#0d1117", activebackground=PANEL, activeforeground=FG,
            command=self.on_apply_suppression,
        )
        self.suppress_enable_button.pack(side="left")

        suppression_values = tk.Frame(audio_settings, bg=PANEL)
        suppression_values.pack(fill="x", pady=2)
        tk.Label(suppression_values, text="Bearing:", bg=PANEL, fg=FG).pack(side="left")
        self.var_sup_bearing = tk.StringVar(value="60.0")
        self.suppress_bearing_entry = tk.Entry(
            suppression_values, textvariable=self.var_sup_bearing, width=6,
            bg="#0d1117", fg=FG, insertbackground=FG,
        )
        self.suppress_bearing_entry.pack(side="left", padx=(4, 8))
        self.suppress_bearing_entry.bind("<Return>", lambda _e: self.on_apply_suppression())
        tk.Label(suppression_values, text="±°:", bg=PANEL, fg=FG).pack(side="left")
        self.var_sup_tol = tk.StringVar(value="10.0")
        self.suppress_tolerance_entry = tk.Entry(
            suppression_values, textvariable=self.var_sup_tol, width=5,
            bg="#0d1117", fg=FG, insertbackground=FG,
        )
        self.suppress_tolerance_entry.pack(side="left", padx=4)
        self.suppress_tolerance_entry.bind("<Return>", lambda _e: self.on_apply_suppression())
        self.suppress_apply_button = tk.Button(
            suppression_values, text="Apply", bg="#21262d", fg=ACCENT, relief="flat",
            command=self.on_apply_suppression,
        )
        self.suppress_apply_button.pack(side="right", padx=4)

        # Camera Settings (matches Local)
        camera_settings = tk.LabelFrame(settings, text=" Camera ", bg=PANEL, fg=FG, padx=6, pady=4)
        camera_settings.grid(row=0, column=1, sticky="nsew", padx=6, pady=4)
        camera_action_row = tk.Frame(camera_settings, bg=PANEL)
        camera_action_row.pack(fill="x", pady=2)
        self.var_vision_enabled = tk.BooleanVar(value=True)
        tk.Checkbutton(
            camera_action_row, text="Enable Camera Detection", variable=self.var_vision_enabled,
            bg=PANEL, fg=FG, selectcolor="#0d1117", activebackground=PANEL, activeforeground=FG,
            command=self.on_toggle_vision,
        ).pack(side="left")
        self.var_tracker = tk.BooleanVar(value=True)
        tk.Checkbutton(
            camera_action_row, text="DyHiT Tracker", variable=self.var_tracker,
            bg=PANEL, fg=FG, selectcolor="#0d1117", activebackground=PANEL, activeforeground=FG,
            command=self.on_apply_camera,
        ).pack(side="left", padx=8)
        tk.Button(
            camera_action_row, text="Refresh Cameras", command=self.on_refresh_all,
            bg="#21262d", fg=ACCENT, relief="flat",
        ).pack(side="right", padx=4)

        camera_source_row = tk.Frame(camera_settings, bg=PANEL)
        camera_source_row.pack(fill="x", pady=2)
        tk.Label(camera_source_row, text="Source:", bg=PANEL, fg=FG).pack(side="left")
        self.var_camera = tk.StringVar(value="rtsp://127.0.0.1:8554/drone")
        self.camera_combo = ttk.Combobox(
            camera_source_row, textvariable=self.var_camera,
            values=["rtsp://127.0.0.1:8554/drone", "/dev/video0"], width=42,
        )
        self.camera_combo.pack(side="left", fill="x", expand=True, padx=4)
        self.camera_combo.bind("<<ComboboxSelected>>", self.on_camera_selected)
        self.camera_combo.bind("<Return>", self.on_camera_selected)

        fov_row = tk.Frame(camera_settings, bg=PANEL)
        fov_row.pack(fill="x", pady=2)
        tk.Label(fov_row, text="Horizontal FOV (°):", bg=PANEL, fg=FG).pack(side="left")
        self.var_camera_fov = tk.StringVar(value="75.0")
        self.camera_fov_entry = tk.Entry(
            fov_row, textvariable=self.var_camera_fov, width=7,
            bg="#0d1117", fg=FG, insertbackground=FG,
        )
        self.camera_fov_entry.pack(side="left", padx=4)
        self.camera_fov_entry.bind("<Return>", lambda _e: self.on_apply_fov())
        tk.Button(
            fov_row, text="Apply FOV", command=self.on_apply_fov,
            bg="#21262d", fg=ACCENT, relief="flat",
        ).pack(side="right", padx=4)

        # Modes & Lock (exact same checkboxes as Local!)
        mode_settings = tk.LabelFrame(settings, text=" Modes & Lock ", bg=PANEL, fg=FG, padx=8, pady=4)
        mode_settings.grid(row=0, column=2, sticky="nsew", padx=(6, 4), pady=4)
        self.var_simulate = tk.BooleanVar(value=False)
        tk.Checkbutton(
            mode_settings, text="Simulation Mode", variable=self.var_simulate, bg=PANEL, fg=FG,
            selectcolor="#0d1117", activebackground=PANEL, activeforeground=FG,
            command=self.on_toggle_sim,
        ).pack(anchor="w")
        self.var_front_only = tk.BooleanVar(value=False)
        tk.Checkbutton(
            mode_settings, text="UMA-8 Only", variable=self.var_front_only, bg=PANEL, fg=FG,
            selectcolor="#0d1117", activebackground=PANEL, activeforeground=FG,
            command=lambda: self.on_array_mode_change("front"),
        ).pack(anchor="w")
        self.var_left_only = tk.BooleanVar(value=False)
        tk.Checkbutton(
            mode_settings, text="UMA-16 Only", variable=self.var_left_only, bg=PANEL, fg=FG,
            selectcolor="#0d1117", activebackground=PANEL, activeforeground=FG,
            command=lambda: self.on_array_mode_change("left"),
        ).pack(anchor="w")
        self.var_uma16_front_only = tk.BooleanVar(value=False)
        tk.Checkbutton(
            mode_settings, text="UMA-16 Front Only (3D)", variable=self.var_uma16_front_only,
            bg=PANEL, fg=FG, selectcolor="#0d1117", activebackground=PANEL, activeforeground=FG,
            command=lambda: self.on_array_mode_change("uma16_front"),
        ).pack(anchor="w")
        self.var_require_drone = tk.BooleanVar(value=False)
        tk.Checkbutton(
            mode_settings, text="Require Drone ID for Lock", variable=self.var_require_drone,
            bg=PANEL, fg=FG, selectcolor="#0d1117", activebackground=PANEL, activeforeground=FG,
        ).pack(anchor="w", pady=(4, 0))

    def _create_metric(self, parent: tk.Frame, title: str, init_val: str) -> tk.Label:
        box = tk.Frame(parent, bg=PANEL)
        box.pack(side="left", padx=12, pady=4)
        tk.Label(box, text=title, font=("Helvetica", 8), bg=PANEL, fg=MUTED).pack(anchor="w")
        val_lbl = tk.Label(box, text=init_val, font=("Monospace", 12, "bold"), bg=PANEL, fg=FG)
        val_lbl.pack(anchor="w")
        return val_lbl

    @staticmethod
    def _extract_dev_id(raw: str) -> str:
        text = str(raw).strip()
        m = re.match(r"^\[?\s*(\d+)\s*\]?", text)
        return m.group(1) if m else text

    def _selected_array_ids(self) -> List[str]:
        if self.var_front_only.get():
            return ["front"]
        if self.var_left_only.get() or self.var_uma16_front_only.get():
            return ["left"]
        return list(self.geo.arrays)

    def _update_suppression_controls(self):
        state = "normal" if self.var_uma16_front_only.get() else "disabled"
        for widget in (
            self.suppress_enable_button,
            self.suppress_bearing_entry,
            self.suppress_tolerance_entry,
            self.suppress_apply_button,
        ):
            widget.configure(state=state)

    def _redraw_static_views(self):
        self.geo.set_uma16_front_only(self.var_uma16_front_only.get())
        selected_ids = self._selected_array_ids()
        single_id = selected_ids[0] if len(selected_ids) == 1 else None
        sup_enabled = bool(self.var_sup_enabled.get() and self.var_uma16_front_only.get())
        try:
            sup_b = float(self.var_sup_bearing.get())
            sup_t = float(self.var_sup_tol.get())
        except ValueError:
            sup_b, sup_t = 60.0, 10.0

        draw_radar_canvas(
            self.canvas_radar,
            geo=self.geo,
            map_scale=self.map_scale,
            selected_ids=selected_ids,
            suppression_enabled=sup_enabled,
            suppression_bearing=sup_b,
            suppression_tolerance=sup_t,
        )
        if single_id != "left":
            draw_direction_3d_canvas(
                self.canvas_direction,
                direction_vector=None,
                bearing_deg=None,
                elevation_deg=None,
                title="SOUND SOURCE DIRECTION",
                blocked_message="SELECT UMA-16 ONLY OR UMA-16 FRONT ONLY",
                default_size=(680, 240),
            )
        else:
            draw_direction_3d_canvas(
                self.canvas_direction,
                direction_vector=None,
                bearing_deg=None,
                elevation_deg=None,
                title="SOUND SOURCE DIRECTION",
                empty_title="WAITING",
                empty_subtitle="ELEVATION UNCERTAIN",
                empty_note="No direction arrow",
                default_size=(680, 240),
            )

    def on_zoom_change(self, val):
        self.map_scale = float(val)
        if not self._cli_running:
            self._redraw_static_views()

    def on_array_mode_change(self, selected: str):
        if selected == "front" and self.var_front_only.get():
            self.var_left_only.set(False)
            self.var_uma16_front_only.set(False)
        elif selected == "left" and self.var_left_only.get():
            self.var_front_only.set(False)
            self.var_uma16_front_only.set(False)
        elif selected == "uma16_front" and self.var_uma16_front_only.get():
            self.var_front_only.set(False)
            self.var_left_only.set(False)

        self._update_suppression_controls()
        self._redraw_static_views()
        if self._cli_running:
            self._restart_cli()

    def on_toggle_sim(self):
        if self._cli_running:
            self._restart_cli()

    def _show_camera_frame_b64(self, frame_b64: str):
        try:
            img = Image.open(io.BytesIO(base64.b64decode(frame_b64)))
            width = max(320, self.camera_video_lbl.winfo_width())
            img.thumbnail((width, 240))
            self.camera_photo = ImageTk.PhotoImage(image=img)
            self.camera_video_lbl.configure(
                image=self.camera_photo,
                text="",
                width=img.width,
                height=img.height,
            )
        except Exception:
            pass

    def _show_camera_text(self, text: str):
        self.camera_photo = None
        self.camera_video_lbl.configure(image="", text=text, width=48, height=11)

    def on_toggle_stream_video(self):
        if not self.var_stream_video.get():
            self._stop_preview()
            if not self._cli_running:
                self._show_camera_text("VIDEO STREAM OFF")
        elif not self._cli_running:
            self._start_preview_if_needed()

    def on_toggle_vision(self):
        if not self.var_vision_enabled.get():
            self.lbl_vision.configure(text="OFF", fg=MUTED)
            self._stop_preview()
            self._show_camera_text("CAMERA OFF")
        elif not self._cli_running:
            self._start_preview_if_needed()
        if self._cli_running:
            self._restart_cli()

    def on_camera_selected(self, _event=None):
        cam = self.var_camera.get().strip()
        if not cam:
            return
        saved_fov = self._camera_fovs.get(cam)
        if saved_fov is not None:
            self.var_camera_fov.set(f"{float(saved_fov):g}")
        self.on_apply_camera()
        if not self._cli_running:
            self._restart_preview()

    def on_apply_fov(self):
        try:
            fov = validate_horizontal_fov(self.var_camera_fov.get())
            if fov is None:
                raise ValueError("Horizontal FOV is required")
        except (TypeError, ValueError):
            self.lbl_vision.configure(text="INVALID FOV", fg=WARN)
            return
        cam = self.var_camera.get().strip()
        if cam:
            self._camera_fovs[cam] = fov
        self.on_apply_camera()

    def on_apply_camera(self):
        self.on_apply_config()
        if self._cli_running:
            self._restart_cli()

    def on_apply_audio(self):
        self.on_apply_config()
        if self._cli_running:
            self._restart_cli()

    def on_apply_suppression(self):
        self._redraw_static_views()
        self.on_apply_config()
        if self._cli_running:
            self._restart_cli()

    def on_open_console(self):
        if self._log_window is not None and self._log_window.winfo_exists():
            self._log_window.lift()
            return
        win = tk.Toplevel(self.root)
        win.title("Jetson Remote Console")
        win.configure(bg=BG)
        win.geometry("820x460")
        txt = scrolledtext.ScrolledText(
            win, bg="#0d1117", fg=FG, insertbackground=FG,
            font=("Monospace", 10), state="normal", wrap="word",
        )
        txt.pack(fill="both", expand=True, padx=8, pady=8)
        if self._log_history:
            txt.insert("end", "\n".join(self._log_history) + "\n")
            txt.see("end")
        txt.configure(state="disabled")
        self._log_window = win
        self._log_widget = txt

    def _target(self) -> jetson_ctl.JetsonTarget:
        return jetson_ctl.JetsonTarget(
            host=self.var_host.get().strip(),
            port=int(self.var_port.get().strip()),
            user=self.var_user.get().strip(),
            remote_dir=self.var_dir.get().strip(),
        )

    def _password(self) -> Optional[str]:
        pw = self.var_pass.get()
        return pw if pw else None

    def _set_disconnected(self):
        self._stop_preview()
        self.conn_status.configure(text="DISCONNECTED", bg="#3b1d1d", fg=DRONE_RED)
        self._reset_live_telemetry("DISCONNECTED")

    def _reset_live_telemetry(self, status_text: str = "STANDBY", preserve_camera: bool = False):
        is_disc = status_text == "DISCONNECTED"
        self.status_lbl.configure(
            text=status_text,
            fg=DRONE_RED if is_disc else MUTED,
            bg="#3b1d1d" if is_disc else "#2d333b",
        )
        self.lbl_freq.configure(text="---.- Hz", fg=MUTED)
        self.lbl_level.configure(text="---.- dBFS", fg=MUTED)
        self.lbl_classification.configure(text="STANDBY", fg=MUTED)
        self.lbl_vision.configure(text="STANDBY" if self.var_vision_enabled.get() else "OFF", fg=MUTED)
        self.lbl_tracker_mode.configure(text="DISCONNECTED" if is_disc else "SEARCHING", fg=DRONE_RED if is_disc else MUTED)
        self.lbl_range.configure(text="-.-- m", fg=MUTED)
        self.lbl_bearing.configure(text="---.-°", fg=MUTED)
        self.lbl_elevation.configure(text="---.-°", fg=MUTED)
        self.lbl_coords.configure(text="X: -.--  Y: -.--", fg=MUTED)
        if preserve_camera:
            if not self.var_vision_enabled.get():
                self._show_camera_text("CAMERA OFF")
            elif self.camera_photo is None:
                self._show_camera_text("LOADING CAMERA & AI...")
        else:
            self._show_camera_text("NO CAMERA FRAME" if self.var_vision_enabled.get() else "CAMERA OFF")
        self.canvas_spec.delete("all")
        self._redraw_static_views()

    def _log(self, text: str):
        self._log_history.append(text)
        if len(self._log_history) > 500:
            self._log_history = self._log_history[-500:]
        self.lbl_last_log.configure(text=text[:85])
        if self._log_widget is not None and self._log_window is not None and self._log_window.winfo_exists():
            self._log_widget.configure(state="normal")
            self._log_widget.insert("end", text + "\n")
            self._log_widget.see("end")
            self._log_widget.configure(state="disabled")

    def _run_bg(self, label: str, cmd: List[str], on_done=None, collect_lines: bool = False):
        password = self._password()
        collected: List[str] = []

        def worker():
            self._ui_queue.put(("log", f"--- {label} ---"))

            def handle_line(line: str):
                if collect_lines:
                    collected.append(line)
                self._ui_queue.put(("line", line))

            rc = jetson_ctl.run_command_streaming(cmd, handle_line, password=password)
            self._ui_queue.put(("done", (label, rc, collected, on_done)))

        threading.Thread(target=worker, daemon=True).start()

    def _drain_queue(self):
        try:
            while True:
                kind, payload = self._ui_queue.get_nowait()
                if kind == "log":
                    self._log(payload)
                elif kind == "line":
                    if payload.startswith("@JSON:"):
                        self._handle_json_line(payload[6:])
                    elif payload.startswith("@DISCOVERY:"):
                        self._handle_discovery_line(payload[11:])
                    elif payload.startswith("@PREVIEW:"):
                        if not self._cli_running and self.var_vision_enabled.get() and self.var_stream_video.get():
                            self._show_camera_frame_b64(payload[9:])
                    else:
                        self._log(payload)
                elif kind == "preview_stopped":
                    self._preview_running = False
                    if self._preview_restart:
                        self._preview_restart = False
                        self._start_preview_if_needed()
                elif kind == "done":
                    label, rc, collected, cb = payload
                    self._log(f"--- {label} finished (exit {rc}) ---")
                    if rc != 0:
                        self._set_disconnected()
                    elif cb is None:
                        self.conn_status.configure(text="CONNECTED", bg="#1b3a24", fg=OK)
                    if cb is not None:
                        cb(rc, collected)
                elif kind == "cli_stopped":
                    rc = payload
                    self._cli_running = False
                    self.btn_toggle.configure(text="START LISTENING", bg="#238636", activebackground="#2ea043")
                    if self._restart_after_stop:
                        self._restart_after_stop = False
                        self.on_start_cli()
                    elif rc != 0 and not self._cli_stop_requested:
                        self._set_disconnected()
                    else:
                        self._reset_live_telemetry("STANDBY", preserve_camera=True)
                        self._start_preview_if_needed()
                    self._log(f"--- Remote Jetson stopped (exit {rc}) ---")
        except queue.Empty:
            pass
        self.root.after(50, self._drain_queue)

    def _handle_discovery_line(self, raw_json: str):
        if raw_json.startswith("@DISCOVERY:"):
            raw_json = raw_json[11:]
        try:
            data = json.loads(raw_json)
        except json.JSONDecodeError:
            return
        self.conn_status.configure(text="CONNECTED", bg="#1b3a24", fg=OK)
        cfg = data.get("config", {})
        devs = data.get("audio_devices", [])
        cams = data.get("cameras", [])

        for d in devs:
            if "label" not in d:
                d["label"] = f"[{d.get('index', '?')}] {d.get('name', '')} ({d.get('channels', '?')}ch)"
        self._audio_devices = devs
        dev_labels = [d["label"] for d in devs]
        self.combo_dev_front.configure(values=dev_labels)
        self.combo_dev_left.configure(values=dev_labels)

        arrays = cfg.get("arrays", {})
        front_idx = int(arrays.get("front", {}).get("device_index", 4))
        left_idx = int(arrays.get("left", {}).get("device_index", 0))
        front_match = next((d["label"] for d in devs if d["index"] == front_idx), str(front_idx))
        left_match = next((d["label"] for d in devs if d["index"] == left_idx), str(left_idx))
        self.var_dev_front.set(front_match)
        self.var_dev_left.set(left_match)

        det = cfg.get("default_detection", {})
        self.var_low.set(str(det.get("target_low_hz", 200.0)))
        self.var_high.set(str(det.get("target_high_hz", 5000.0)))
        self.var_threshold.set(str(det.get("threshold_dbfs", -70.0)))
        self.var_avg.set(str(det.get("average_blocks", 4)))

        vis = cfg.get("vision", {})
        self._camera_fovs = dict(vis.get("camera_fovs", {}))
        if cams:
            self.camera_combo.configure(values=cams)
        cur_cam = str(vis.get("camera", "")).strip()
        if cur_cam and not cur_cam.startswith("/dev/video0/dev/"):
            self.var_camera.set(cur_cam)
        elif cams:
            self.var_camera.set(cams[0])
        fov_val = vis.get("horizontal_fov_deg", 75.0)
        self.var_camera_fov.set("" if fov_val is None else f"{float(fov_val):g}")

        sup = cfg.get("uma16_front_only", {}).get("direction_suppression", {})
        self.var_sup_enabled.set(bool(sup.get("enabled", False)))
        self.var_sup_bearing.set(f"{float(sup.get('bearing_deg', 60.0)):g}")
        self.var_sup_tol.set(f"{float(sup.get('tolerance_deg', 10.0)):g}")
        self.var_classifier.set(bool(cfg.get("classifier", {}).get("enabled", True)))
        self.var_tracker.set(bool(vis.get("tracker", {}).get("enabled", True)))

        if not self._cli_running:
            self._redraw_static_views()
            self._start_preview_if_needed()
        self._log(f"Discovered {len(devs)} audio device(s) and {len(cams)} camera(s) on Jetson.")

    def _handle_json_line(self, raw_json: str):
        try:
            data = json.loads(raw_json)
        except json.JSONDecodeError:
            return
        self.conn_status.configure(text="CONNECTED", bg="#1b3a24", fg=OK)

        fused = data.get("fused", {})
        det = data.get("detection")
        clf = data.get("classification", {})
        cam = data.get("camera", {})
        single = data.get("single_doa")
        tri = data.get("triangulation")
        sup = data.get("suppression") or {}
        array_mode = data.get("array_mode", "dual")

        # Keep geometry synced with the active array_mode
        self.geo.set_uma16_front_only(array_mode == "uma16_front_only")

        # 1. Status & Tracker Mode
        mode = fused.get("mode", "SEARCHING")
        source = fused.get("source", "NONE")
        if data.get("audio_waiting"):
            self.status_lbl.configure(text="WAITING FOR AUDIO INPUT", fg=WARN, bg="#2b2010")
        elif data.get("suppressed"):
            self.status_lbl.configure(text="FIXED DIRECTION SUPPRESSED", fg=MUTED, bg="#1a2733")
        elif mode == "VISION_TRACKING":
            self.status_lbl.configure(text="VISION TRACKING", fg="#c678dd", bg="#241932")
        elif mode == "ACOUSTIC_TRACKING":
            if clf.get("is_drone"):
                self.status_lbl.configure(text="TARGET LOCKED - DRONE CONFIRMED", fg=DRONE_RED, bg="#3b1d1d")
            else:
                self.status_lbl.configure(text="ACOUSTIC TRACKING", fg=OK, bg="#1a2733")
        else:
            self.status_lbl.configure(text="SCANNING", fg=ACCENT, bg="#1a2733")

        if mode == "VISION_TRACKING":
            b = fused.get("bearing_deg")
            self.lbl_tracker_mode.configure(text=f"VISION {b:.1f}°" if b is not None else "VISION", fg="#c678dd")
        elif mode == "ACOUSTIC_TRACKING":
            b = fused.get("bearing_deg")
            self.lbl_tracker_mode.configure(text=f"SOUND {b:.1f}°" if b is not None else "SOUND", fg=ACCENT)
        elif fused.get("vision_uncalibrated"):
            self.lbl_tracker_mode.configure(text="VISION UNCALIB", fg=WARN)
        else:
            self.lbl_tracker_mode.configure(text="SEARCHING", fg=MUTED)

        # 2. Frequency, Level, Classification, Camera
        if det and det.get("peak_freq") is not None:
            self.lbl_freq.configure(text=f"{det['peak_freq']:6.1f} Hz", fg=WARN if det.get("detected") else FG)
        else:
            self.lbl_freq.configure(text="---.- Hz", fg=MUTED)
        if det and det.get("peak_db") is not None:
            self.lbl_level.configure(text=f"{det['peak_db']:5.1f} dBFS", fg=WARN if det.get("detected") else FG)
        else:
            self.lbl_level.configure(text="---.- dBFS", fg=MUTED)

        clf_text = clf.get("text", "OFF")
        self.lbl_classification.configure(text=clf_text, fg=DRONE_RED if clf.get("is_drone") else FG)
        cam_status = cam.get("status", "OFF")
        if cam.get("target_detected") and cam.get("bearing_deg") is not None:
            cam_text = f"{cam_status} @ {cam['bearing_deg']:.1f}°"
            cam_color = DRONE_RED
        elif cam.get("target_detected"):
            cam_text = f"{cam_status} NO FOV"
            cam_color = DRONE_RED
        else:
            cam_text = cam_status
            cam_color = ERR if cam_status == "OFFLINE" else (UNKNOWN_AMBER if cam_status == "SEARCHING" else ACCENT)
        self.lbl_vision.configure(text=cam_text, fg=cam_color)

        # 3. Bearing, Elevation, Range, Coords
        b_deg = fused.get("bearing_deg")
        el_deg = fused.get("elevation_deg")
        self.lbl_bearing.configure(text=f"{b_deg:5.1f}°" if b_deg is not None else "---.-°", fg=FG if b_deg is not None else MUTED)
        if el_deg is not None:
            vert = "ABOVE" if el_deg > 2.0 else "BELOW" if el_deg < -2.0 else "LEVEL"
            self.lbl_elevation.configure(text=f"{el_deg:+.1f}° {vert}", fg=ACCENT)
        elif array_mode in ("uma16_only", "uma16_front_only"):
            self.lbl_elevation.configure(text="UNCERTAIN", fg=MUTED)
        else:
            self.lbl_elevation.configure(text="N/A", fg=MUTED)

        if tri and tri.get("success") and tri.get("x") is not None and tri.get("y") is not None:
            self.lbl_range.configure(text=f"{tri['range_m']:4.2f} m", fg=OK)
            self.lbl_coords.configure(text=f"X:{tri['x']:+4.2f}m  Y:{tri['y']:+4.2f}m", fg=OK)
        else:
            self.lbl_range.configure(text="-.-- m", fg=MUTED)
            self.lbl_coords.configure(text="DIRECTION ONLY" if single else "X: -.--  Y: -.--", fg=MUTED)

        # 4. Spectrum
        if det and det.get("freqs") and det.get("mags"):
            draw_spectrum_canvas(
                self.canvas_spec,
                freqs=det["freqs"],
                mags=det["mags"],
                band_low=det["band"][0],
                band_high=det["band"][1],
                threshold_db=det["threshold_db"],
                peak_freq=det.get("peak_freq"),
            )

        # 5. Camera Frame
        frame_b64 = cam.get("frame_jpg_b64")
        if frame_b64:
            self._show_camera_frame_b64(frame_b64)
        elif not self.var_vision_enabled.get():
            self._show_camera_text("CAMERA OFF")
        elif not self.var_stream_video.get():
            self._show_camera_text(f"VIDEO STREAM OFF ({cam_status})")

        # 6. 2D Tactical Map
        selected_ids = (
            ["front"] if array_mode == "uma8_only"
            else ["left"] if array_mode in ("uma16_only", "uma16_front_only")
            else list(self.geo.arrays)
        )
        single_bearing = None
        single_label = None
        if single and single.get("global_angle_deg") is not None and (single.get("prominence", 0.0) >= 1.4 or single.get("stable")):
            single_bearing = single["global_angle_deg"]
            single_label = "UMA-8" if single.get("array_id") == "front" else ("UMA-16 FRONT" if array_mode == "uma16_front_only" else "UMA-16")
        draw_radar_canvas(
            self.canvas_radar,
            geo=self.geo,
            map_scale=self.map_scale,
            selected_ids=selected_ids,
            suppression_enabled=bool(sup.get("enabled", False) and array_mode == "uma16_front_only"),
            suppression_bearing=float(sup.get("bearing_deg", 0.0)),
            suppression_tolerance=float(sup.get("tolerance_deg", 10.0)),
            single_bearing=single_bearing,
            single_array_id=single.get("array_id") if single else None,
            single_label=single_label,
            tri_x=tri.get("x") if tri and tri.get("success") else None,
            tri_y=tri.get("y") if tri and tri.get("success") else None,
            tri_label="DRONE" if clf.get("is_drone") else "TARGET",
            drone_confirmed=bool(clf.get("is_drone")),
            camera_bearing=cam.get("bearing_deg"),
            camera_elevation=cam.get("elevation_deg"),
        )

        # 7. 3D Direction (matches Local behavior + supports Vision 3D vector)
        vec = fused.get("direction_vector")
        if array_mode not in ("uma16_only", "uma16_front_only") and source != "VISION":
            draw_direction_3d_canvas(
                self.canvas_direction,
                direction_vector=None,
                bearing_deg=None,
                elevation_deg=None,
                title="SOUND SOURCE DIRECTION",
                blocked_message="SELECT UMA-16 ONLY OR UMA-16 FRONT ONLY",
                default_size=(680, 240),
            )
        else:
            draw_direction_3d_canvas(
                self.canvas_direction,
                direction_vector=tuple(vec) if vec is not None else None,
                bearing_deg=b_deg,
                elevation_deg=el_deg,
                title=f"JETSON 3D DIRECTION ({source})" if source != "NONE" else "SOUND SOURCE DIRECTION",
                empty_title="WAITING",
                empty_subtitle="ELEVATION UNCERTAIN",
                empty_note="No direction arrow",
                arrow_color="#c678dd" if source == "VISION" else DRONE_RED,
                default_size=(680, 240),
            )

    def on_refresh_all(self):
        target = self._target()
        remote_cmd = jetson_ctl.build_remote_discovery_script(target.remote_dir)
        self._run_bg("Discover Jetson Hardware & Config", jetson_ctl.build_ssh_command(target, remote_cmd))

    def on_test_conn(self):
        self.on_refresh_all()

    def on_load_config(self):
        self.on_refresh_all()

    def on_list_devices(self):
        self.on_refresh_all()

    def on_list_cameras(self):
        self.on_refresh_all()

    def on_sync(self):
        target = self._target()
        self._run_bg("Sync Code → Jetson", jetson_ctl.build_rsync_command(target, include_config=False))

    def on_run_tests(self):
        target = self._target()
        self.on_open_console()
        remote_cmd = (
            f"cd {shlex.quote(target.remote_dir)} && .venv/bin/python -m unittest "
            "test_state_machine test_classifier test_vertical_uma16 "
            "test_vision test_direction_suppression test_vision_tracking"
        )
        self._run_bg("Jetson Unit Tests", jetson_ctl.build_ssh_command(target, remote_cmd))

    def on_apply_config(self):
        try:
            argv = [
                "config", "set",
                "--dev-front", self._extract_dev_id(self.var_dev_front.get()),
                "--dev-left", self._extract_dev_id(self.var_dev_left.get()),
                "--low", self.var_low.get().strip(),
                "--high", self.var_high.get().strip(),
                "--threshold", self.var_threshold.get().strip(),
                "--average", self.var_avg.get().strip(),
                "--camera", self.var_camera.get().strip(),
                "--suppress-bearing", self.var_sup_bearing.get().strip(),
                "--suppress-tolerance", self.var_sup_tol.get().strip(),
                "--suppress-enable" if self.var_sup_enabled.get() else "--suppress-disable",
                "--classifier-on" if self.var_classifier.get() else "--classifier-off",
                "--tracker-on" if self.var_tracker.get() else "--tracker-off",
            ]
            if self.var_camera_fov.get().strip():
                argv.extend(["--camera-fov", self.var_camera_fov.get().strip()])
            args = jetson_ctl.build_parser().parse_args(argv)
            patch = jetson_ctl.build_config_patch(args)
        except (SystemExit, ValueError) as exc:
            self._log(f"Invalid config value: {exc}")
            return

        target = self._target()
        remote_cmd = jetson_ctl._remote_config_update_script(target.remote_dir, patch)
        self._run_bg("Apply Config to Jetson", jetson_ctl.build_ssh_command(target, remote_cmd))

    def _build_run_flags(self) -> List[str]:
        flags: List[str] = ["--stream-json"]
        if self.var_stream_video.get() and self.var_vision_enabled.get():
            flags.append("--stream-video")
        if self.var_front_only.get():
            flags.append("--uma8-only")
        elif self.var_left_only.get():
            flags.append("--uma16-only")
        elif self.var_uma16_front_only.get():
            flags.append("--uma16-front-only")

        dev_front = self._extract_dev_id(self.var_dev_front.get())
        dev_left = self._extract_dev_id(self.var_dev_left.get())
        if dev_front:
            flags.extend(["--dev-front", dev_front])
        if dev_left:
            flags.extend(["--dev-left", dev_left])

        flags.extend(["--low", self.var_low.get().strip()])
        flags.extend(["--high", self.var_high.get().strip()])
        flags.extend(["--threshold", self.var_threshold.get().strip()])
        flags.extend(["--average", self.var_avg.get().strip()])

        if not self.var_vision_enabled.get():
            flags.append("--no-vision")
        else:
            flags.extend(["--camera", self.var_camera.get().strip()])
            if self.var_camera_fov.get().strip():
                flags.extend(["--camera-fov", self.var_camera_fov.get().strip()])
            if not self.var_tracker.get():
                flags.append("--no-tracker")

        if not self.var_classifier.get():
            flags.append("--no-classifier")
        elif self.var_require_drone.get():
            flags.append("--require-drone")

        if self.var_simulate.get():
            flags.append("--simulate")
        if self.var_uma16_front_only.get() and self.var_sup_enabled.get():
            flags.extend(["--suppress-bearing", self.var_sup_bearing.get().strip()])
            flags.extend(["--suppress-tolerance", self.var_sup_tol.get().strip()])
        return flags

    def on_toggle_listening(self):
        if not self._cli_running:
            self.on_start_cli()
        else:
            self.on_stop_cli()

    def _restart_cli(self):
        if self._cli_running:
            self._restart_after_stop = True
            self.on_stop_cli()

    def _start_preview_if_needed(self):
        if (
            self._preview_running
            or self._cli_running
            or not self._tab_active
            or not self.var_vision_enabled.get()
            or not self.var_stream_video.get()
        ):
            return
        cam = self.var_camera.get().strip()
        if not cam:
            return
        self._preview_running = True
        self._preview_stop_requested = False
        target = self._target()
        remote_cmd = jetson_ctl.build_remote_camera_preview_command(target, cam)
        ssh_cmd = jetson_ctl.build_ssh_command(target, remote_cmd, tty=True)
        password = self._password()

        def worker():
            rc = jetson_ctl.run_command_streaming(
                ssh_cmd,
                lambda line: self._ui_queue.put(("line", line)),
                password=password,
                stop_flag=lambda: self._preview_stop_requested or self._cli_running,
            )
            self._ui_queue.put(("preview_stopped", rc))

        threading.Thread(target=worker, daemon=True).start()

    def _stop_preview(self):
        self._preview_restart = False
        was_running = self._preview_running
        self._preview_stop_requested = True
        if was_running:
            target = self._target()
            cmd = jetson_ctl.build_ssh_command(target, "pkill -f '[@]PREVIEW:' || true")
            threading.Thread(
                target=lambda: jetson_ctl.run_command_streaming(cmd, lambda _l: None, password=self._password()),
                daemon=True,
            ).start()

    def _restart_preview(self):
        if self._preview_running:
            self._preview_restart = True
            self._preview_stop_requested = True
            target = self._target()
            cmd = jetson_ctl.build_ssh_command(target, "pkill -f '[@]PREVIEW:' || true")
            threading.Thread(
                target=lambda: jetson_ctl.run_command_streaming(cmd, lambda _l: None, password=self._password()),
                daemon=True,
            ).start()
        else:
            self._start_preview_if_needed()

    def on_start_cli(self):
        if self._cli_running:
            return
        self._stop_preview()
        self._cli_running = True
        self._cli_stop_requested = False
        self.btn_toggle.configure(text="STOP", bg=WARN, activebackground="#da3633")
        self._reset_live_telemetry("SEARCHING...", preserve_camera=True)

        target = self._target()
        remote_cmd = jetson_ctl.build_remote_cli_command(target, self._build_run_flags())
        ssh_cmd = jetson_ctl.build_ssh_command(target, remote_cmd, tty=True)
        password = self._password()

        def worker():
            self._ui_queue.put(("log", "--- Starting Remote Jetson Stream ---"))
            rc = jetson_ctl.run_command_streaming(
                ssh_cmd,
                lambda line: self._ui_queue.put(("line", line)),
                password=password,
                stop_flag=lambda: self._cli_stop_requested,
            )
            self._ui_queue.put(("cli_stopped", rc))

        threading.Thread(target=worker, daemon=True).start()

    def on_stop_cli(self):
        self._cli_stop_requested = True
        target = self._target()
        cmd = jetson_ctl.build_ssh_command(target, "pkill -INT -f 'main.py cli' || true")
        threading.Thread(
            target=lambda: jetson_ctl.run_command_streaming(cmd, lambda _l: None, password=self._password()),
            daemon=True,
        ).start()

    def stop_if_running(self):
        self._restart_after_stop = False
        self._stop_preview()
        if self._cli_running:
            self.on_stop_cli()
