#!/usr/bin/env python3
"""Tactical GUI for acoustic tracking and independent YOLO camera detection.

Displays the real-time FFT spectrum, selectable-camera annotated video, drone
telemetry, and the top-down acoustic bearing or triangulated target position.
"""

from dataclasses import replace
import json
import time
import tkinter as tk
from tkinter import ttk
from typing import Dict

import numpy as np
from PIL import Image, ImageTk

from geometry import RigGeometry
from vision_detector import discover_cameras, normalize_camera_source, validate_horizontal_fov
from detector import DetectionResult, FrequencyDetector
from direction_suppression import DirectionSuppression, displayable_doa, select_front_only_target
from doa_engine import ArrayDOA
from triangulation import TriangulationResult, Triangulator
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
        self.lbl_range = self._create_metric(top_frame, "RANGE", "-.-- m")
        self.lbl_bearing = self._create_metric(top_frame, "BEARING", "---.-°")
        self.lbl_elevation = self._create_metric(top_frame, "ELEVATION", "---.-°")
        self.lbl_coords = self._create_metric(top_frame, "COORDINATES (X, Y)", "X: -.--  Y: -.--")

        # 2. Main Visual Body (Spectrum on Left, Radar Map on Right)
        body = tk.Frame(self.root, bg=BG)
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
        ctrl = tk.Frame(self.root, bg=PANEL, highlightbackground=BORDER, highlightthickness=1)
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
        self._update_telemetry(det_res, tri_res, doa_errors, single_doa, single_array_id)
        self._draw_spectrum(det_res)
        self._draw_tactical_map(tri_res, single_doa if det_res.detected else None, single_array_id)
        self._draw_direction_3d(single_doa, single_array_id)

    def _show_suppressed_target(self, detection):
        self._reset_classifier()
        self._update_classification_metric()
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

    def _draw_direction_3d(self, result, array_id):
        c = self.canvas_direction
        c.delete("all")
        w, h = c.winfo_width(), c.winfo_height()
        if w < 10:
            w, h = 680, 260
        c.create_text(16, 12, text="SOUND SOURCE DIRECTION", fill=MUTED, anchor="nw", font=("Helvetica", 9, "bold"))
        if array_id != "left":
            c.create_text(
                w / 2,
                h / 2,
                text="SELECT UMA-16 ONLY OR UMA-16 FRONT ONLY",
                fill=MUTED,
                font=("Helvetica", 13, "bold"),
            )
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
        if result is None or result.direction_vector is None:
            c.create_text(status_x, h * 0.38, text="WAITING", fill=UNKNOWN_AMBER, font=("Helvetica", 22, "bold"))
            c.create_text(status_x, h * 0.58, text="ELEVATION UNCERTAIN", fill=FG, font=("Helvetica", 11, "bold"))
            c.create_text(status_x, h * 0.72, text="No direction arrow", fill=MUTED, font=("Helvetica", 10))
            return

        vector = result.direction_vector
        endpoint = project(vector)
        ground_endpoint = project((vector[0], vector[1], 0.0))
        c.create_line(origin[0], origin[1], ground_endpoint[0], ground_endpoint[1], fill=UNKNOWN_AMBER, width=2)
        c.create_line(ground_endpoint[0], ground_endpoint[1], endpoint[0], endpoint[1], fill=UNKNOWN_AMBER, width=2, dash=(5, 4))
        c.create_line(origin[0], origin[1], endpoint[0], endpoint[1], fill="#7a1f1f", width=10)
        c.create_line(origin[0], origin[1], endpoint[0], endpoint[1], fill=DRONE_RED, width=6, arrow="last", arrowshape=(18, 22, 8))
        c.create_oval(origin[0] - 5, origin[1] - 5, origin[0] + 5, origin[1] + 5, fill=FG, outline="")

        elevation = result.elevation_deg
        vertical = "ABOVE" if elevation > 2.0 else "BELOW" if elevation < -2.0 else "LEVEL"
        bearing = result.global_angle_deg % 360.0
        directions = (
            "FORWARD", "FORWARD RIGHT", "RIGHT", "BACK RIGHT",
            "BACK", "BACK LEFT", "LEFT", "FORWARD LEFT",
        )
        horizontal = directions[int((bearing + 22.5) // 45.0) % 8]
        c.create_text(status_x, 50, text=vertical, fill=DRONE_RED, font=("Helvetica", 24, "bold"))
        c.create_text(status_x, 92, text=f"{elevation:+.1f}°", fill=FG, font=("Monospace", 24, "bold"))
        c.create_text(status_x, 119, text="ELEVATION", fill=MUTED, font=("Helvetica", 9, "bold"))
        c.create_text(status_x, 158, text=horizontal, fill=ACCENT, font=("Helvetica", 16, "bold"), tags=("direction-value",))
        c.create_text(status_x, 187, text=f"AZIMUTH {bearing:.1f}°", fill=FG, font=("Monospace", 11, "bold"))
        c.create_text(status_x, 214, text="Arrow points to the source", fill=MUTED, font=("Helvetica", 9))

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

        suppression = self._current_suppression()
        if suppression.enabled:
            origin = to_canvas(*self.geo.arrays["left"].center)
            for boundary in (
                suppression.bearing_deg - suppression.tolerance_deg,
                suppression.bearing_deg + suppression.tolerance_deg,
            ):
                angle = np.deg2rad(boundary)
                endpoint = (
                    origin[0] + 1.8 * self.map_scale * np.sin(angle),
                    origin[1] - 1.8 * self.map_scale * np.cos(angle),
                )
                c.create_line(
                    *origin,
                    *endpoint,
                    fill=SUPPRESSION_MAGENTA,
                    width=1,
                    dash=(3, 5),
                )
            angle = np.deg2rad(suppression.bearing_deg)
            endpoint = (
                origin[0] + 2.0 * self.map_scale * np.sin(angle),
                origin[1] - 2.0 * self.map_scale * np.cos(angle),
            )
            c.create_line(
                *origin,
                *endpoint,
                fill=SUPPRESSION_MAGENTA,
                width=3,
                dash=(8, 4),
            )
            c.create_text(
                endpoint[0],
                endpoint[1] - 10,
                text=f"SUPPRESSED {suppression.bearing_deg:.0f}° ±{suppression.tolerance_deg:.0f}°",
                fill=SUPPRESSION_MAGENTA,
                font=("Helvetica", 9, "bold"),
            )

        if single_doa is not None and single_doa.global_angle_deg is not None and (single_doa.prominence >= 1.4 or single_doa.stable):
            arr = self.geo.arrays[single_array_id]
            ray_color = RAY_FRONT if single_array_id == "front" else RAY_LEFT
            if single_array_id == "front":
                array_label = "UMA-8"
            elif self.var_uma16_front_only.get():
                array_label = "UMA-16 FRONT"
            else:
                array_label = "UMA-16"
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
            drone_confirmed = self._drone_confirmed()
            color = DRONE_RED if drone_confirmed else UNKNOWN_AMBER
            if drone_confirmed:
                radius = 8.0 + 2.0 * np.sin(time.monotonic() * 6.0)
                label = f"DRONE ({self.classification.probability:.0%})"
            elif self.classification is not None and self.classification.is_ready:
                radius = 6.0
                label = f"TARGET ({self.classification.probability:.0%})"
            else:
                radius = 6.0
                label = "TARGET (UNKNOWN)"
            c.create_line(origin[0], origin[1], tx, ty, fill=color, width=3)
            c.create_oval(tx - radius, ty - radius, tx + radius, ty + radius, fill=color, outline="#ffffff", width=2)
            c.create_text(
                tx + 12,
                ty - 8,
                text=f"{label} ({tri.x:+.1f}m, {tri.y:+.1f}m)",
                fill=color,
                anchor="w",
                font=("Helvetica", 9, "bold"),
            )

        visual = self.vision_result
        if visual is not None and visual.target is not None and visual.direction is not None:
            direction = visual.direction
            angle = np.deg2rad(direction.global_bearing_deg)
            origin = to_canvas(0.0, 0.0)
            endpoint = to_canvas(4.0 * np.sin(angle), 4.0 * np.cos(angle))
            c.create_line(
                origin[0],
                origin[1],
                endpoint[0],
                endpoint[1],
                fill="#c678dd",
                width=2,
                dash=(8, 4),
                arrow="last",
            )
            c.create_text(
                endpoint[0] + 8,
                endpoint[1],
                text=(
                    f"CAMERA {direction.global_bearing_deg:.1f}° "
                    f"(EL {direction.elevation_deg:+.1f}°, DIRECTION ONLY)"
                ),
                fill="#c678dd",
                anchor="w",
                font=("Helvetica", 8, "bold"),
            )

    def on_close(self):
        self.audio_mgr.stop()
        if self.vision is not None:
            self.vision.stop()
        self.root.destroy()
