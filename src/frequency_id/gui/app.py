#!/usr/bin/env python3
"""Tactical GUI main window for local acoustic tracking, vision, and remote Jetson control."""

from dataclasses import replace
import json
import queue
import time
import tkinter as tk
from tkinter import ttk
from typing import Dict

import numpy as np
from PIL import Image, ImageTk

from frequency_id.acoustics.audio_manager import MultiDeviceAudioManager
from frequency_id.acoustics.detector import DetectionResult, FrequencyDetector
from frequency_id.acoustics.direction_suppression import (
    DirectionSuppression,
    displayable_doa,
    select_front_only_target,
)
from frequency_id.acoustics.doa_engine import ArrayDOA
from frequency_id.acoustics.doa_factory import create_doa_engine
from frequency_id.acoustics.geometry import RigGeometry
from frequency_id.acoustics.triangulation import TriangulationResult, Triangulator
from frequency_id.control.state_machine import (
    StateMachineConfig,
    TargetStateMachine,
    TrackingMode,
)
from frequency_id.gui.jetson_tab import JetsonRemoteTab
from frequency_id.gui.px4_launcher import PX4_URI_PRESETS, PX4FlightTestMixin, sync_px4_bench_toggles
from frequency_id.gui.sim_launcher import SimulationLauncherMixin
from frequency_id.gui.widgets import (
    ACCENT,
    BG,
    BORDER,
    DRONE_RED,
    FG,
    MUTED,
    OK,
    PANEL,
    UNKNOWN_AMBER,
    WARN,
    draw_direction_3d_canvas,
    draw_radar_canvas,
    draw_spectrum_canvas,
)
from frequency_id.vision.detector import (
    discover_cameras,
    normalize_camera_source,
    validate_horizontal_fov,
)


class TacticalGUI(SimulationLauncherMixin, PX4FlightTestMixin):
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
        for arr_id in self.geo.arrays:
            self.doa_engines[arr_id] = create_doa_engine(
                self.geo,
                arr_id,
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
        self.sim_process = None
        self._sim_paused_vision = False
        self._sim_paused_audio = False
        self.px4_test_process = None
        self._px4_queue: queue.Queue = queue.Queue()
        self._px4_terminal_event = None
        self._px4_reader_thread = None

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
        self.sim_button = tk.Button(
            action_row,
            text="RUN SIMULATION",
            font=("Helvetica", 10, "bold"),
            bg="#1f6feb",
            fg="#ffffff",
            activebackground="#388bfd",
            padx=12,
            pady=4,
            command=self.on_run_simulation,
            relief="flat",
        )
        self.sim_button.pack(side="left", padx=(14, 4))
        self.sim_stop_button = tk.Button(
            action_row,
            text="STOP SIM",
            font=("Helvetica", 10, "bold"),
            bg=WARN,
            fg="#ffffff",
            activebackground="#da3633",
            padx=10,
            pady=4,
            command=self.on_stop_simulation,
            relief="flat",
            state="disabled",
        )
        self.sim_stop_button.pack(side="left", padx=(0, 6))
        self.sim_status_lbl = tk.Label(
            action_row,
            text="SIM READY",
            font=("Helvetica", 9, "bold"),
            bg=PANEL,
            fg=MUTED,
        )
        self.sim_status_lbl.pack(side="left", padx=4)
        tk.Label(action_row, text="Map Zoom:", bg=PANEL, fg=FG).pack(side="right", padx=(4, 2))
        self.slider_zoom = tk.Scale(
            action_row, from_=30, to=200, orient="horizontal", showvalue=True, bg=PANEL,
            fg=FG, troughcolor="#0d1117", highlightthickness=0, length=110,
            command=self.on_zoom_change,
        )
        self.slider_zoom.set(int(self.map_scale))
        self.slider_zoom.pack(side="right", padx=4)

        px4_cfg = self.geo.cfg.get("px4", {}) if isinstance(self.geo.cfg.get("px4"), dict) else {}
        px4_row = tk.Frame(ctrl, bg=PANEL)
        px4_row.pack(fill="x", padx=8, pady=(0, 4))
        tk.Label(px4_row, text="PX4 URI:", bg=PANEL, fg=FG).pack(side="left", padx=(4, 2))
        self.var_px4_connect = tk.StringVar(value=str(px4_cfg.get("connect", "udp://:14540")))
        self.px4_connect_combo = ttk.Combobox(
            px4_row,
            textvariable=self.var_px4_connect,
            values=list(PX4_URI_PRESETS),
            width=26,
        )
        self.px4_connect_combo.pack(side="left", padx=(0, 8))
        tk.Label(px4_row, text="Alt (m):", bg=PANEL, fg=FG).pack(side="left", padx=(2, 2))
        self.var_px4_alt = tk.StringVar(value=f"{float(px4_cfg.get('takeoff_altitude_m', 2.5)):g}")
        self.px4_alt_entry = tk.Entry(
            px4_row,
            textvariable=self.var_px4_alt,
            width=5,
            bg="#0d1117",
            fg=FG,
            insertbackground=FG,
        )
        self.px4_alt_entry.pack(side="left", padx=(0, 8))
        self.var_px4_auto_land = tk.BooleanVar(value=bool(px4_cfg.get("auto_land", True)))
        tk.Checkbutton(
            px4_row,
            text="Auto-Land",
            variable=self.var_px4_auto_land,
            bg=PANEL,
            fg=FG,
            selectcolor="#0d1117",
            activebackground=PANEL,
            activeforeground=FG,
        ).pack(side="left", padx=(0, 6))
        self.var_px4_arm_only = tk.BooleanVar(value=False)
        self.var_px4_virtual_rc = tk.BooleanVar(value=False)
        tk.Checkbutton(
            px4_row,
            text="Arm Only (Bench)",
            variable=self.var_px4_arm_only,
            bg=PANEL,
            fg=FG,
            selectcolor="#0d1117",
            activebackground=PANEL,
            activeforeground=FG,
            command=lambda: sync_px4_bench_toggles(self.var_px4_arm_only, self.var_px4_virtual_rc, "arm_only"),
        ).pack(side="left", padx=(0, 6))
        tk.Checkbutton(
            px4_row,
            text="Virtual RC (Bench)",
            variable=self.var_px4_virtual_rc,
            bg=PANEL,
            fg=FG,
            selectcolor="#0d1117",
            activebackground=PANEL,
            activeforeground=FG,
            command=lambda: sync_px4_bench_toggles(self.var_px4_arm_only, self.var_px4_virtual_rc, "virtual_rc"),
        ).pack(side="left", padx=(0, 10))
        self.px4_test_button = tk.Button(
            px4_row,
            text="TEST ARM & TAKEOFF",
            font=("Helvetica", 10, "bold"),
            bg="#9e6a03",
            fg="#ffffff",
            activebackground="#bb8009",
            padx=12,
            pady=3,
            command=self.on_test_arm_takeoff,
            relief="flat",
        )
        self.px4_test_button.pack(side="left", padx=(0, 4))
        self.px4_stop_button = tk.Button(
            px4_row,
            text="LAND / STOP",
            font=("Helvetica", 10, "bold"),
            bg=WARN,
            fg="#ffffff",
            activebackground="#da3633",
            padx=10,
            pady=3,
            command=self.on_stop_px4_test,
            relief="flat",
            state="disabled",
        )
        self.px4_stop_button.pack(side="left", padx=(0, 8))
        self.px4_status_lbl = tk.Label(
            px4_row,
            text="PX4 IDLE",
            font=("Helvetica", 9, "bold"),
            bg=PANEL,
            fg=MUTED,
        )
        self.px4_status_lbl.pack(side="left", padx=4)

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
        self.doa_engines["left"] = create_doa_engine(
            self.geo,
            "left",
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
                float(self.var_low.get())
                float(self.var_high.get())
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
            if hasattr(self, "var_px4_connect"):
                px4_section = self.geo.cfg.setdefault("px4", {})
                connect_uri = self.var_px4_connect.get().strip()
                if connect_uri:
                    px4_section["connect"] = connect_uri
                if hasattr(self, "var_px4_alt"):
                    try:
                        alt = float(self.var_px4_alt.get().strip())
                        if 0.5 <= alt <= 5.0:
                            px4_section["takeoff_altitude_m"] = alt
                    except (TypeError, ValueError):
                        pass
                if hasattr(self, "var_px4_auto_land"):
                    px4_section["auto_land"] = bool(self.var_px4_auto_land.get())
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
        det_ch = det_block.samples[:, 0]
        det_res = self.detector.process(det_ch, low, high)
        front_doa = None
        if self.var_uma16_front_only.get():
            selection = select_front_only_target(
                det_res,
                blocks["left"].samples,
                self.doa_engines["left"],
                (low, high),
                self._current_suppression(),
                captured_at=blocks["left"].captured_at,
                now=time.monotonic(),
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
                    blocks[arr_id].samples,
                    target_freq=target_f,
                    band=(low, high),
                    captured_at=blocks[arr_id].captured_at,
                    now=now,
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
        self._stop_px4_test()
        self._stop_simulation(restore_devices=False)
        self.jetson_remote.stop_if_running()
        self.audio_mgr.stop()
        if self.vision is not None:
            self.vision.stop()
        self.root.destroy()
