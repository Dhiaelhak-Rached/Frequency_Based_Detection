"""Embedded Jetson Remote tab mirroring Local Tactical Radar controls and telemetry over SSH."""

import base64
import io
import json
import os
import queue
import re
import shlex
import threading
import tkinter as tk
from tkinter import scrolledtext, ttk
from typing import Dict, List, Optional

from PIL import Image, ImageTk

from frequency_id.acoustics.direction_suppression import DirectionSuppression
from frequency_id.acoustics.geometry import RigGeometry
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
from frequency_id.ops import jetson_ctl
from frequency_id.vision.detector import validate_horizontal_fov


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
        self.var_dir = tk.StringVar(
            value=os.environ.get("JETSON_REMOTE_DIR") or os.environ.get("JETSON_DIR", jetson_ctl.DEFAULT_REMOTE_DIR)
        )
        self.var_pass = tk.StringVar(
            value=os.environ.get("JETSON_PASS") or os.environ.get("SSHPASS") or ""
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
            cam_color = WARN if cam_status == "OFFLINE" else (UNKNOWN_AMBER if cam_status == "SEARCHING" else ACCENT)
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
