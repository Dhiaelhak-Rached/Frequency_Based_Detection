import json
from pathlib import Path
import unittest
from unittest.mock import patch

import jetson_ctl


class TestJetsonCtl(unittest.TestCase):
    def test_build_rsync_command_excludes_rig_config_by_default(self):
        target = jetson_ctl.JetsonTarget()
        cmd = jetson_ctl.build_rsync_command(target, local_dir=Path("/tmp/src"))
        joined = " ".join(cmd)
        self.assertIn("--exclude rig_config.json", joined)
        self.assertIn("--exclude .venv", joined)
        self.assertIn("ssh -p 2222", joined)
        self.assertTrue(cmd[-1].endswith("meltem@41.228.129.118:/home/meltem/frequency_id/"))

    def test_build_rsync_command_includes_config_when_requested(self):
        target = jetson_ctl.JetsonTarget()
        cmd = jetson_ctl.build_rsync_command(
            target, local_dir=Path("/tmp/src"), include_config=True, dry_run=True
        )
        self.assertNotIn("rig_config.json", cmd)
        self.assertIn("--dry-run", cmd)

    def test_build_remote_cli_command_foreground_and_detach(self):
        target = jetson_ctl.JetsonTarget()
        fg = jetson_ctl.build_remote_cli_command(
            target, ["--uma16-front-only", "--low", "900", "--high", "1100"]
        )
        self.assertIn("PYTHONUNBUFFERED=1 .venv/bin/python -u main.py cli", fg)
        self.assertIn("--uma16-front-only --low 900 --high 1100", fg)
        self.assertNotIn("nohup", fg)

        bg = jetson_ctl.build_remote_cli_command(target, ["--simulate"], detach=True)
        self.assertIn("nohup bash -lc", bg)
        self.assertIn("/tmp/frequency_id_cli.log", bg)

    def test_config_patch_validation_and_fields(self):
        parser = jetson_ctl.build_parser()
        args = parser.parse_args([
            "config", "set",
            "--dev-front", "2",
            "--dev-left", "5",
            "--low", "800",
            "--high", "1200",
            "--camera", "/dev/video0",
            "--camera-fov", "75",
            "--suppress-enable",
            "--suppress-bearing", "390",
            "--suppress-tolerance", "12",
            "--classifier-off",
            "--tracker-on",
        ])
        patch_dict = jetson_ctl.build_config_patch(args)
        self.assertEqual(patch_dict["dev_front"], 2)
        self.assertEqual(patch_dict["dev_left"], 5)
        self.assertEqual(patch_dict["low"], 800.0)
        self.assertEqual(patch_dict["high"], 1200.0)
        self.assertEqual(patch_dict["camera"], "/dev/video0")
        self.assertEqual(patch_dict["camera_fov"], 75.0)
        self.assertTrue(patch_dict["suppress_enabled"])
        self.assertEqual(patch_dict["suppress_bearing"], 30.0)
        self.assertEqual(patch_dict["suppress_tolerance"], 12.0)
        self.assertFalse(patch_dict["classifier_enabled"])
        self.assertTrue(patch_dict["tracker_enabled"])

    def test_config_patch_rejects_invalid_values(self):
        parser = jetson_ctl.build_parser()
        bad_fov = parser.parse_args(["config", "set", "--camera-fov", "180"])
        with self.assertRaises(ValueError):
            jetson_ctl.build_config_patch(bad_fov)

        bad_band = parser.parse_args(["config", "set", "--low", "1200", "--high", "900"])
        with self.assertRaises(ValueError):
            jetson_ctl.build_config_patch(bad_band)

        empty = parser.parse_args(["config", "set"])
        with self.assertRaises(ValueError):
            jetson_ctl.build_config_patch(empty)

    def test_run_forwards_cli_args_with_tty_enabled(self):
        with patch("jetson_ctl.subprocess.run") as mock_run:
            mock_run.return_value.returncode = 0
            rc = jetson_ctl.main(["run", "--uma16-front-only", "--low", "900", "--high", "1100"])
            self.assertEqual(rc, 0)
            called_cmd = mock_run.call_args[0][0]
            self.assertEqual(called_cmd[0], "ssh")
            self.assertIn("-tt", called_cmd)
            self.assertIn("--uma16-front-only --low 900 --high 1100", called_cmd[-1])

    def test_parse_cli_block_and_gui_import(self):
        import tactical_gui

        self.assertTrue(hasattr(tactical_gui, "TacticalGUI"))
        self.assertTrue(hasattr(tactical_gui, "JetsonRemoteTab"))
        self.assertTrue(hasattr(tactical_gui, "draw_direction_3d_canvas"))
        parsed = jetson_ctl.parse_cli_block([
            "MODE: VISION_TRACKING",
            "SOURCE: VISION",
            "BEARING: 45.2°",
            "ELEVATION: 12.4°",
            "DIRECTION_VECTOR: (+0.693, +0.688, +0.215)",
        ])
        self.assertEqual(parsed["MODE"], "VISION_TRACKING")
        self.assertEqual(parsed["SOURCE"], "VISION")
        self.assertEqual(parsed["BEARING"], "45.2°")
        self.assertIsNone(jetson_ctl.parse_cli_block(["MODE: SEARCHING"]))

    def test_jetson_remote_tab_handles_json_and_disconnect(self):
        import tkinter as tk
        from geometry import RigGeometry
        import tactical_gui

        try:
            root = tk.Tk()
        except tk.TclError:
            self.skipTest("No X display available")
        root.withdraw()
        try:
            geo = RigGeometry("rig_config.json")
            frame = tk.Frame(root)
            tab = tactical_gui.JetsonRemoteTab(frame, root, geo)
            sample = {
                "type": "telemetry",
                "audio_waiting": False,
                "suppressed": False,
                "array_mode": "dual",
                "fused": {
                    "mode": "ACOUSTIC_TRACKING",
                    "source": "SOUND",
                    "bearing_deg": 346.47,
                    "elevation_deg": None,
                    "direction_vector": None,
                    "vision_uncalibrated": False,
                },
                "detection": {
                    "detected": True,
                    "peak_freq": 999.95,
                    "peak_db": -26.87,
                    "threshold_db": -65.0,
                    "band": [200.0, 5000.0],
                    "freqs": [200.0, 1000.0, 5000.0],
                    "mags": [-80.0, -26.87, -80.0],
                },
                "classification": {"text": "DRONE (92%) [3/5]", "is_drone": True},
                "camera": {
                    "status": "SEARCHING",
                    "target_detected": False,
                    "bearing_deg": None,
                    "elevation_deg": None,
                    "uncalibrated": False,
                    "frame_jpg_b64": None,
                },
                "single_doa": None,
                "triangulation": {
                    "success": True,
                    "x": -0.702,
                    "y": 2.919,
                    "range_m": 3.002,
                    "bearing_deg": 346.47,
                    "bearings": {"front": 346.47, "left": 352.6},
                },
                "suppression": None,
                "doa_errors": {},
            }
            tab._handle_json_line(json.dumps(sample))
            self.assertEqual(tab.lbl_freq.cget("text").strip(), "1000.0 Hz")
            self.assertEqual(tab.lbl_range.cget("text").strip(), "3.00 m")
            self.assertEqual(tab.lbl_bearing.cget("text").strip(), "346.5°")
            tab._set_disconnected()
            self.assertEqual(tab.conn_status.cget("text"), "DISCONNECTED")
            self.assertEqual(tab.lbl_bearing.cget("text"), "---.-°")

            # Verify UMA-16 Front Only geometry update in JetsonRemoteTab
            self.assertEqual(tab.geo.arrays["left"].name, "UMA-16 Left")
            tab.var_uma16_front_only.set(True)
            tab.on_array_mode_change("uma16_front")
            self.assertEqual(tab.geo.arrays["left"].name, "UMA-16 Front")
            self.assertEqual(float(tab.geo.arrays["left"].center[0]), 0.0)
            self.assertEqual(float(tab.geo.arrays["left"].center[1]), 0.0)
            self.assertEqual(float(tab.geo.arrays["left"].orientation_deg), 0.0)

            tab.var_uma16_front_only.set(False)
            tab.on_array_mode_change("uma16_front")
            self.assertEqual(tab.geo.arrays["left"].name, "UMA-16 Left")

            # Verify @DISCOVERY: line populates microphone and camera dropdowns
            discovery_payload = {
                "config": {
                    "arrays": {
                        "front": {"device_index": 2},
                        "left": {"device_index": 0},
                    },
                    "default_detection": {"target_low_hz": 250, "target_high_hz": 4500, "average_blocks": 6},
                    "uma16_front_only": {
                        "direction_suppression": {"enabled": True, "bearing_deg": 45.0, "tolerance_deg": 12.0},
                    },
                    "classifier": {"enabled": True},
                    "vision": {
                        "camera": "rtsp://127.0.0.1:8554/drone",
                        "horizontal_fov_deg": 75.0,
                        "tracker": {"enabled": True},
                    },
                },
                "audio_devices": [
                    {"index": 0, "name": "UMA16v2: USB Audio (hw:0,0)", "channels": 16, "sr": 44100},
                    {"index": 2, "name": "miniDSP VocalFusion: USB Audio (hw:1,0)", "channels": 8, "sr": 48000},
                ],
                "cameras": ["rtsp://127.0.0.1:8554/drone", "/dev/video0"],
            }
            tab._handle_discovery_line("@DISCOVERY:" + json.dumps(discovery_payload))
            self.assertEqual(len(tab.combo_dev_left["values"]), 2)
            self.assertIn("[0] UMA16v2: USB Audio (hw:0,0) (16ch)", tab.var_dev_left.get())
            self.assertIn("[2] miniDSP VocalFusion: USB Audio (hw:1,0) (8ch)", tab.var_dev_front.get())
            self.assertEqual(list(tab.camera_combo["values"]), ["rtsp://127.0.0.1:8554/drone", "/dev/video0"])
            self.assertEqual(tab.var_camera.get(), "rtsp://127.0.0.1:8554/drone")
            self.assertEqual(tab.var_avg.get(), "6")
            self.assertTrue(tab.var_tracker.get())

            # Verify camera label sizing switches cleanly between image pixels and text units
            import base64
            import io
            from PIL import Image

            buf = io.BytesIO()
            Image.new("RGB", (320, 180), (10, 20, 30)).save(buf, format="JPEG")
            b64_img = base64.b64encode(buf.getvalue()).decode("ascii")
            tab._show_camera_frame_b64(b64_img)
            self.assertIsNotNone(tab.camera_photo)
            self.assertGreater(int(tab.camera_video_lbl.cget("width")), 48)
            self.assertGreater(int(tab.camera_video_lbl.cget("height")), 11)
            tab._show_camera_text("NO CAMERA FRAME")
            self.assertIsNone(tab.camera_photo)
            self.assertEqual(int(tab.camera_video_lbl.cget("width")), 48)
            self.assertEqual(int(tab.camera_video_lbl.cget("height")), 11)
        finally:
            root.destroy()


if __name__ == "__main__":
    unittest.main()
