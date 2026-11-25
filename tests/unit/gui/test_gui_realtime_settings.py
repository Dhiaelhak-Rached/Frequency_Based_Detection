import json
from pathlib import Path
import tkinter as tk
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from frequency_id.acoustics.audio_manager import AudioBlock
from frequency_id.acoustics.geometry import RigGeometry
from frequency_id.gui.app import TacticalGUI
from frequency_id.vision.detector import VisionResult


EXAMPLE_CONFIG = Path(__file__).resolve().parents[3] / "config" / "rig_config.example.json"


class GuiRealtimeAndSettingsTest(unittest.TestCase):
    def setUp(self):
        try:
            self.root = tk.Tk()
        except tk.TclError:
            self.skipTest("No X display available")
        self.root.withdraw()

    def tearDown(self):
        if hasattr(self, "root") and self.root is not None:
            try:
                self.root.destroy()
            except Exception:
                pass

    def _make_app(self, tmp_path: Path, simulate: bool = False):
        cfg_path = tmp_path / "rig_config.json"
        cfg_path.write_text(EXAMPLE_CONFIG.read_text())
        geo = RigGeometry(str(cfg_path))
        with patch.object(TacticalGUI, "save_config", lambda self_app: None):
            app = TacticalGUI(self.root, geometry=geo, simulate=simulate)
        return app

    def test_settings_changes_update_gui_in_realtime_while_standby(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            app = self._make_app(Path(tmp), simulate=False)

            # Initial visuals should already be populated on startup
            self.assertGreater(len(app.canvas_radar.find_all()), 0)
            self.assertGreater(len(app.canvas_spec.find_all()), 0)

            # 1. Changing threshold and band updates detector parameters and spectrum immediately
            app.var_low.set("600")
            app.var_high.set("2400")
            app.var_thresh.set("-52.5")
            app.var_avg.set("2")
            app.on_apply_band()
            self.assertEqual(app.detector.threshold_dbfs, -52.5)
            self.assertEqual(app.detector.average_blocks, 2)
            spec_texts = [
                app.canvas_spec.itemcget(item, "text")
                for item in app.canvas_spec.find_all()
                if app.canvas_spec.type(item) == "text"
            ]
            self.assertTrue(any("Thresh -52dB" in t or "Thresh -53dB" in t for t in spec_texts))

            # 2. Changing map zoom redraws radar canvas immediately even when not listening
            before_coords = [
                app.canvas_radar.coords(item)
                for item in app.canvas_radar.find_all()
                if app.canvas_radar.type(item) == "oval"
            ]
            app.on_zoom_change("150")
            self.assertEqual(app.map_scale, 150.0)
            after_coords = [
                app.canvas_radar.coords(item)
                for item in app.canvas_radar.find_all()
                if app.canvas_radar.type(item) == "oval"
            ]
            self.assertNotEqual(before_coords, after_coords)

            # 3. Enabling UMA-16 Front Only and applying suppression draws suppression sector immediately
            app.var_uma16_front_only.set(True)
            app.on_array_mode_change("uma16_front")
            app.var_suppress_enabled.set(True)
            app.var_suppress_bearing.set("45")
            app.var_suppress_tolerance.set("12")
            app.on_apply_suppression()
            radar_texts = [
                app.canvas_radar.itemcget(item, "text")
                for item in app.canvas_radar.find_all()
                if app.canvas_radar.type(item) == "text"
            ]
            self.assertTrue(any("SUPPRESSED 45° ±12°" in t for t in radar_texts))

    def test_duplicate_audio_blocks_and_below_threshold_frames_skip_redundant_doa(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            app = self._make_app(Path(tmp), simulate=False)
            app.is_active = True

            # Silence block (below threshold): should not run expensive broadband DOA
            silent_samples = np.zeros((8820, 16), dtype=np.float32)
            block_seq1 = AudioBlock(samples=silent_samples, captured_at=100.0, sequence=1)
            app.audio_mgr.get_latest_blocks = MagicMock(
                return_value={"front": block_seq1, "left": block_seq1}
            )
            with patch.object(app.doa_engines["front"], "estimate") as est_front, patch.object(
                app.doa_engines["left"], "estimate"
            ) as est_left:
                app._process_frame()
                est_front.assert_not_called()
                est_left.assert_not_called()

            # Active tone block (above threshold): runs DOA once for sequence 2, then skips duplicate sequence 2
            app.var_avg.set("1")
            t = np.arange(8820) / 44100.0
            tone = (0.5 * np.sin(2.0 * np.pi * 1000.0 * t)).astype(np.float32)
            active_samples = np.tile(tone[:, None], (1, 16))
            now = 200.0
            block_seq2 = AudioBlock(samples=active_samples, captured_at=now, sequence=2)
            app.audio_mgr.get_latest_blocks = MagicMock(
                return_value={"front": block_seq2, "left": block_seq2}
            )
            with patch("frequency_id.gui.app.time.monotonic", return_value=now), patch.object(
                app.doa_engines["front"], "estimate", wraps=app.doa_engines["front"].estimate
            ) as est_front, patch.object(
                app.doa_engines["left"], "estimate", wraps=app.doa_engines["left"].estimate
            ) as est_left:
                app._process_frame()
                self.assertEqual(est_front.call_count, 1)
                self.assertEqual(est_left.call_count, 1)

                # Calling _process_frame again with the exact same sequence number skips re-running DOA
                app._process_frame()
                self.assertEqual(est_front.call_count, 1)
                self.assertEqual(est_left.call_count, 1)

                # Live threshold change to +10 dBFS (above signal) re-evaluates immediately even on same sequence
                app.var_thresh.set("10.0")
                app._process_frame()
                self.assertFalse(app._last_detection.detected)

    def test_vision_frame_caching_skips_identical_frame_reconversion(self):
        import tempfile
        from PIL import ImageTk

        with tempfile.TemporaryDirectory() as tmp:
            app = self._make_app(Path(tmp), simulate=False)
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            vis_res = VisionResult(frame=frame, timestamp=12.34)
            fake_vision = MagicMock()
            fake_vision.get_result.return_value = vis_res
            app.vision = fake_vision
            app.var_vision_enabled.set(True)

            with patch("frequency_id.gui.app.ImageTk.PhotoImage", wraps=ImageTk.PhotoImage) as photo_mock:
                app._update_vision()
                first_photo = app.camera_photo
                self.assertIsNotNone(first_photo)
                self.assertEqual(photo_mock.call_count, 1)
                app._update_vision()
                self.assertIs(app.camera_photo, first_photo)
                self.assertEqual(photo_mock.call_count, 1)


if __name__ == "__main__":
    unittest.main()
