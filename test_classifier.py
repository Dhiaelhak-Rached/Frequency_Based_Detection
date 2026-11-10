import time
import unittest
from pathlib import Path

import numpy as np
import scipy.signal as signal

from audio_manager import MultiDeviceAudioManager
from classifier import DroneClassifier
from detector import FrequencyDetector
from doa_engine import ArrayDOA
from geometry import RigGeometry
from triangulation import Triangulator


class FakeSession:
    def __init__(self, probabilities):
        self.probabilities = iter(probabilities)

    def run(self, output_names, inputs):
        return [np.asarray([next(self.probabilities)], dtype=np.float32)]


class ClassifierTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.geo = RigGeometry()
        cls.classifier = DroneClassifier(sample_rate=cls.geo.sample_rate)
        cls.reference_path = Path(__file__).resolve().parent / "model" / "reference_outputs.npz"

    def test_onnx_numerical_accuracy(self):
        passed, max_diff = self.classifier.validate_against_reference(
            self.reference_path,
            tolerance=1e-3,
        )
        print(f"ONNX maximum absolute probability error: {max_diff:.9f}", flush=True)
        self.assertTrue(passed, f"ONNX reference error {max_diff:.9f} exceeds 1e-3")

    def test_resampling_fidelity(self):
        sample_rate = 44100
        target_rate = 16000
        frequency = 1000.0
        t = np.arange(sample_rate, dtype=np.float64) / sample_rate
        waveform = (0.5 * np.sin(2.0 * np.pi * frequency * t)).astype(np.float32)

        start = time.perf_counter()
        resampled = signal.resample_poly(waveform, 160, 441)
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        spectrum = np.abs(np.fft.rfft(resampled * np.hanning(len(resampled))))
        freqs = np.fft.rfftfreq(len(resampled), 1.0 / target_rate)
        peak_frequency = freqs[int(np.argmax(spectrum))]
        print(f"Resampling latency: {elapsed_ms:.3f} ms", flush=True)
        self.assertEqual(len(resampled), target_rate)
        self.assertAlmostEqual(peak_frequency, frequency, delta=2.0)
        self.assertTrue(np.isfinite(resampled).all())

    def test_temporal_voting(self):
        classifier = DroneClassifier(sample_rate=16000)
        classifier.session = FakeSession(
            [
                (0.2, 0.8),
                (0.2, 0.8),
                (0.8, 0.2),
                (0.2, 0.8),
                (0.8, 0.2),
                (0.8, 0.2),
            ]
        )
        classifier.add_audio(np.zeros(classifier.buffer_size, dtype=np.float32))

        results = [classifier.classify() for _ in range(6)]
        self.assertEqual([result.decision_votes for result in results], [1, 2, 2, 3, 3, 2])
        self.assertEqual(
            [result.is_drone for result in results],
            [False, False, False, True, True, False],
        )

    def test_end_to_end_classification_with_doa(self):
        np.random.seed(7)
        audio_mgr = MultiDeviceAudioManager(geometry=self.geo, simulate=True)
        audio_mgr.sim_speed_rad_per_s = 0.0
        target = np.array([-0.6, 1.5])
        audio_mgr.set_simulation_target(x=target[0], y=target[1], freq_hz=1000.0, snr_db=20.0)

        classifier = DroneClassifier(sample_rate=self.geo.sample_rate)
        detector = FrequencyDetector(
            sample_rate=self.geo.sample_rate,
            block_duration=0.2,
            average_blocks=1,
            threshold_dbfs=-65.0,
        )
        doa_engines = {
            arr_id: ArrayDOA(
                arr_id,
                arr.local_mics,
                arr.orientation_deg,
                self.geo.sample_rate,
                mount=arr.mount,
            )
            for arr_id, arr in self.geo.arrays.items()
        }
        triangulator = Triangulator(self.geo, min_prominence=1.4)

        classification = None
        detection = None
        triangulation = None
        for step in range(8):
            audio_mgr.sim_t0 = time.time() - step * audio_mgr.block_duration
            blocks = audio_mgr.get_latest_blocks()
            mono = blocks["front"][:, 0]
            classification = classifier.classify(mono)
            detection = detector.process(mono, 900.0, 1100.0)
            bearings = {}
            prominences = {}
            stable = {}
            for arr_id, doa in doa_engines.items():
                result = doa.estimate(
                    blocks[arr_id],
                    target_freq=detection.peak_freq,
                    band=(900.0, 1100.0),
                )
                bearings[arr_id] = result.global_angle_deg
                prominences[arr_id] = result.prominence
                stable[arr_id] = result.stable
            triangulation = triangulator.triangulate(bearings, prominences, stable=stable)

        self.assertTrue(detection.detected)
        self.assertTrue(classification.is_ready)
        self.assertTrue(
            classification.is_drone,
            f"Simulated drone was not confirmed: {classification}",
        )
        self.assertTrue(triangulation.success, f"Triangulation failed: {triangulation}")
        position = np.array([triangulation.x, triangulation.y])
        self.assertLess(np.linalg.norm(position - target), 0.5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
