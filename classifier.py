#!/usr/bin/env python3
"""MobileNetV2 Drone Audio Classification Engine.

Integrates a lightweight deep neural network (ONNX) to distinguish drone acoustic
signatures from background environmental sound, wind, and non-drone sources in real time.
"""

from collections import deque
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import onnxruntime as ort
import scipy.signal as signal


@dataclass
class ClassificationResult:
    """Represents a single audio classification inference and temporal decision."""
    is_drone: bool
    probability: float  # P(drone) for the current window [0.0, 1.0]
    smoothed_probability: float  # Mean P(drone) over temporal window
    decision_votes: int  # Number of positive votes in window
    vote_window: int  # Size of temporal voting window (e.g. 5)
    threshold: float  # Probability threshold used (e.g. 0.75)
    raw_probabilities: Tuple[float, float]  # (P(no_drone), P(drone))
    is_ready: bool  # True if sufficient audio has been buffered


class DroneClassifier:
    """Real-time drone acoustic classifier using MobileNetV2 ONNX."""

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        metadata_path: Optional[Union[str, Path]] = None,
        sample_rate: int = 44100,
        threshold: Optional[float] = None,
        votes_required: Optional[int] = None,
        vote_window: Optional[int] = None,
    ):
        base_dir = Path(__file__).resolve().parent

        # 1. Resolve paths
        if model_path is None:
            model_path = base_dir / "model" / "mobilenetv2_drone.onnx"
            if not model_path.exists():
                model_path = Path("/home/dhia/Desktop/projects/MobileNetV2/model/mobilenetv2_drone.onnx")
        self.model_path = Path(model_path)

        if metadata_path is None:
            metadata_path = base_dir / "model" / "metadata.json"
            if not metadata_path.exists():
                metadata_path = Path("/home/dhia/Desktop/projects/MobileNetV2/model/metadata.json")
        self.metadata_path = Path(metadata_path)

        # 2. Load metadata defaults if available
        self.metadata: Dict = {}
        if self.metadata_path.exists():
            try:
                with open(self.metadata_path, "r") as f:
                    self.metadata = json.load(f)
            except Exception as e:
                print(f"[WARNING] Failed to load metadata from {self.metadata_path}: {e}")

        # Model native sample rate and window
        self.target_sr = int(self.metadata.get("sample_rate", 16000))
        self.target_window_samples = int(self.metadata.get("window_samples", 16000))
        self.window_seconds = float(self.target_window_samples / self.target_sr)

        # Decision parameters
        decision_cfg = self.metadata.get("decision", {})
        self.threshold = float(threshold if threshold is not None else decision_cfg.get("threshold", 0.75))
        self.votes_required = int(votes_required if votes_required is not None else decision_cfg.get("votes_required", 3))
        self.vote_window = int(vote_window if vote_window is not None else decision_cfg.get("vote_window", 5))

        # 3. Initialize ONNX runtime session
        if not self.model_path.exists():
            raise FileNotFoundError(f"Classifier ONNX model not found: {self.model_path}")

        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session_options.intra_op_num_threads = 2
        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options=session_options,
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        # 4. Audio buffering and resampling setup
        self.input_sample_rate = sample_rate
        self.buffer_size = int(round(sample_rate * self.window_seconds))
        self.ring_buffer = np.zeros(self.buffer_size, dtype=np.float32)
        self.samples_buffered = 0

        # Resampling ratio (up/down) using greatest common divisor
        g = math.gcd(self.target_sr, self.input_sample_rate)
        self.resample_up = self.target_sr // g
        self.resample_down = self.input_sample_rate // g

        # 5. Temporal voting queues
        self.vote_history = deque(maxlen=self.vote_window)
        self.prob_history = deque(maxlen=self.vote_window)

    def add_audio(self, audio: np.ndarray):
        """Append incoming audio samples to the rolling buffer.

        Accepts 1D mono audio or 2D multichannel audio (first channel is used).
        """
        if audio.ndim > 1:
            audio = audio[:, 0]
        audio = np.asarray(audio, dtype=np.float32)
        n = len(audio)
        if n == 0:
            return

        if n >= self.buffer_size:
            self.ring_buffer[:] = audio[-self.buffer_size:]
            self.samples_buffered = self.buffer_size
        else:
            self.ring_buffer = np.roll(self.ring_buffer, -n)
            self.ring_buffer[-n:] = audio
            self.samples_buffered = min(self.buffer_size, self.samples_buffered + n)

    def reset(self):
        """Clear the audio buffer and voting history."""
        self.ring_buffer.fill(0.0)
        self.samples_buffered = 0
        self.vote_history.clear()
        self.prob_history.clear()

    def classify(self, audio: Optional[np.ndarray] = None) -> ClassificationResult:
        """Run classification on the buffered audio or an explicit audio segment."""
        if audio is not None:
            self.add_audio(audio)

        is_ready = self.samples_buffered >= self.buffer_size
        if not is_ready:
            return ClassificationResult(
                is_drone=False,
                probability=0.0,
                smoothed_probability=0.0,
                decision_votes=0,
                vote_window=self.vote_window,
                threshold=self.threshold,
                raw_probabilities=(1.0, 0.0),
                is_ready=False,
            )

        # Resample from input_sample_rate (e.g. 44.1 kHz) to target_sr (16 kHz)
        wav = self.ring_buffer
        if self.input_sample_rate != self.target_sr:
            resampled = signal.resample_poly(wav, self.resample_up, self.resample_down).astype(np.float32)
            if len(resampled) > self.target_window_samples:
                resampled = resampled[-self.target_window_samples:]
            elif len(resampled) < self.target_window_samples:
                resampled = np.pad(resampled, (self.target_window_samples - len(resampled), 0))
            wav_16k = resampled
        else:
            wav_16k = wav

        # Input tensor shape: [1, 16000]
        input_tensor = wav_16k[None, :].astype(np.float32)

        # Inference
        probs = self.session.run([self.output_name], {self.input_name: input_tensor})[0][0]
        p_no_drone = float(probs[0])
        p_drone = float(probs[1])

        # Voting update
        vote = bool(p_drone >= self.threshold)
        self.vote_history.append(vote)
        self.prob_history.append(p_drone)

        votes_count = sum(self.vote_history)
        is_drone = bool(votes_count >= self.votes_required and len(self.vote_history) >= min(3, self.votes_required))
        smoothed = float(np.mean(self.prob_history))

        return ClassificationResult(
            is_drone=is_drone,
            probability=p_drone,
            smoothed_probability=smoothed,
            decision_votes=votes_count,
            vote_window=self.vote_window,
            threshold=self.threshold,
            raw_probabilities=(p_no_drone, p_drone),
            is_ready=True,
        )

    def validate_against_reference(
        self,
        reference_npz_path: Union[str, Path],
        tolerance: float = 1e-3,
    ) -> Tuple[bool, float]:
        """Validate ONNX inference against PyTorch reference predictions."""
        ref_path = Path(reference_npz_path)
        if not ref_path.exists():
            raise FileNotFoundError(f"Reference NPZ not found: {ref_path}")

        data = np.load(ref_path)
        waveforms = data["waveforms"]
        ref_probs = data["pytorch_probs"]

        preds = self.session.run([self.output_name], {self.input_name: waveforms})[0]
        max_diff = float(np.max(np.abs(preds - ref_probs)))
        passed = max_diff < tolerance
        return passed, max_diff
