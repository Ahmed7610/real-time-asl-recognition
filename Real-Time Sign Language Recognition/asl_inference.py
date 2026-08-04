"""Inference engine for the live web application.

Default mode:
    24 static ASL letters.

Manual dynamic mode:
    The frontend sends dynamic_start, frames are collected, and then
    dynamic_stop triggers J/Z classification.
"""

from __future__ import annotations

import json
import os
import time
from collections import Counter, deque

import joblib
import numpy as np
from tensorflow.keras.models import load_model

from asl_landmarks import create_hand_detector, normalize_landmarks


TARGET_FRAMES = 40

J_THRESHOLD = 0.50
Z_THRESHOLD = 0.56

MIN_J_POSE_RATIO = 0.55
MIN_Z_POSE_RATIO = 0.55

MIN_PINKY_PATH_LENGTH = 0.50
MIN_INDEX_PATH_LENGTH = 0.18

MAX_MANUAL_FRAMES = 140


def resample_sequence(
    sequence: np.ndarray,
    target_frames: int = TARGET_FRAMES,
) -> np.ndarray:
    sequence = np.asarray(sequence, dtype=np.float32)

    if sequence.ndim != 2 or sequence.shape[1] != 63:
        raise ValueError(
            f"Expected dynamic sequence with shape (frames, 63), got {sequence.shape}."
        )

    if len(sequence) < 2:
        raise ValueError("Dynamic sequence needs at least 2 frames.")

    if len(sequence) == target_frames:
        return sequence

    old_positions = np.linspace(0.0, 1.0, len(sequence))
    new_positions = np.linspace(0.0, 1.0, target_frames)

    output = np.empty(
        (target_frames, sequence.shape[1]),
        dtype=np.float32,
    )

    for feature_index in range(sequence.shape[1]):
        output[:, feature_index] = np.interp(
            new_positions,
            old_positions,
            sequence[:, feature_index],
        )

    return output


def trim_motion(
    sequence: np.ndarray,
    landmark_index: int,
    padding: int = 3,
) -> np.ndarray:
    sequence = np.asarray(sequence, dtype=np.float32)

    if len(sequence) < 10:
        return sequence

    start = landmark_index * 3
    fingertip = sequence[:, start:start + 3]

    motion = np.linalg.norm(
        np.diff(fingertip, axis=0),
        axis=1,
    )

    if len(motion) == 0:
        return sequence

    smoothing_kernel = np.ones(3, dtype=np.float32) / 3.0
    smoothed_motion = np.convolve(
        motion,
        smoothing_kernel,
        mode="same",
    )

    threshold = max(
        0.004,
        float(np.percentile(smoothed_motion, 75)) * 0.30,
    )

    active_indices = np.where(
        smoothed_motion >= threshold,
    )[0]

    if len(active_indices) < 3:
        return sequence

    first = max(
        0,
        int(active_indices[0]) - padding,
    )
    last = min(
        len(sequence),
        int(active_indices[-1]) + 2 + padding,
    )

    trimmed = sequence[first:last]

    if len(trimmed) < 10:
        return sequence

    return trimmed


def add_motion_features(
    sequence: np.ndarray,
    focus_index: int,
) -> np.ndarray:
    sequence = np.asarray(sequence, dtype=np.float32)

    all_velocity = np.diff(
        sequence,
        axis=0,
        prepend=sequence[:1],
    )

    focus_start = focus_index * 3
    focus_position = sequence[
        :,
        focus_start:focus_start + 3,
    ]

    focus_velocity = np.diff(
        focus_position,
        axis=0,
        prepend=focus_position[:1],
    )

    focus_acceleration = np.diff(
        focus_velocity,
        axis=0,
        prepend=focus_velocity[:1],
    )

    features = np.concatenate(
        [
            sequence,
            all_velocity,
            focus_position,
            focus_velocity,
            focus_acceleration,
        ],
        axis=1,
    )

    return features.astype(np.float32)


def path_length(
    sequence: np.ndarray,
    landmark_index: int,
) -> float:
    sequence = np.asarray(sequence, dtype=np.float32)

    start = landmark_index * 3
    fingertip = sequence[:, start:start + 3]

    differences = np.diff(
        fingertip,
        axis=0,
    )

    return float(
        np.linalg.norm(
            differences,
            axis=1,
        ).sum()
    )


def finger_is_extended(
    landmarks: np.ndarray,
    tip_index: int,
    pip_index: int,
    margin: float,
) -> bool:
    points = np.asarray(
        landmarks,
        dtype=np.float32,
    )

    wrist = points[0]

    tip_distance = float(
        np.linalg.norm(
            points[tip_index] - wrist,
        )
    )

    pip_distance = float(
        np.linalg.norm(
            points[pip_index] - wrist,
        )
    )

    return tip_distance > pip_distance + margin


def is_z_pose(landmarks: np.ndarray) -> bool:
    return (
        finger_is_extended(landmarks, 8, 6, 0.025)
        and not finger_is_extended(landmarks, 12, 10, 0.015)
        and not finger_is_extended(landmarks, 16, 14, 0.015)
        and not finger_is_extended(landmarks, 20, 18, 0.015)
    )


def is_j_pose(landmarks: np.ndarray) -> bool:
    return (
        finger_is_extended(landmarks, 20, 18, 0.020)
        and not finger_is_extended(landmarks, 8, 6, 0.015)
        and not finger_is_extended(landmarks, 12, 10, 0.015)
        and not finger_is_extended(landmarks, 16, 14, 0.015)
    )


class SignLanguageTranslator:
    """Loads the models and stores runtime state for one session."""

    def __init__(
        self,
        models_dir: str = "models",
        config: dict | None = None,
    ):
        if config is None:
            config_path = os.path.join(
                models_dir,
                "config.json",
            )

            with open(
                config_path,
                encoding="utf-8",
            ) as config_file:
                config = json.load(config_file)

        self.cfg = config
        self.models_dir = models_dir

        self.static_model = load_model(
            os.path.join(
                models_dir,
                "StaticModel.keras",
            )
        )

        self.static_scaler = joblib.load(
            os.path.join(
                models_dir,
                "StaticScaler.pkl",
            )
        )

        self.static_encoder = joblib.load(
            os.path.join(
                models_dir,
                "StaticLabelEncoder.pkl",
            )
        )

        self.j_model = load_model(
            os.path.join(
                models_dir,
                "DynamicJModel_clean.keras",
            )
        )

        self.z_model = load_model(
            os.path.join(
                models_dir,
                "DynamicZModel_trimmed.keras",
            )
        )

        self.min_confidence = float(
            config.get(
                "min_confidence",
                0.55,
            )
        )

        # Keep the stable static behavior.
        self.detector = create_hand_detector(
            static_image_mode=True,
        )

        self.reset()

    def reset(self) -> None:
        self.smooth = deque(
            maxlen=int(
                self.cfg["smoothing_window"]
            )
        )

        self.state = "static"
        self.manual_dynamic = False

        self.dynamic_sequence: list[np.ndarray] = []
        self.j_pose_history: list[bool] = []
        self.z_pose_history: list[bool] = []

        self.last_dynamic_debug = self.empty_dynamic_debug()

    @staticmethod
    def empty_dynamic_debug() -> dict:
        return {
            "j_probability": 0.0,
            "z_probability": 0.0,
            "pinky_path": 0.0,
            "index_path": 0.0,
            "j_pose_ratio": 0.0,
            "z_pose_ratio": 0.0,
            "frames_recorded": 0,
        }

    def static_predict(
        self,
        coords: np.ndarray,
    ) -> tuple[str, float]:
        normalized = normalize_landmarks(
            coords,
        ).reshape(1, -1)

        scaled = self.static_scaler.transform(
            normalized,
        )

        probabilities = self.static_model(
            scaled,
            training=False,
        ).numpy()[0]

        predicted_index = int(
            probabilities.argmax()
        )

        predicted_letter = str(
            self.static_encoder.classes_[
                predicted_index
            ]
        )

        confidence = float(
            probabilities[predicted_index]
        )

        return predicted_letter, confidence

    def static_result(
        self,
        coords: np.ndarray | None,
    ) -> dict:
        if coords is None:
            self.smooth.append("nothing")

            return self.finalize(
                letter="nothing",
                raw_letter="nothing",
                confidence=1.0,
                source="none",
            )

        raw_letter, confidence = self.static_predict(
            coords,
        )

        accepted_letter = (
            raw_letter
            if confidence >= self.min_confidence
            else "nothing"
        )

        self.smooth.append(
            accepted_letter,
        )

        smoothed_letter = Counter(
            self.smooth,
        ).most_common(1)[0][0]

        stability = (
            sum(
                item == smoothed_letter
                for item in self.smooth
            )
            / len(self.smooth)
        )

        return self.finalize(
            letter=smoothed_letter,
            raw_letter=raw_letter,
            confidence=confidence,
            source="static",
            stability=stability,
        )

    def finalize(
        self,
        letter: str,
        raw_letter: str,
        confidence: float,
        source: str,
        stability: float = 0.0,
    ) -> dict:
        return {
            "letter": letter,
            "raw_letter": raw_letter,
            "confidence": float(confidence),
            "stability": float(stability),
            "source": source,
            "motion_state": self.state,
            "dynamic": dict(
                self.last_dynamic_debug,
            ),
        }

    def start_dynamic_recording(self) -> dict:
        self.manual_dynamic = True
        self.state = "recording"

        self.dynamic_sequence = []
        self.j_pose_history = []
        self.z_pose_history = []

        self.smooth.clear()
        self.last_dynamic_debug = (
            self.empty_dynamic_debug()
        )

        print(
            "[dynamic] manual recording started"
        )

        return self.finalize(
            letter="nothing",
            raw_letter="nothing",
            confidence=0.0,
            source="dynamic",
        )

    def finish_dynamic_recording(self) -> dict:
        frames_recorded = len(
            self.dynamic_sequence,
        )

        if (
            self.state != "recording"
            or not self.manual_dynamic
        ):
            return self.finalize(
                letter="nothing",
                raw_letter="nothing",
                confidence=0.0,
                source="dynamic",
            )

        if frames_recorded < 10:
            self.last_dynamic_debug[
                "frames_recorded"
            ] = frames_recorded

            self.return_to_static()

            return self.finalize(
                letter="nothing",
                raw_letter="nothing",
                confidence=0.0,
                source="dynamic",
            )

        self.state = "processing"

        try:
            letter, confidence = (
                self.classify_dynamic()
            )
        except Exception as error:
            print(
                f"[dynamic] processing error: {error}"
            )

            letter = "nothing"
            confidence = 0.0

        debug_result = dict(
            self.last_dynamic_debug,
        )

        self.return_to_static()

        # Preserve the debug values after resetting recording state.
        self.last_dynamic_debug = debug_result

        result = self.finalize(
            letter=letter,
            raw_letter=letter,
            confidence=confidence,
            source="dynamic",
        )

        print(
            "[dynamic] manual recording stopped | "
            f"result={letter} | "
            f"debug={debug_result}"
        )

        return result

    def return_to_static(self) -> None:
        self.state = "static"
        self.manual_dynamic = False

        self.dynamic_sequence = []
        self.j_pose_history = []
        self.z_pose_history = []

        self.smooth.clear()

    def classify_dynamic(
        self,
    ) -> tuple[str, float]:
        raw_sequence = np.asarray(
            self.dynamic_sequence,
            dtype=np.float32,
        )

        # Z uses the index fingertip.
        z_trimmed = trim_motion(
            raw_sequence,
            landmark_index=8,
        )

        z_resampled = resample_sequence(
            z_trimmed,
        )

        z_features = add_motion_features(
            z_resampled,
            focus_index=8,
        )

        z_probability = float(
            self.z_model.predict(
                z_features[None, ...],
                verbose=0,
            ).reshape(-1)[0]
        )

        index_path = path_length(
            z_resampled,
            landmark_index=8,
        )

        # J uses the pinky fingertip.
        j_trimmed = trim_motion(
            raw_sequence,
            landmark_index=20,
        )

        j_resampled = resample_sequence(
            j_trimmed,
        )

        j_features = add_motion_features(
            j_resampled,
            focus_index=20,
        )

        j_probability = float(
            self.j_model.predict(
                j_features[None, ...],
                verbose=0,
            ).reshape(-1)[0]
        )

        pinky_path = path_length(
            j_trimmed,
            landmark_index=20,
        )

        j_pose_ratio = (
            float(np.mean(self.j_pose_history))
            if self.j_pose_history
            else 0.0
        )

        z_pose_ratio = (
            float(np.mean(self.z_pose_history))
            if self.z_pose_history
            else 0.0
        )

        self.last_dynamic_debug = {
            "j_probability": j_probability,
            "z_probability": z_probability,
            "pinky_path": pinky_path,
            "index_path": index_path,
            "j_pose_ratio": j_pose_ratio,
            "z_pose_ratio": z_pose_ratio,
            "frames_recorded": int(
                len(raw_sequence)
            ),
        }

        j_accepted = (
            j_probability >= J_THRESHOLD
            and pinky_path >= MIN_PINKY_PATH_LENGTH
            and j_pose_ratio >= MIN_J_POSE_RATIO
        )

        z_accepted = (
            z_probability >= Z_THRESHOLD
            and index_path >= MIN_INDEX_PATH_LENGTH
            and z_pose_ratio >= MIN_Z_POSE_RATIO
        )

        if j_accepted and z_accepted:
            normalized_j_score = (
                j_probability / J_THRESHOLD
            )
            normalized_z_score = (
                z_probability / Z_THRESHOLD
            )

            letter = (
                "J"
                if normalized_j_score
                >= normalized_z_score
                else "Z"
            )
        elif j_accepted:
            letter = "J"
        elif z_accepted:
            letter = "Z"
        else:
            letter = "nothing"

        if letter == "J":
            confidence = j_probability
        elif letter == "Z":
            confidence = z_probability
        else:
            confidence = max(
                j_probability,
                z_probability,
            )

        return letter, confidence

    def process_landmarks(
        self,
        coords: np.ndarray | None,
    ) -> dict:
        # Normal static mode.
        if self.state != "recording":
            return self.static_result(
                coords,
            )

        # Manual dynamic mode.
        if coords is None:
            self.last_dynamic_debug[
                "frames_recorded"
            ] = len(
                self.dynamic_sequence,
            )

            return self.finalize(
                letter="nothing",
                raw_letter="nothing",
                confidence=0.0,
                source="dynamic",
            )

        normalized = normalize_landmarks(
            coords,
        ).astype(np.float32)

        self.dynamic_sequence.append(
            normalized,
        )

        self.j_pose_history.append(
            is_j_pose(coords),
        )

        self.z_pose_history.append(
            is_z_pose(coords),
        )

        self.last_dynamic_debug[
            "frames_recorded"
        ] = len(
            self.dynamic_sequence,
        )

        if (
            len(self.dynamic_sequence)
            >= MAX_MANUAL_FRAMES
        ):
            return self.finish_dynamic_recording()

        return self.finalize(
            letter="nothing",
            raw_letter="nothing",
            confidence=0.0,
            source="dynamic",
        )

    def process_frame(
        self,
        frame_bgr,
    ) -> dict:
        raise RuntimeError(
            "The FastAPI backend should extract landmarks "
            "and call process_landmarks()."
        )


class WordBuilder:
    """Build text using a time-based static hold and accepted dynamic letters."""

    def __init__(
        self,
        stability_frames: int = 12,
        hold_seconds: float = 0.75,
        max_gap_seconds: float = 0.55,
    ):
        # Kept for backwards compatibility and API/config reporting.
        self.stability = int(stability_frames)

        self.hold_seconds = max(
            0.10,
            float(hold_seconds),
        )

        self.max_gap_seconds = max(
            0.10,
            float(max_gap_seconds),
        )

        self.reset()

    def reset(self) -> None:
        self.text = ""
        self._candidate = None
        self._candidate_since = 0.0
        self._last_seen_at = 0.0
        self._locked = False

    def hand_missing(self) -> None:
        self._candidate = None
        self._candidate_since = 0.0
        self._last_seen_at = 0.0
        self._locked = False

    def update_static(
        self,
        token: str,
        now: float | None = None,
    ) -> str:
        if not token or token == "nothing":
            return self.text

        current_time = (
            time.monotonic()
            if now is None
            else float(now)
        )

        gap_too_long = (
            self._last_seen_at > 0.0
            and current_time - self._last_seen_at
            > self.max_gap_seconds
        )

        if (
            token != self._candidate
            or gap_too_long
        ):
            self._candidate = token
            self._candidate_since = current_time

        self._last_seen_at = current_time

        if self._locked:
            return self.text

        held_for = (
            current_time
            - self._candidate_since
        )

        if held_for < self.hold_seconds:
            return self.text

        self.text += token
        self._locked = True

        return self.text

    # Compatibility with the original static backend.
    def update(
        self,
        token: str,
    ) -> str:
        return self.update_static(
            token,
        )

    def commit_dynamic(
        self,
        token: str,
    ) -> str:
        if token not in {"J", "Z"}:
            return self.text

        self.text += token

        self._candidate = token
        self._candidate_since = 0.0
        self._last_seen_at = 0.0
        self._locked = True

        return self.text
