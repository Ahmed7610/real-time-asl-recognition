"""Reusable landmark extraction and normalization using MediaPipe Tasks."""

import os
import urllib.request

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision


HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/"
    "hand_landmarker.task"
)

HAND_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "hand_landmarker.task",
)

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


def ensure_hand_model(
    path=HAND_MODEL_PATH,
    url=HAND_MODEL_URL,
):
    if not os.path.exists(path):
        urllib.request.urlretrieve(url, path)

    return path


def create_hand_detector(
    static_image_mode=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
):
    ensure_hand_model()

    base_options = mp_tasks.BaseOptions(
        model_asset_path=HAND_MODEL_PATH
    )

    if static_image_mode:
        running_mode = mp_vision.RunningMode.IMAGE
    else:
        running_mode = mp_vision.RunningMode.VIDEO

    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=running_mode,
        num_hands=1,
        min_hand_detection_confidence=min_detection_confidence,
        min_hand_presence_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    return mp_vision.HandLandmarker.create_from_options(
        options
    )


def extract_raw_landmarks(
    image_bgr,
    detector,
    timestamp_ms=None,
):
    """Return (21, 3) landmarks or None."""

    rgb = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2RGB,
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=np.ascontiguousarray(rgb),
    )

    if timestamp_ms is None:
        result = detector.detect(mp_image)
    else:
        result = detector.detect_for_video(
            mp_image,
            int(timestamp_ms),
        )

    if not result.hand_landmarks:
        return None

    landmarks = result.hand_landmarks[0]

    return np.array(
        [
            [landmark.x, landmark.y, landmark.z]
            for landmark in landmarks
        ],
        dtype=np.float32,
    )


def normalize_landmarks(coords):
    """Wrist-relative and scale-invariant feature vector."""

    coords = coords.astype(
        np.float32
    ).copy()

    coords -= coords[0]

    scale = np.max(
        np.linalg.norm(coords, axis=1)
    )

    if scale > 1e-6:
        coords /= scale

    return coords.flatten()


def image_to_static_feature(
    image_bgr,
    detector,
):
    coords = extract_raw_landmarks(
        image_bgr,
        detector,
    )

    if coords is None:
        return None

    return normalize_landmarks(coords)
