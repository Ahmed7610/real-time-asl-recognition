import numpy as np


TARGET_FRAMES = 40
RAW_FEATURES = 63
OUTPUT_FEATURES = 135

PINKY_TIP = 20


def get_landmark(sequence, landmark_index):
    start = landmark_index * 3
    return sequence[:, start:start + 3]


def trim_j_motion(sequence, padding=3):
    sequence = np.asarray(
        sequence,
        dtype=np.float32,
    )

    if len(sequence) < 10:
        return sequence

    pinky_tip = get_landmark(
        sequence,
        PINKY_TIP,
    )

    motion = np.linalg.norm(
        np.diff(pinky_tip, axis=0),
        axis=1,
    )

    if len(motion) == 0:
        return sequence

    kernel = np.ones(
        3,
        dtype=np.float32,
    ) / 3.0

    smoothed_motion = np.convolve(
        motion,
        kernel,
        mode="same",
    )

    high_motion = float(
        np.percentile(
            smoothed_motion,
            75,
        )
    )

    threshold = max(
        0.004,
        high_motion * 0.30,
    )

    active = np.where(
        smoothed_motion >= threshold
    )[0]

    if len(active) < 3:
        return sequence

    start = max(
        0,
        int(active[0]) - padding,
    )

    end = min(
        len(sequence),
        int(active[-1]) + 2 + padding,
    )

    trimmed = sequence[start:end]

    if len(trimmed) < 10:
        return sequence

    return trimmed


def resample_sequence(
    sequence,
    target_frames=TARGET_FRAMES,
):
    sequence = np.asarray(
        sequence,
        dtype=np.float32,
    )

    old_length = len(sequence)

    if old_length < 2:
        raise ValueError(
            "Sequence needs at least 2 frames."
        )

    old_positions = np.linspace(
        0.0,
        1.0,
        old_length,
    )

    new_positions = np.linspace(
        0.0,
        1.0,
        target_frames,
    )

    result = np.empty(
        (
            target_frames,
            sequence.shape[1],
        ),
        dtype=np.float32,
    )

    for feature_index in range(
        sequence.shape[1]
    ):
        result[:, feature_index] = np.interp(
            new_positions,
            old_positions,
            sequence[:, feature_index],
        )

    return result


def add_j_motion_features(sequence):
    sequence = np.asarray(
        sequence,
        dtype=np.float32,
    )

    if (
        sequence.ndim != 2
        or sequence.shape[1] != RAW_FEATURES
    ):
        raise ValueError(
            "Expected shape (frames, 63), "
            f"received {sequence.shape}."
        )

    all_velocity = np.diff(
        sequence,
        axis=0,
        prepend=sequence[:1],
    )

    pinky_position = get_landmark(
        sequence,
        PINKY_TIP,
    )

    pinky_velocity = np.diff(
        pinky_position,
        axis=0,
        prepend=pinky_position[:1],
    )

    pinky_acceleration = np.diff(
        pinky_velocity,
        axis=0,
        prepend=pinky_velocity[:1],
    )

    features = np.concatenate(
        [
            sequence,
            all_velocity,
            pinky_position,
            pinky_velocity,
            pinky_acceleration,
        ],
        axis=1,
    ).astype(np.float32)

    if features.shape[1] != OUTPUT_FEATURES:
        raise ValueError(
            "Unexpected feature count: "
            f"{features.shape[1]}"
        )

    return features


def preprocess_j_sequence(sequence):
    sequence = trim_j_motion(sequence)

    sequence = resample_sequence(
        sequence,
        TARGET_FRAMES,
    )

    return add_j_motion_features(sequence)
