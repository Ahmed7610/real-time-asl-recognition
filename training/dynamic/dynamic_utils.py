import numpy as np


TARGET_FRAMES = 40
RAW_FEATURES = 63
DYNAMIC_FEATURES = 144

INDEX_TIP = 8
PINKY_TIP = 20


def _landmark(sequence, landmark_index):
    start = landmark_index * 3
    return sequence[:, start:start + 3]


def trim_motion(sequence, padding=3):
    """
    Removes mostly-static frames from the beginning and end.

    Motion is calculated using both:
    - index fingertip for Z
    - pinky fingertip for J
    """
    sequence = np.asarray(sequence, dtype=np.float32)

    if len(sequence) < 10:
        return sequence

    index_tip = _landmark(sequence, INDEX_TIP)
    pinky_tip = _landmark(sequence, PINKY_TIP)

    index_motion = np.linalg.norm(
        np.diff(index_tip, axis=0),
        axis=1,
    )

    pinky_motion = np.linalg.norm(
        np.diff(pinky_tip, axis=0),
        axis=1,
    )

    # Keep movement made by either dynamic fingertip.
    frame_motion = np.maximum(
        index_motion,
        pinky_motion,
    )

    if len(frame_motion) == 0:
        return sequence

    kernel = np.ones(3, dtype=np.float32) / 3.0

    smoothed_motion = np.convolve(
        frame_motion,
        kernel,
        mode="same",
    )

    high_motion = float(
        np.percentile(smoothed_motion, 75)
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
            "Sequence must contain at least 2 frames."
        )

    if old_length == target_frames:
        return sequence

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

    output = np.empty(
        (
            target_frames,
            sequence.shape[1],
        ),
        dtype=np.float32,
    )

    for feature_index in range(sequence.shape[1]):
        output[:, feature_index] = np.interp(
            new_positions,
            old_positions,
            sequence[:, feature_index],
        )

    return output


def add_motion_features(sequence):
    """
    Output:
        63 raw landmarks
        63 landmark velocities
         3 index position
         3 index velocity
         3 index acceleration
         3 pinky position
         3 pinky velocity
         3 pinky acceleration
        --------------------------------
        144 features
    """
    sequence = np.asarray(
        sequence,
        dtype=np.float32,
    )

    if (
        sequence.ndim != 2
        or sequence.shape[1] != RAW_FEATURES
    ):
        raise ValueError(
            "Expected sequence shape (frames, 63), "
            f"received {sequence.shape}."
        )

    velocity = np.diff(
        sequence,
        axis=0,
        prepend=sequence[:1],
    )

    index_tip = _landmark(
        sequence,
        INDEX_TIP,
    )

    index_velocity = np.diff(
        index_tip,
        axis=0,
        prepend=index_tip[:1],
    )

    index_acceleration = np.diff(
        index_velocity,
        axis=0,
        prepend=index_velocity[:1],
    )

    pinky_tip = _landmark(
        sequence,
        PINKY_TIP,
    )

    pinky_velocity = np.diff(
        pinky_tip,
        axis=0,
        prepend=pinky_tip[:1],
    )

    pinky_acceleration = np.diff(
        pinky_velocity,
        axis=0,
        prepend=pinky_velocity[:1],
    )

    features = np.concatenate(
        [
            sequence,
            velocity,
            index_tip,
            index_velocity,
            index_acceleration,
            pinky_tip,
            pinky_velocity,
            pinky_acceleration,
        ],
        axis=1,
    ).astype(np.float32)

    if features.shape[1] != DYNAMIC_FEATURES:
        raise ValueError(
            "Unexpected feature size: "
            f"{features.shape[1]}"
        )

    return features


def preprocess_sequence(
    sequence,
    target_frames=TARGET_FRAMES,
):
    sequence = trim_motion(sequence)

    sequence = resample_sequence(
        sequence,
        target_frames,
    )

    return add_motion_features(sequence)


def fingertip_path_length(
    sequence,
    landmark_index,
):
    sequence = np.asarray(
        sequence,
        dtype=np.float32,
    )

    tip = _landmark(
        sequence,
        landmark_index,
    )

    steps = np.diff(
        tip,
        axis=0,
    )

    return float(
        np.linalg.norm(
            steps,
            axis=1,
        ).sum()
    )
