from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
)


PROJECT_DIR = Path(__file__).resolve().parents[2]

TEST_DIR = PROJECT_DIR / "data" / "dynamic_test"

MODEL_PATH = (
    PROJECT_DIR
    / "Real-Time Sign Language Recognition"
    / "DynamicZModel_trimmed.keras"
)

TARGET_FRAMES = 40

LABEL_TO_INDEX = {
    "NONE": 0,
    "Z": 1,
}

INDEX_TO_LABEL = {
    0: "NONE",
    1: "Z",
}


def infer_label(file: Path, data):
    if "label" in data.files:
        raw = data["label"]

        if getattr(raw, "shape", None) == ():
            return str(raw.item()).upper()

        return str(raw).upper()

    name = file.name.upper()

    if "NONE" in name:
        return "NONE"

    return "Z"

def trim_motion(sequence, padding=3):
    sequence = np.asarray(
        sequence,
        dtype=np.float32,
    )

    if len(sequence) < 10:
        return sequence

    index_tip = sequence[
        :,
        8 * 3 : 8 * 3 + 3,
    ]

    frame_motion = np.linalg.norm(
        np.diff(index_tip, axis=0),
        axis=1,
    )

    if len(frame_motion) == 0:
        return sequence

    kernel = np.ones(
        3,
        dtype=np.float32,
    ) / 3.0

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


def resample_sequence(sequence, target_frames):
    sequence = np.asarray(
        sequence,
        dtype=np.float32,
    )

    old_length = len(sequence)

    if old_length == target_frames:
        return sequence

    if old_length < 2:
        raise ValueError(
            "Sequence must contain at least 2 frames."
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

    output = np.empty(
        (
            target_frames,
            sequence.shape[1],
        ),
        dtype=np.float32,
    )

    for feature_index in range(
        sequence.shape[1]
    ):
        output[:, feature_index] = np.interp(
            new_positions,
            old_positions,
            sequence[:, feature_index],
        )

    return output


def add_motion_features(sequence):
    base = sequence

    velocity = np.diff(
        sequence,
        axis=0,
        prepend=sequence[:1],
    )

    index_tip = sequence[
        :,
        8 * 3 : 8 * 3 + 3,
    ]

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

    return np.concatenate(
        [
            base,
            velocity,
            index_tip,
            index_velocity,
            index_acceleration,
        ],
        axis=1,
    ).astype(np.float32)

def load_test_data():
    files = sorted(
        TEST_DIR.rglob("*.npz")
    )

    if not files:
        raise FileNotFoundError(
            f"No test files found in {TEST_DIR}"
        )

    X = []
    y = []
    filenames = []

    for file in files:
        data = np.load(file)
        sequence = data["landmarks"]

        if (
            sequence.ndim != 2
            or sequence.shape[1] != 63
            or len(sequence) < 10
            or not np.isfinite(sequence).all()
        ):
            print(f"Skipping invalid file: {file}")
            continue

        label = infer_label(
            file,
            data,
        )

        if label not in LABEL_TO_INDEX:
            print(
                f"Skipping unknown label {label}: {file}"
            )
            continue
        
        sequence= trim_motion(
            sequence
        )
        sequence = resample_sequence(
            sequence,
            TARGET_FRAMES,
        )

        sequence = add_motion_features(
            sequence
        )

        X.append(sequence)
        y.append(LABEL_TO_INDEX[label])
        filenames.append(file.name)

    return (
        np.stack(X).astype(np.float32),
        np.asarray(y, dtype=np.int64),
        filenames,
    )


def evaluate_threshold(
    y_true,
    probabilities,
    threshold,
):
    predictions = (
        probabilities >= threshold
    ).astype(np.int64)

    accuracy = accuracy_score(
        y_true,
        predictions,
    )

    matrix = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    )

    tn, fp, fn, tp = matrix.ravel()

    return {
        "threshold": threshold,
        "accuracy": accuracy,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def main():
    X, y, filenames = load_test_data()

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    probabilities = model.predict(
        X,
        verbose=0,
    ).reshape(-1)

    print()
    print("=" * 72)
    print("DYNAMIC Z MODEL TEST")
    print("=" * 72)
    print(f"Samples: {len(y)}")
    print(f"NONE: {int(np.sum(y == 0))}")
    print(f"Z:    {int(np.sum(y == 1))}")
    print()

    thresholds = [
        0.50,
        0.60,
        0.70,
        0.80,
        0.90,
    ]

    results = []

    for threshold in thresholds:
        result = evaluate_threshold(
            y,
            probabilities,
            threshold,
        )

        results.append(result)

        print(
            f"Threshold {threshold:.2f} | "
            f"accuracy={result['accuracy'] * 100:.2f}% | "
            f"FP={result['fp']} | "
            f"FN={result['fn']} | "
            f"TP={result['tp']} | "
            f"TN={result['tn']}"
        )

    best = max(
        results,
        key=lambda item: (
            item["accuracy"],
            -item["fp"],
        ),
    )

    best_threshold = best["threshold"]

    predictions = (
        probabilities >= best_threshold
    ).astype(np.int64)

    print()
    print("=" * 72)
    print(
        f"BEST THRESHOLD: {best_threshold:.2f}"
    )
    print("=" * 72)

    print(
        classification_report(
            y,
            predictions,
            target_names=["NONE", "Z"],
            digits=4,
        )
    )

    print("Confusion matrix:")
    print(
        confusion_matrix(
            y,
            predictions,
            labels=[0, 1],
        )
    )

    print()
    print("=" * 72)
    print("MISCLASSIFIED FILES")
    print("=" * 72)

    mistakes = 0

    for filename, actual, predicted, probability in zip(
        filenames,
        y,
        predictions,
        probabilities,
    ):
        if actual != predicted:
            mistakes += 1

            print(
                f"{filename}\n"
                f"  actual={INDEX_TO_LABEL[actual]}, "
                f"predicted={INDEX_TO_LABEL[predicted]}, "
                f"Z_probability={probability:.4f}"
            )

    if mistakes == 0:
        print("No mistakes.")

    print()
    print("=" * 72)
    print("PROBABILITY SUMMARY")
    print("=" * 72)

    for class_index, class_name in INDEX_TO_LABEL.items():
        class_probabilities = probabilities[
            y == class_index
        ]

        print(
            f"{class_name}: "
            f"min={class_probabilities.min():.4f}, "
            f"max={class_probabilities.max():.4f}, "
            f"mean={class_probabilities.mean():.4f}"
        )


if __name__ == "__main__":
    main()
