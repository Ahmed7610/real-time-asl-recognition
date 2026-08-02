from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from j_utils import preprocess_j_sequence


PROJECT_DIR = Path(__file__).resolve().parents[2]

TEST_DIR = (
    PROJECT_DIR
    / "data"
    / "dynamic_test"
)

MODEL_PATH = (
    PROJECT_DIR
    / "Real-Time Sign Language Recognition"
    / "DynamicJModel.keras"
)


def infer_label(file, data):
    if "label" in data.files:
        raw = data["label"]

        if getattr(raw, "shape", None) == ():
            return str(raw.item()).upper()

        return str(raw).upper()

    name = file.name.upper()

    if "_J_" in name:
        return "J"

    if "_Z_" in name:
        return "Z"

    if "NONE" in name:
        return "NONE"

    return None


def load_test_data():
    files = sorted(
        TEST_DIR.rglob("*.npz")
    )

    if not files:
        raise FileNotFoundError(
            f"No test files found in {TEST_DIR}"
        )

    X = []
    binary_targets = []
    source_labels = []
    filenames = []
    skipped = 0

    for file in files:
        try:
            with np.load(file) as data:
                sequence = data["landmarks"]
                label = infer_label(
                    file,
                    data,
                )

            if label not in {
                "J",
                "Z",
                "NONE",
            }:
                skipped += 1
                continue

            if (
                sequence.ndim != 2
                or sequence.shape[1] != 63
                or len(sequence) < 10
                or not np.isfinite(sequence).all()
            ):
                print(
                    f"Skipping invalid file: {file}"
                )
                skipped += 1
                continue

            features = preprocess_j_sequence(
                sequence
            )

            target = 1 if label == "J" else 0

            X.append(features)
            binary_targets.append(target)
            source_labels.append(label)
            filenames.append(file.name)

        except Exception as error:
            print(
                f"Skipping {file.name}: {error}"
            )
            skipped += 1

    if not X:
        raise RuntimeError(
            "No valid test samples loaded."
        )

    return (
        np.stack(X).astype(np.float32),
        np.asarray(
            binary_targets,
            dtype=np.int64,
        ),
        np.asarray(source_labels),
        filenames,
        skipped,
    )


def print_group_summary(
    group_name,
    mask,
    probabilities,
    predictions,
):
    if not np.any(mask):
        return

    group_probabilities = probabilities[mask]
    group_predictions = predictions[mask]

    print()
    print(group_name)
    print(
        f"  samples: {int(np.sum(mask))}"
    )
    print(
        f"  J probability min: "
        f"{group_probabilities.min():.4f}"
    )
    print(
        f"  J probability mean: "
        f"{group_probabilities.mean():.4f}"
    )
    print(
        f"  J probability max: "
        f"{group_probabilities.max():.4f}"
    )
    print(
        f"  predicted as J: "
        f"{int(np.sum(group_predictions == 1))}"
    )


def main():
    (
        X,
        y,
        source_labels,
        filenames,
        skipped,
    ) = load_test_data()

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    print()
    print("Model input:", model.input_shape)
    print("Test input:", X.shape)

    probabilities = (
        model.predict(
            X,
            verbose=0,
        )
        .reshape(-1)
    )

    threshold = 0.50

    predictions = (
        probabilities >= threshold
    ).astype(np.int64)

    accuracy = accuracy_score(
        y,
        predictions,
    )

    print()
    print("=" * 72)
    print("DYNAMIC J MODEL TEST")
    print("=" * 72)
    print("Input shape:", X.shape)
    print("Threshold:", threshold)
    print("Skipped:", skipped)
    print("Total samples:", len(y))
    print(
        "J:",
        int(np.sum(source_labels == "J")),
    )
    print(
        "Z negatives:",
        int(np.sum(source_labels == "Z")),
    )
    print(
        "NONE negatives:",
        int(np.sum(source_labels == "NONE")),
    )
    print()
    print(
        f"Accuracy: {accuracy * 100:.2f}%"
    )

    print()
    print("=" * 72)
    print("CLASSIFICATION REPORT")
    print("=" * 72)

    print(
        classification_report(
            y,
            predictions,
            labels=[0, 1],
            target_names=[
                "NOT_J",
                "J",
            ],
            digits=4,
            zero_division=0,
        )
    )

    print("=" * 72)
    print("CONFUSION MATRIX")
    print("=" * 72)
    print(
        "Rows = actual, columns = predicted"
    )
    print("Order: NOT_J, J")

    print(
        confusion_matrix(
            y,
            predictions,
            labels=[0, 1],
        )
    )

    print()
    print("=" * 72)
    print("PROBABILITY SUMMARY BY ORIGINAL CLASS")
    print("=" * 72)

    print_group_summary(
        "J",
        source_labels == "J",
        probabilities,
        predictions,
    )

    print_group_summary(
        "Z",
        source_labels == "Z",
        probabilities,
        predictions,
    )

    print_group_summary(
        "NONE",
        source_labels == "NONE",
        probabilities,
        predictions,
    )

    print()
    print("=" * 72)
    print("MISCLASSIFIED FILES")
    print("=" * 72)

    mistakes = 0

    for (
        filename,
        source_label,
        actual,
        predicted,
        probability,
    ) in zip(
        filenames,
        source_labels,
        y,
        predictions,
        probabilities,
    ):
        if actual == predicted:
            continue

        mistakes += 1

        print()
        print(filename)
        print(
            f"  original label: "
            f"{source_label}"
        )
        print(
            f"  expected binary: "
            f"{'J' if actual == 1 else 'NOT_J'}"
        )
        print(
            f"  predicted: "
            f"{'J' if predicted == 1 else 'NOT_J'}"
        )
        print(
            f"  J probability: "
            f"{probability:.4f}"
        )

    if mistakes == 0:
        print("No mistakes.")

    print()
    print("=" * 72)
    print("THRESHOLD ANALYSIS")
    print("=" * 72)

    for test_threshold in [
        0.30,
        0.40,
        0.50,
        0.60,
        0.70,
        0.80,
        0.90,
    ]:
        test_predictions = (
            probabilities >= test_threshold
        ).astype(np.int64)

        true_positive = int(
            np.sum(
                (y == 1)
                & (test_predictions == 1)
            )
        )

        false_negative = int(
            np.sum(
                (y == 1)
                & (test_predictions == 0)
            )
        )

        false_positive = int(
            np.sum(
                (y == 0)
                & (test_predictions == 1)
            )
        )

        true_negative = int(
            np.sum(
                (y == 0)
                & (test_predictions == 0)
            )
        )

        recall = (
            true_positive
            / (true_positive + false_negative)
            if true_positive + false_negative > 0
            else 0.0
        )

        precision = (
            true_positive
            / (true_positive + false_positive)
            if true_positive + false_positive > 0
            else 0.0
        )

        print(
            f"threshold={test_threshold:.2f} | "
            f"TP={true_positive:2d} "
            f"FN={false_negative:2d} "
            f"FP={false_positive:2d} "
            f"TN={true_negative:2d} | "
            f"precision={precision:.3f} "
            f"recall={recall:.3f}"
        )


if __name__ == "__main__":
    main()
