from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from dynamic_utils import (
    TARGET_FRAMES,
    preprocess_sequence,
)


PROJECT_DIR = Path(__file__).resolve().parents[2]

TEST_DIR = PROJECT_DIR / "data" / "dynamic_test"

# لو لم توجد بيانات اختبار مستقلة، يمكن تشغيله مؤقتًا
# على dynamic_train فقط للتأكد أن الـpipeline يعمل.
FALLBACK_DIR = PROJECT_DIR / "data" / "dynamic_train"

MODEL_PATH = (
    PROJECT_DIR
    / "Real-Time Sign Language Recognition"
    / "DynamicJZModel.keras"
)

LABEL_TO_INDEX = {
    "NONE": 0,
    "J": 1,
    "Z": 2,
}

INDEX_TO_LABEL = {
    0: "NONE",
    1: "J",
    2: "Z",
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

    if "_J_" in name:
        return "J"

    if "_Z_" in name:
        return "Z"

    return None


def load_data():
    data_dir = TEST_DIR

    files = sorted(
        data_dir.rglob("*.npz")
    )

    if not files:
        print(
            f"No test files found in {TEST_DIR}."
        )
        print(
            "Using training data temporarily "
            "for pipeline verification."
        )

        data_dir = FALLBACK_DIR
        files = sorted(
            data_dir.rglob("*.npz")
        )

    if not files:
        raise FileNotFoundError(
            "No NPZ files were found."
        )

    X = []
    y = []
    filenames = []
    skipped = 0

    for file in files:
        try:
            with np.load(file) as data:
                sequence = data["landmarks"]
                label = infer_label(file, data)

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

            if label not in LABEL_TO_INDEX:
                print(
                    f"Skipping unknown label "
                    f"{label}: {file}"
                )
                skipped += 1
                continue

            features = preprocess_sequence(
                sequence,
                TARGET_FRAMES,
            )

            X.append(features)
            y.append(
                LABEL_TO_INDEX[label]
            )
            filenames.append(file.name)

        except Exception as error:
            print(
                f"Skipping {file.name}: {error}"
            )
            skipped += 1

    if not X:
        raise RuntimeError(
            "No valid samples were loaded."
        )

    return (
        np.stack(X).astype(np.float32),
        np.asarray(y, dtype=np.int64),
        filenames,
        data_dir,
        skipped,
    )


def main():
    X, y, filenames, data_dir, skipped = (
        load_data()
    )

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    if model.input_shape[1:] != X.shape[1:]:
        raise ValueError(
            "Input shape mismatch: "
            f"model expects {model.input_shape[1:]}, "
            f"data is {X.shape[1:]}"
        )

    probabilities = model.predict(
        X,
        verbose=0,
    )

    predictions = np.argmax(
        probabilities,
        axis=1,
    )

    confidence = np.max(
        probabilities,
        axis=1,
    )

    accuracy = accuracy_score(
        y,
        predictions,
    )

    print()
    print("=" * 72)
    print("DYNAMIC J/Z MODEL TEST")
    print("=" * 72)
    print("Data directory:", data_dir)
    print("Input shape:", X.shape)
    print("Skipped:", skipped)
    print("Samples:", len(y))

    for class_index, class_name in INDEX_TO_LABEL.items():
        print(
            f"{class_name}: "
            f"{int(np.sum(y == class_index))}"
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
            labels=[0, 1, 2],
            target_names=[
                "NONE",
                "J",
                "Z",
            ],
            digits=4,
            zero_division=0,
        )
    )

    print("=" * 72)
    print("CONFUSION MATRIX")
    print("=" * 72)

    matrix = confusion_matrix(
        y,
        predictions,
        labels=[0, 1, 2],
    )

    print("Rows = actual, columns = predicted")
    print("Order: NONE, J, Z")
    print(matrix)

    print()
    print("=" * 72)
    print("PROBABILITY SUMMARY")
    print("=" * 72)

    for class_index, class_name in INDEX_TO_LABEL.items():
        mask = y == class_index

        if not np.any(mask):
            continue

        correct_class_probabilities = (
            probabilities[
                mask,
                class_index,
            ]
        )

        class_confidence = confidence[mask]

        print()
        print(class_name)
        print(
            "  correct-class probability:"
        )
        print(
            f"    min={correct_class_probabilities.min():.4f}"
        )
        print(
            f"    mean={correct_class_probabilities.mean():.4f}"
        )
        print(
            f"    max={correct_class_probabilities.max():.4f}"
        )
        print(
            f"  predicted confidence mean="
            f"{class_confidence.mean():.4f}"
        )

    print()
    print("=" * 72)
    print("MISCLASSIFIED FILES")
    print("=" * 72)

    mistakes = 0

    for (
        filename,
        actual,
        predicted,
        sample_probabilities,
    ) in zip(
        filenames,
        y,
        predictions,
        probabilities,
    ):
        if actual == predicted:
            continue

        mistakes += 1

        probability_text = ", ".join(
            f"{INDEX_TO_LABEL[index]}="
            f"{sample_probabilities[index]:.4f}"
            for index in range(3)
        )

        print()
        print(filename)
        print(
            f"  actual: "
            f"{INDEX_TO_LABEL[actual]}"
        )
        print(
            f"  predicted: "
            f"{INDEX_TO_LABEL[predicted]}"
        )
        print(
            f"  probabilities: "
            f"{probability_text}"
        )

    if mistakes == 0:
        print("No mistakes.")

    print()
    print("=" * 72)
    print("LOW-CONFIDENCE CORRECT PREDICTIONS")
    print("=" * 72)

    found = 0

    for (
        filename,
        actual,
        predicted,
        sample_confidence,
        sample_probabilities,
    ) in zip(
        filenames,
        y,
        predictions,
        confidence,
        probabilities,
    ):
        if (
            actual == predicted
            and sample_confidence < 0.80
        ):
            found += 1

            probability_text = ", ".join(
                f"{INDEX_TO_LABEL[index]}="
                f"{sample_probabilities[index]:.4f}"
                for index in range(3)
            )

            print()
            print(filename)
            print(
                f"  actual/predicted: "
                f"{INDEX_TO_LABEL[actual]}"
            )
            print(
                f"  confidence: "
                f"{sample_confidence:.4f}"
            )
            print(
                f"  probabilities: "
                f"{probability_text}"
            )

    if found == 0:
        print(
            "No correct predictions below "
            "0.80 confidence."
        )


if __name__ == "__main__":
    main()
