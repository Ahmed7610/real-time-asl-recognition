from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model


PROJECT_DIR = Path(__file__).resolve().parents[1]

MODEL_DIR = PROJECT_DIR / "Real-Time Sign Language Recognition"
TEST_DIR = PROJECT_DIR / "data" / "recheck_test"

MODEL_PATH = (
    MODEL_DIR / "StaticModel_custom_unweighted.keras"
)
SCALER_PATH = MODEL_DIR / "StaticScaler.pkl"
ENCODER_PATH = MODEL_DIR / "StaticLabelEncoder.pkl"

METADATA_COLUMNS = {
    "letter",
    "person",
    "session",
    "sample_number",
    "timestamp",
}


def main():
    csv_files = sorted(TEST_DIR.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            f"No test files found in {TEST_DIR}"
        )

    frames = [
        pd.read_csv(csv_file)
        for csv_file in csv_files
    ]

    data = pd.concat(frames, ignore_index=True)

    feature_columns = [
        column
        for column in data.columns
        if column not in METADATA_COLUMNS
    ]

    X = data[feature_columns].to_numpy(dtype=np.float32)
    y = data["letter"].astype(str).to_numpy()

    model = load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    encoder = joblib.load(ENCODER_PATH)

    X_scaled = scaler.transform(X)

    probabilities = model.predict(
        X_scaled,
        verbose=0,
    )

    predicted_indexes = probabilities.argmax(axis=1)
    predicted_labels = encoder.inverse_transform(predicted_indexes)
    confidences = probabilities.max(axis=1)

    print()
    print("=" * 75)
    print("FINE-TUNED MODEL TEST")
    print("=" * 75)

    total_correct = 0

    for letter in sorted(np.unique(y)):
        mask = y == letter

        predictions = predicted_labels[mask]
        letter_confidences = confidences[mask]

        total = int(mask.sum())
        correct = int(np.sum(predictions == letter))
        total_correct += correct

        values, counts = np.unique(
            predictions,
            return_counts=True,
        )

        most_common = values[counts.argmax()]

        print(
            f"{letter}: "
            f"{correct}/{total} correct | "
            f"accuracy={correct / total * 100:.2f}% | "
            f"avg confidence={letter_confidences.mean() * 100:.2f}% | "
            f"most predicted={most_common}"
        )

    overall_accuracy = total_correct / len(y)

    print()
    print(f"Overall accuracy: {overall_accuracy * 100:.2f}%")


if __name__ == "__main__":
    main()

