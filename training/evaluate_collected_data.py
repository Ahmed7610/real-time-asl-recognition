import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from tensorflow.keras.models import load_model


PROJECT_DIR = Path(__file__).resolve().parents[1]

MODEL_DIR = PROJECT_DIR / "Real-Time Sign Language Recognition"
DATA_DIR = PROJECT_DIR / "data" / "collected_landmarks"

MODEL_PATH = MODEL_DIR / "StaticModel.keras"
SCALER_PATH = MODEL_DIR / "StaticScaler.pkl"
ENCODER_PATH = MODEL_DIR / "StaticLabelEncoder.pkl"

METADATA_COLUMNS = [
    "letter",
    "person",
    "session",
    "sample_number",
    "timestamp",
]


def load_collected_data():
    csv_files = sorted(DATA_DIR.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found inside: {DATA_DIR}"
        )

    frames = []

    print("=" * 70)
    print("COLLECTED FILES")
    print("=" * 70)

    for csv_file in csv_files:
        df = pd.read_csv(csv_file)

        print(
            f"{csv_file.name:<45} "
            f"rows={len(df):<5} columns={len(df.columns)}"
        )

        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)

    return combined


def validate_data(df):
    print()
    print("=" * 70)
    print("DATA VALIDATION")
    print("=" * 70)

    feature_columns = [
        column
        for column in df.columns
        if column not in METADATA_COLUMNS
    ]

    print(f"Total samples:       {len(df)}")
    print(f"Feature columns:     {len(feature_columns)}")
    print(f"Missing values:      {int(df.isna().sum().sum())}")
    print(f"Duplicated rows:     {int(df.duplicated().sum())}")
    print()

    print("Samples per letter:")
    print(df["letter"].value_counts().sort_index().to_string())

    if len(feature_columns) != 63:
        raise ValueError(
            f"Expected 63 feature columns, found {len(feature_columns)}"
        )

    if df[feature_columns].isna().any().any():
        raise ValueError(
            "The collected features contain missing values."
        )

    X = df[feature_columns].to_numpy(dtype=np.float32)
    y = df["letter"].astype(str).to_numpy()

    if not np.isfinite(X).all():
        raise ValueError(
            "The collected features contain infinite or invalid values."
        )

    feature_std = X.std(axis=0)
    near_constant_features = int(np.sum(feature_std < 1e-7))

    print()
    print(f"Near-constant features: {near_constant_features}/63")

    sample_distances = np.linalg.norm(
        np.diff(X, axis=0),
        axis=1,
    )

    if len(sample_distances) > 0:
        print(
            "Average change between consecutive samples: "
            f"{sample_distances.mean():.6f}"
        )

        print(
            "Minimum change between consecutive samples: "
            f"{sample_distances.min():.6f}"
        )

    return X, y


def evaluate_model(X, y):
    print()
    print("=" * 70)
    print("ORIGINAL MODEL EVALUATION")
    print("=" * 70)

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

    overall_accuracy = accuracy_score(y, predicted_labels)

    print(f"Overall accuracy: {overall_accuracy * 100:.2f}%")
    print()

    letters = sorted(np.unique(y))

    print(
        f"{'Letter':<10}"
        f"{'Samples':<12}"
        f"{'Correct':<12}"
        f"{'Accuracy':<14}"
        f"{'Avg confidence':<16}"
        f"Most common result"
    )

    print("-" * 80)

    for letter in letters:
        mask = y == letter

        actual_predictions = predicted_labels[mask]
        actual_confidences = confidences[mask]

        correct = int(np.sum(actual_predictions == letter))
        total = int(mask.sum())
        accuracy = correct / total

        values, counts = np.unique(
            actual_predictions,
            return_counts=True,
        )

        most_common_prediction = values[counts.argmax()]

        print(
            f"{letter:<10}"
            f"{total:<12}"
            f"{correct:<12}"
            f"{accuracy * 100:<14.2f}"
            f"{actual_confidences.mean() * 100:<16.2f}"
            f"{most_common_prediction}"
        )

    print()
    print("=" * 70)
    print("MISCLASSIFICATIONS")
    print("=" * 70)

    mistakes_exist = False

    for actual_letter in letters:
        mask = y == actual_letter
        wrong_predictions = predicted_labels[mask]
        wrong_predictions = wrong_predictions[
            wrong_predictions != actual_letter
        ]

        if len(wrong_predictions) == 0:
            print(f"{actual_letter}: No mistakes")
            continue

        mistakes_exist = True

        values, counts = np.unique(
            wrong_predictions,
            return_counts=True,
        )

        order = np.argsort(counts)[::-1]

        mistakes = ", ".join(
            f"{values[index]} ({counts[index]})"
            for index in order[:5]
        )

        print(f"{actual_letter}: {mistakes}")

    print()
    print("=" * 70)
    print("LOW-CONFIDENCE SAMPLES")
    print("=" * 70)

    for threshold in [0.50, 0.60, 0.70, 0.80, 0.90]:
        count = int(np.sum(confidences < threshold))

        print(
            f"Confidence below {threshold:.2f}: "
            f"{count}/{len(confidences)}"
        )

    return predicted_labels, confidences


def main():
    df = load_collected_data()
    X, y = validate_data(df)
    evaluate_model(X, y)

    print()
    print("Evaluation completed successfully.")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print()
        print(f"ERROR: {error}")
        sys.exit(1)
