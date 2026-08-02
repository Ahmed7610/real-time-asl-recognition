from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import load_model


PROJECT_DIR = Path(__file__).resolve().parents[1]

MODEL_DIR = PROJECT_DIR / "Real-Time Sign Language Recognition"
TRAIN_DIR = PROJECT_DIR / "data" / "collected_landmarks"

ORIGINAL_MODEL_PATH = (
    PROJECT_DIR
    / "backups"
    / "models"
    / "StaticModel_original.keras"
	)
SCALER_PATH = MODEL_DIR / "StaticScaler.pkl"
ENCODER_PATH = MODEL_DIR / "StaticLabelEncoder.pkl"

OUTPUT_MODEL_PATH = MODEL_DIR / "StaticModel_custom_unweighted.keras"

METADATA_COLUMNS = {
    "letter",
    "person",
    "session",
    "sample_number",
    "timestamp",
}


def load_training_data():
    csv_files = sorted(TRAIN_DIR.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            f"No training CSV files found in {TRAIN_DIR}"
        )

    frames = []

    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        frames.append(df)
        print(f"Loaded {csv_file.name}: {len(df)} samples")

    data = pd.concat(frames, ignore_index=True)

    feature_columns = [
        column
        for column in data.columns
        if column not in METADATA_COLUMNS
    ]

    if len(feature_columns) != 63:
        raise ValueError(
            f"Expected 63 features, found {len(feature_columns)}"
        )

    X = data[feature_columns].to_numpy(dtype=np.float32)
    labels = data["letter"].astype(str).to_numpy()

    print()
    print("Samples per letter:")
    print(data["letter"].value_counts().sort_index())

    return X, labels


def main():
    np.random.seed(42)
    tf.random.set_seed(42)

    X, labels = load_training_data()

    scaler = joblib.load(SCALER_PATH)
    encoder = joblib.load(ENCODER_PATH)
    model = load_model(ORIGINAL_MODEL_PATH)

    unknown_labels = sorted(set(labels) - set(encoder.classes_))

    if unknown_labels:
        raise ValueError(
            f"Labels not present in original encoder: {unknown_labels}"
        )

    X_scaled = scaler.transform(X).astype(np.float32)
    y = encoder.transform(labels)
    # class_weight = {
    # 	index: 1.0
    # 	for index in range(len(encoder.classes_))
	# }

    # weak_class_weights = {
    # 	"M": 2.0,
    # 	"N": 1.4,
    # 	"S": 2.0,
    # 	"T": 1.7,
    # 	"U": 2.0,
	# }

    # for letter, weight in weak_class_weights.items():
    # 	class_index = int(encoder.transform([letter])[0])
    # 	class_weight[class_index] = weight

    # print()
    # print("Class weights for weak letters:")
    # for letter, weight in weak_class_weights.items():
    # 	print(f"{letter}: {weight}")

    X_train, X_validation, y_train, y_validation = train_test_split(
        X_scaled,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    print()
    print("=" * 65)
    print("MODEL LAYERS")
    print("=" * 65)

    for index, layer in enumerate(model.layers):
        print(
            f"{index:<3} {layer.name:<25} "
            f"{layer.__class__.__name__:<20}"
        )

    # Freeze the full model first.
    for layer in model.layers:
    	layer.trainable = False

    for layer in model.layers:
    	if isinstance(
    	    layer,
    	    (
    	        tf.keras.layers.Dense,
    	        tf.keras.layers.BatchNormalization,
    	    ),
    	):
            layer.trainable = True

    print()
    print("Trainable layers:")

    for layer in model.layers:
        if layer.trainable:
            print(f"- {layer.name}")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.0001
        ),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=6,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
    ]

    print()
    print("=" * 65)
    print("STARTING FINE-TUNING")
    print("=" * 65)

    model.fit(
        X_train,
        y_train,
        validation_data=(X_validation, y_validation),
        epochs=100,
        batch_size=32,
        # class_weight=class_weight
        callbacks=callbacks,
        verbose=2,
    )

    train_loss, train_accuracy = model.evaluate(
        X_train,
        y_train,
        verbose=0,
    )

    validation_loss, validation_accuracy = model.evaluate(
        X_validation,
        y_validation,
        verbose=0,
    )

    print()
    print("=" * 65)
    print("TRAINING RESULT")
    print("=" * 65)
    print(f"Training accuracy:   {train_accuracy * 100:.2f}%")
    print(f"Validation accuracy: {validation_accuracy * 100:.2f}%")

    model.save(OUTPUT_MODEL_PATH)

    print()
    print(f"Saved new model to:")
    print(OUTPUT_MODEL_PATH)
    print()
    print("The original StaticModel.keras was not modified.")


if __name__ == "__main__":
    main()