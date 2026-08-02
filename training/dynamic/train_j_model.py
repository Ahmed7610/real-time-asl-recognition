from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
)

from j_utils import preprocess_j_sequence


PROJECT_DIR = Path(__file__).resolve().parents[2]

TRAIN_DIR = (
    PROJECT_DIR
    / "data"
    / "dynamic_train"
)

MODEL_PATH = (
    PROJECT_DIR
    / "Real-Time Sign Language Recognition"
    / "DynamicJModel_clean.keras"
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


def load_dataset():
    files = sorted(
        TRAIN_DIR.rglob("*.npz")
    )

    X = []
    y = []
    source_labels = []
    skipped = 0

    for file in files:
        try:
            with np.load(file) as data:
                sequence = data["landmarks"]
                label = infer_label(
                    file,
                    data,
                )

            # J model clean dataset:
            # - Use every J sample as positive.
            # - Use only the newly recorded clean NOT_J files.
            # - Ignore old NONE files because some contain real J movements.
            # - Ignore the separate Z folder because clean NOT_J already
            #   contains deliberately recorded Z movements.
            if label == "J":
                pass
            elif (
                label == "NONE"
                and "notj" in file.name.lower()
            ):
                pass
            else:
                skipped += 1
                continue

            if (
                sequence.ndim != 2
                or sequence.shape[1] != 63
                or len(sequence) < 10
                or not np.isfinite(sequence).all()
            ):
                skipped += 1
                continue

            features = preprocess_j_sequence(
                sequence
            )

            # Binary target:
            # J = 1
            # NONE or Z = 0
            target = 1 if label == "J" else 0

            X.append(features)
            y.append(target)
            source_labels.append(label)

        except Exception as error:
            print(
                f"Skipping {file.name}: {error}"
            )
            skipped += 1

    if not X:
        raise RuntimeError(
            "No valid samples loaded."
        )

    X = np.stack(X).astype(np.float32)
    y = np.asarray(y, dtype=np.int64)
    source_labels = np.asarray(source_labels)

    print()
    print("=" * 65)
    print("DYNAMIC J BINARY DATASET")
    print("=" * 65)
    print("Shape:", X.shape)
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
    print(
        "Total NOT_J:",
        int(np.sum(y == 0)),
    )
    print("Skipped:", skipped)

    return X, y


def build_model(input_shape):
    inputs = tf.keras.Input(
        shape=input_shape
    )

    x = tf.keras.layers.Conv1D(
        64,
        kernel_size=5,
        padding="same",
        activation="relu",
    )(inputs)

    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.MaxPooling1D(
        pool_size=2
    )(x)

    x = tf.keras.layers.Conv1D(
        128,
        kernel_size=3,
        padding="same",
        activation="relu",
    )(x)

    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Conv1D(
        128,
        kernel_size=3,
        padding="same",
        activation="relu",
    )(x)

    x = tf.keras.layers.GlobalAveragePooling1D()(x)

    x = tf.keras.layers.Dense(
        64,
        activation="relu",
    )(x)

    x = tf.keras.layers.Dropout(
        0.35
    )(x)

    outputs = tf.keras.layers.Dense(
        1,
        activation="sigmoid",
    )(x)

    return tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
    )


def main():
    np.random.seed(42)
    tf.random.set_seed(42)

    X, y = load_dataset()

    X_train, X_val, y_train, y_val = (
        train_test_split(
            X,
            y,
            test_size=0.25,
            random_state=42,
            stratify=y,
        )
    )

    classes = np.unique(y_train)

    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train,
    )

    class_weight = {
        int(class_index): float(weight)
        for class_index, weight
        in zip(classes, weights)
    }

    print()
    print(
        "Class weights:",
        class_weight,
    )

    model = build_model(
        X.shape[1:]
    )

    model.summary()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=1e-3
        ),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=15,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    model.fit(
        X_train,
        y_train,
        validation_data=(
            X_val,
            y_val,
        ),
        epochs=120,
        batch_size=16,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=2,
    )

    val_loss, val_accuracy = model.evaluate(
        X_val,
        y_val,
        verbose=0,
    )

    model.save(MODEL_PATH)

    print()
    print("=" * 65)
    print("J TRAINING RESULT")
    print("=" * 65)
    print(
        f"Validation loss: "
        f"{val_loss:.4f}"
    )
    print(
        f"Validation accuracy: "
        f"{val_accuracy * 100:.2f}%"
    )
    print()
    print("MODEL SAVED:")
    print(MODEL_PATH)


if __name__ == "__main__":
    main()
