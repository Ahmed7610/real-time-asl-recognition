from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
)


PROJECT_DIR = Path(__file__).resolve().parents[2]

TRAIN_DIR = PROJECT_DIR / "data" / "dynamic_train"

MODEL_DIR = (
    PROJECT_DIR
    / "Real-Time Sign Language Recognition"
)

OUTPUT_MODEL = (
    MODEL_DIR / "DynamicZModel_trimmed.keras"
)

TARGET_FRAMES = 40

LABEL_TO_INDEX = {
    "NONE": 0,
    "Z": 1,
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


def load_dataset():
    files = sorted(
        TRAIN_DIR.rglob("*.npz")
    )

    if not files:
        raise FileNotFoundError(
            f"No NPZ files found in {TRAIN_DIR}"
        )

    X = []
    y = []

    skipped = 0

    for file in files:
        try:
            data = np.load(file)
            sequence = data["landmarks"]

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

            label = infer_label(
                file,
                data,
            )

            if label not in LABEL_TO_INDEX:
                print(
                    f"Skipping unknown label "
                    f"{label}: {file}"
                )
                skipped += 1
                continue

            sequence = trim_motion(
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
            y.append(
                LABEL_TO_INDEX[label]
            )

        except Exception as error:
            print(
                f"Skipping {file.name}: {error}"
            )
            skipped += 1

    if not X:
        raise RuntimeError(
            "No valid sequences were loaded."
        )

    X = np.stack(X).astype(
        np.float32
    )

    y = np.asarray(
        y,
        dtype=np.int64,
    )

    print()
    print("=" * 65)
    print("DATASET")
    print("=" * 65)
    print("Shape:", X.shape)
    print("NONE:", int(np.sum(y == 0)))
    print("Z:", int(np.sum(y == 1)))
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

    model = build_model(
        X.shape[1:]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=1e-3
        ),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(
                name="precision"
            ),
            tf.keras.metrics.Recall(
                name="recall"
            ),
        ],
    )

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=12,
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
        epochs=100,
        batch_size=16,
        callbacks=callbacks,
        verbose=2,
    )

    model.save(
        OUTPUT_MODEL
    )

    print()
    print("=" * 65)
    print("MODEL SAVED")
    print("=" * 65)
    print(OUTPUT_MODEL)


if __name__ == "__main__":
    main()