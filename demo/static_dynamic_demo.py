import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf


PROJECT_DIR = Path(__file__).resolve().parents[1]

MODEL_DIR = (
    PROJECT_DIR
    / "Real-Time Sign Language Recognition"
)

sys.path.insert(0, str(MODEL_DIR))

from asl_inference import SignLanguageTranslator  # noqa: E402
from asl_landmarks import (  # noqa: E402
    extract_raw_landmarks,
    normalize_landmarks,
)


DYNAMIC_MODEL_PATH = MODEL_DIR / "DynamicZModel_trimmed.keras"

TARGET_FRAMES = 40

MIN_RECORDED_FRAMES = 25

# نسبة الفريمات التي يجب أن يكون فيها وضع اليد مناسبًا لحرف Z.
MIN_Z_POSE_RATIO = 0.55

# يمنع تكرار ظهور Z عدة مرات بسرعة.
Z_COOLDOWN_SECONDS = 1.5

# الموديل الحالي يخرج احتمالات Z الحقيقية تقريبًا بين 0.55 و0.59.
# لذلك 0.90 سيجعل Z مستحيلة عمليًا أثناء الاختبار.
Z_THRESHOLD = 0.56

# يمنع تصنيف حركة شبه ثابتة على أنها Z.
MIN_INDEX_PATH_LENGTH = 0.18


HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


def configure_camera(camera):
    camera.set(
        cv2.CAP_PROP_FOURCC,
        cv2.VideoWriter_fourcc(*"MJPG"),
    )

    camera.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        640,
    )

    camera.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        480,
    )

    camera.set(
        cv2.CAP_PROP_FPS,
        30,
    )

    camera.set(
        cv2.CAP_PROP_BUFFERSIZE,
        1,
    )


def close_resources(camera, translator):
    print()
    print("Closing camera resources...")

    if camera is not None:
        try:
            camera.release()
        except Exception:
            pass

    if translator is not None:
        try:
            translator.detector.close()
        except Exception:
            pass

    try:
        cv2.destroyAllWindows()

        for _ in range(5):
            cv2.waitKey(1)

    except Exception:
        pass

    time.sleep(1)

    print("Camera cleanup completed.")


def draw_landmarks(frame, landmarks):
    height, width = frame.shape[:2]

    points = [
        (
            int(point[0] * width),
            int(point[1] * height),
        )
        for point in landmarks
    ]

    for start, end in HAND_CONNECTIONS:
        cv2.line(
            frame,
            points[start],
            points[end],
            (0, 255, 0),
            2,
        )

    for index, point in enumerate(points):
        radius = 9 if index == 8 else 4

        color = (
            (255, 0, 255)
            if index == 8
            else (0, 0, 255)
        )

        cv2.circle(
            frame,
            point,
            radius,
            color,
            -1,
        )


def resample_sequence(sequence, target_frames):
    sequence = np.asarray(
        sequence,
        dtype=np.float32,
    )

    old_length = len(sequence)

    if old_length < 2:
        raise ValueError(
            "Dynamic sequence needs at least 2 frames."
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

    for feature_index in range(
        sequence.shape[1]
    ):
        output[:, feature_index] = np.interp(
            new_positions,
            old_positions,
            sequence[:, feature_index],
        )

    return output


def build_dynamic_features(
    sequence,
    expected_features,
):
    sequence = np.asarray(
        sequence,
        dtype=np.float32,
    )

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

    if expected_features == 132:
        # 63 landmarks + 63 velocity
        # + 3 index position + 3 index velocity
        return np.concatenate(
            [
                sequence,
                velocity,
                index_tip,
                index_velocity,
            ],
            axis=1,
        ).astype(np.float32)

    if expected_features == 135:
        return np.concatenate(
            [
                sequence,
                velocity,
                index_tip,
                index_velocity,
                index_acceleration,
            ],
            axis=1,
        ).astype(np.float32)

    if expected_features == 9:
        trajectory = (
            index_tip - index_tip[0:1]
        )

        trajectory_velocity = np.diff(
            trajectory,
            axis=0,
            prepend=trajectory[:1],
        )

        trajectory_acceleration = np.diff(
            trajectory_velocity,
            axis=0,
            prepend=trajectory_velocity[:1],
        )

        return np.concatenate(
            [
                trajectory,
                trajectory_velocity,
                trajectory_acceleration,
            ],
            axis=1,
        ).astype(np.float32)

    raise ValueError(
        "Unsupported dynamic model feature size: "
        f"{expected_features}"
    )


def index_path_length(sequence):
    sequence = np.asarray(
        sequence,
        dtype=np.float32,
    )

    index_tip = sequence[
        :,
        8 * 3 : 8 * 3 + 3,
    ]

    steps = np.diff(
        index_tip,
        axis=0,
    )

    return float(
        np.linalg.norm(
            steps,
            axis=1,
        ).sum()
    )



def finger_is_extended(
    landmarks,
    tip_index,
    pip_index,
    margin=0.03,
):
    """
    تحديد ما إذا كان الإصبع ممدودًا باستخدام المسافة من الرسغ.

    هذه الطريقة أكثر ثباتًا من مقارنة محور Y فقط،
    لأنها تعمل حتى لو كانت اليد مائلة قليلًا.
    """
    points = np.asarray(
        landmarks,
        dtype=np.float32,
    )

    wrist = points[0]
    tip = points[tip_index]
    pip = points[pip_index]

    tip_distance = float(
        np.linalg.norm(tip - wrist)
    )

    pip_distance = float(
        np.linalg.norm(pip - wrist)
    )

    return (
        tip_distance
        > pip_distance + margin
    )


def is_z_hand_pose(landmarks):
    """
    وضع حرف Z:
    - السبابة ممدودة.
    - الوسطى والبنصر والخنصر غير ممدودة.
    - الإبهام لا ندخله في القرار لأنه يختلف بين الأشخاص.
    """
    index_extended = finger_is_extended(
        landmarks,
        tip_index=8,
        pip_index=6,
        margin=0.025,
    )

    middle_extended = finger_is_extended(
        landmarks,
        tip_index=12,
        pip_index=10,
        margin=0.015,
    )

    ring_extended = finger_is_extended(
        landmarks,
        tip_index=16,
        pip_index=14,
        margin=0.015,
    )

    pinky_extended = finger_is_extended(
        landmarks,
        tip_index=20,
        pip_index=18,
        margin=0.015,
    )

    return (
        index_extended
        and not middle_extended
        and not ring_extended
        and not pinky_extended
    )


def calculate_pose_ratio(pose_history):
    if not pose_history:
        return 0.0

    return float(
        np.mean(
            np.asarray(
                pose_history,
                dtype=np.float32,
            )
        )
    )

def classify_dynamic(
    dynamic_model,
    sequence,
):
    resampled = resample_sequence(
        sequence,
        TARGET_FRAMES,
    )

    expected_features = int(
        dynamic_model.input_shape[-1]
    )

    features = build_dynamic_features(
        resampled,
        expected_features,
    )

    if features.shape != (
        TARGET_FRAMES,
        expected_features,
    ):
        raise ValueError(
            "Unexpected dynamic feature shape: "
            f"{features.shape}; expected "
            f"({TARGET_FRAMES}, {expected_features})"
        )

    print(
        "Dynamic feature shape:",
        features.shape,
    )

    probability = float(
        dynamic_model.predict(
            features[None, ...],
            verbose=0,
        ).reshape(-1)[0]
    )

    path_length = index_path_length(
        resampled
    )

    is_z = (
        probability >= Z_THRESHOLD
        and path_length >= MIN_INDEX_PATH_LENGTH
    )

    return is_z, probability, path_length


def main():
    camera = None
    translator = None

    try:
        print("Loading static model...")
        translator = SignLanguageTranslator(
            str(MODEL_DIR)
        )

        print("Loading dynamic Z model...")
        dynamic_model = tf.keras.models.load_model(
            DYNAMIC_MODEL_PATH
        )

        print(
            "Dynamic input shape:",
            dynamic_model.input_shape,
        )

        camera = cv2.VideoCapture(
            0,
            cv2.CAP_V4L2,
        )

        if not camera.isOpened():
            raise RuntimeError(
                "Could not open camera 0."
            )

        configure_camera(camera)

        recording_dynamic = False
        dynamic_sequence = []

        displayed_letter = "–"
        displayed_source = "STATIC"
        dynamic_probability = 0.0
        dynamic_path = 0.0
        pose_ratio = 0.0

        # نتيجة Z تظهر مدة قصيرة بعد التصنيف.
        z_display_until = 0.0

        static_history = deque(maxlen=5)

        last_timestamp_ms = 0
        dynamic_pose_history = []
        last_z_time = 0.0

        print()
        print("=" * 65)
        print("STATIC + DYNAMIC Z DEMO")
        print("=" * 65)
        print("R : start recording Z movement")
        print("S : stop and classify movement")
        print("C : cancel current dynamic recording")
        print("Q : quit")
        print("=" * 65)

        while True:
            success, frame = camera.read()

            if not success or frame is None:
                raise RuntimeError(
                    "Camera stopped returning frames."
                )

            frame = cv2.flip(
                frame,
                1,
            )

            timestamp_ms = int(
                time.monotonic() * 1000
            )

            if timestamp_ms <= last_timestamp_ms:
                timestamp_ms = last_timestamp_ms + 1

            last_timestamp_ms = timestamp_ms

            landmarks = extract_raw_landmarks(
                frame,
                translator.detector,
                timestamp_ms=timestamp_ms,
            )

            if landmarks is not None:
                draw_landmarks(
                    frame,
                    landmarks,
                )

                normalized = normalize_landmarks(
                    landmarks
                )

                if recording_dynamic:
                    dynamic_sequence.append(
                        normalized.astype(
                            np.float32
                        )
                    )

                    dynamic_pose_history.append(
                        is_z_hand_pose(
                            landmarks
                        )
                    )

                else:
                    result = translator.process_landmarks(
                        landmarks
                    )

                    raw_letter = result.get(
                        "raw_letter",
                        "nothing",
                    )

                    static_history.append(
                        raw_letter
                    )

                    if (
                        time.time()
                        >= z_display_until
                    ):
                        letter = result["letter"]

                        displayed_letter = (
                            "–"
                            if letter == "nothing"
                            else letter
                        )

                        displayed_source = "STATIC"

            else:
                if not recording_dynamic:
                    translator.reset()
                    static_history.clear()

                    if (
                        time.time()
                        >= z_display_until
                    ):
                        displayed_letter = "–"
                        displayed_source = "NO HAND"

            if recording_dynamic:
                displayed_letter = "..."
                displayed_source = "RECORDING Z"

                cv2.rectangle(
                    frame,
                    (0, 0),
                    (frame.shape[1], 70),
                    (0, 0, 180),
                    -1,
                )

                cv2.putText(
                    frame,
                    (
                        "RECORDING DYNAMIC: "
                        f"{len(dynamic_sequence)} frames"
                    ),
                    (20, 43),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                )

            cv2.putText(
                frame,
                f"Result: {displayed_letter}",
                (20, frame.shape[0] - 105),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 255),
                3,
            )

            cv2.putText(
                frame,
                f"Source: {displayed_source}",
                (20, frame.shape[0] - 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
            )

            cv2.putText(
                frame,
                (
                    f"Z prob: {dynamic_probability:.3f} | "
                    f"path: {dynamic_path:.3f}"
                ),
                (20, frame.shape[0] - 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (255, 255, 255),
                2,
            )

            cv2.putText(
                frame,
                f"Z pose ratio: {pose_ratio:.0%}",
                (20, frame.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
            )

            cv2.putText(
                frame,
                "R=start Z | S=stop | C=cancel | Q=quit",
                (20, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (255, 255, 255),
                2,
            )

            cv2.imshow(
                "Static + Dynamic Sign Recognition",
                frame,
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == ord("r"):
                dynamic_sequence = []
                dynamic_pose_history = []
                recording_dynamic = True
                translator.reset()

                print()
                print("Dynamic recording started.")

            elif key == ord("c"):
                dynamic_sequence = []
                dynamic_pose_history = []
                recording_dynamic = False

                print("Dynamic recording cancelled.")

            elif key == ord("s"):
                if not recording_dynamic:
                    print(
                        "Press R before pressing S."
                    )
                    continue

                recording_dynamic = False

                if len(dynamic_sequence) < MIN_RECORDED_FRAMES:
                    print(
                        f"Too few frames: {len(dynamic_sequence)}. "
                        f"Record at least {MIN_RECORDED_FRAMES} frames."
                    )

                    displayed_letter = "NONE"
                    displayed_source = "DYNAMIC"
                    continue

                try:
                    (
                        is_z,
                        dynamic_probability,
                        dynamic_path,
                    ) = classify_dynamic(
                        dynamic_model,
                        dynamic_sequence,
                    )

                    pose_ratio = calculate_pose_ratio(
                        dynamic_pose_history
                    )

                    cooldown_ready = (
                        time.time() - last_z_time
                        >= Z_COOLDOWN_SECONDS
                    )

                    final_is_z = (
                        is_z
                        and pose_ratio
                        >= MIN_Z_POSE_RATIO
                        and cooldown_ready
                    )

                    if final_is_z:
                        displayed_letter = "Z"
                        last_z_time = time.time()
                        z_display_until = (
                            time.time() + 2.0
                        )
                    else:
                        displayed_letter = "NONE"

                    displayed_source = "DYNAMIC"

                    print()
                    print(
                        f"Dynamic result: "
                        f"{'Z' if final_is_z else 'NONE'}"
                    )
                    print(
                        f"Z probability: "
                        f"{dynamic_probability:.4f}"
                    )
                    print(
                        f"Index path length: "
                        f"{dynamic_path:.4f}"
                    )
                    print(
                        f"Z hand-pose ratio: "
                        f"{pose_ratio:.2%}"
                    )
                    print(
                        f"Cooldown ready: "
                        f"{cooldown_ready}"
                    )
                    print(
                        f"Recorded frames: "
                        f"{len(dynamic_sequence)}"
                    )

                except Exception as error:
                    print(
                        f"Dynamic classification error: "
                        f"{error}"
                    )

                finally:
                    dynamic_sequence = []
                    dynamic_pose_history = []
                    translator.reset()
                    static_history.clear()

    except KeyboardInterrupt:
        print()
        print("Interrupted.")

    except Exception as error:
        print()
        print(f"ERROR: {error}")

    finally:
        close_resources(
            camera,
            translator,
        )


if __name__ == "__main__":
    main()