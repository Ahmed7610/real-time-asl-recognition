import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_DIR / "Real-Time Sign Language Recognition"

sys.path.insert(0, str(MODEL_DIR))

from asl_landmarks import (  # noqa: E402
    create_hand_detector,
    extract_raw_landmarks,
    normalize_landmarks,
)


VALID_LABELS = ["Z", "NONE"]

MIN_SEQUENCE_FRAMES = 15
MAX_SEQUENCE_FRAMES = 120

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect variable-length dynamic ASL sequences."
    )

    parser.add_argument(
        "--label",
        required=True,
        type=str.upper,
        choices=VALID_LABELS,
    )

    parser.add_argument(
        "--sequences",
        type=int,
        default=40,
    )

    parser.add_argument(
        "--camera",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--person",
        type=str,
        default="ahmed",
    )

    parser.add_argument(
        "--session",
        type=str,
        default="session1",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/dynamic_train",
    )

    return parser.parse_args()


def configure_camera(camera):
    camera.set(
        cv2.CAP_PROP_FOURCC,
        cv2.VideoWriter_fourcc(*"MJPG"),
    )
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    camera.set(cv2.CAP_PROP_FPS, 30)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)


def draw_landmarks(frame, landmarks):
    height, width = frame.shape[:2]

    points = [
        (
            int(landmark[0] * width),
            int(landmark[1] * height),
        )
        for landmark in landmarks
    ]

    for start, end in HAND_CONNECTIONS:
        cv2.line(
            frame,
            points[start],
            points[end],
            (0, 255, 0),
            2,
        )

    for point in points:
        cv2.circle(
            frame,
            point,
            4,
            (0, 0, 255),
            -1,
        )

    return points


def close_resources(camera, detector):
    print()
    print("Closing camera and MediaPipe resources...")

    if camera is not None:
        try:
            camera.release()
        except Exception as error:
            print(f"Camera release warning: {error}")

    if detector is not None:
        try:
            detector.close()
        except Exception as error:
            print(f"Detector close warning: {error}")

    try:
        cv2.destroyAllWindows()

        for _ in range(5):
            cv2.waitKey(1)

    except Exception:
        pass

    time.sleep(1)
    print("Camera cleanup completed.")


def save_sequence(
    sequence,
    output_dir,
    label,
    person,
    session,
    sequence_number,
    duration_seconds,
):
    label_dir = output_dir / label
    label_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    )

    output_file = label_dir / (
        f"{person}_{session}_{label}_"
        f"{sequence_number:04d}_{timestamp}.npz"
    )

    array = np.asarray(
        sequence,
        dtype=np.float32,
    )

    np.savez_compressed(
        output_file,
        landmarks=array,
        label=label,
        person=person,
        session=session,
        duration_seconds=np.float32(duration_seconds),
        original_frames=np.int32(len(array)),
    )

    return output_file


def main():
    args = parse_args()

    output_dir = PROJECT_DIR / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    camera = None
    detector = None

    recording = False
    current_sequence = []
    index_trajectory = []

    recording_start_time = None
    saved_sequences = 0

    try:
        camera = cv2.VideoCapture(
            args.camera,
            cv2.CAP_V4L2,
        )

        if not camera.isOpened():
            raise RuntimeError(
                f"Could not open camera index {args.camera}."
            )

        configure_camera(camera)

        detector = create_hand_detector(
            static_image_mode=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        video_start = time.monotonic()
        previous_timestamp = -1

        print()
        print("=" * 70)
        print(f"Dynamic label: {args.label}")
        print(f"Target sequences: {args.sequences}")
        print()
        print("Press Z once to START recording.")
        print("Draw the motion naturally.")
        print("Press Z again to STOP and save.")
        print("Press C while recording to cancel.")
        print("Press Q to quit.")
        print("=" * 70)

        while saved_sequences < args.sequences:
            success, frame = camera.read()

            if not success or frame is None:
                raise RuntimeError(
                    "Camera stopped returning frames."
                )

            frame = cv2.flip(frame, 1)

            timestamp_ms = int(
                (time.monotonic() - video_start) * 1000
            )

            if timestamp_ms <= previous_timestamp:
                timestamp_ms = previous_timestamp + 1

            previous_timestamp = timestamp_ms

            landmarks = extract_raw_landmarks(
                frame,
                detector,
                timestamp_ms=timestamp_ms,
            )

            points = None

            if landmarks is not None:
                points = draw_landmarks(
                    frame,
                    landmarks,
                )

            if recording:
                if landmarks is None:
                    cv2.putText(
                        frame,
                        "HAND LOST - sequence will be cancelled",
                        (20, 210),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (0, 0, 255),
                        2,
                    )
                else:
                    features = normalize_landmarks(
                        landmarks
                    )

                    if features.shape != (63,):
                        raise ValueError(
                            f"Expected 63 features, "
                            f"received {features.shape}."
                        )

                    if not np.isfinite(features).all():
                        raise ValueError(
                            "Invalid landmark values detected."
                        )

                    current_sequence.append(features)

                    if points is not None:
                        index_trajectory.append(points[8])

                    for point_index in range(
                        1,
                        len(index_trajectory),
                    ):
                        cv2.line(
                            frame,
                            index_trajectory[
                                point_index - 1
                            ],
                            index_trajectory[
                                point_index
                            ],
                            (255, 0, 255),
                            3,
                        )

                if len(current_sequence) >= MAX_SEQUENCE_FRAMES:
                    recording = False
                    current_sequence = []
                    index_trajectory = []
                    recording_start_time = None

                    print(
                        "Sequence cancelled: too long. "
                        "Keep each motion below about 4 seconds."
                    )

            status = (
                "RECORDING - press Z to stop"
                if recording
                else "READY - press Z to start"
            )

            status_color = (
                (0, 0, 255)
                if recording
                else (0, 255, 0)
            )

            cv2.putText(
                frame,
                f"Label: {args.label}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )

            cv2.putText(
                frame,
                status,
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.72,
                status_color,
                2,
            )

            cv2.putText(
                frame,
                (
                    f"Sequences: {saved_sequences}"
                    f"/{args.sequences}"
                ),
                (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.72,
                (255, 255, 255),
                2,
            )

            if recording:
                duration = (
                    time.monotonic()
                    - recording_start_time
                )

                cv2.putText(
                    frame,
                    (
                        f"Frames: {len(current_sequence)} "
                        f"| Time: {duration:.2f}s"
                    ),
                    (20, 150),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.68,
                    (255, 255, 0),
                    2,
                )

            cv2.putText(
                frame,
                "Z: start/stop | C: cancel | Q: quit",
                (20, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (255, 255, 255),
                2,
            )

            cv2.imshow(
                "Dynamic ASL Sequence Collector",
                frame,
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == ord("c") and recording:
                recording = False
                current_sequence = []
                index_trajectory = []
                recording_start_time = None

                print("Sequence cancelled manually.")

            if key == ord("z"):
                if not recording:
                    if landmarks is None:
                        print(
                            "Cannot start: no hand detected."
                        )
                        continue

                    recording = True
                    current_sequence = []
                    index_trajectory = []
                    recording_start_time = time.monotonic()

                    print(
                        "Recording started. "
                        "Draw the motion, then press Z."
                    )

                else:
                    duration_seconds = (
                        time.monotonic()
                        - recording_start_time
                    )

                    if landmarks is None:
                        print(
                            "Sequence cancelled: "
                            "hand was lost at the end."
                        )

                    elif (
                        len(current_sequence)
                        < MIN_SEQUENCE_FRAMES
                    ):
                        print(
                            "Sequence cancelled: too short. "
                            f"Only {len(current_sequence)} frames."
                        )

                    else:
                        saved_sequences += 1

                        output_file = save_sequence(
                            current_sequence,
                            output_dir,
                            args.label,
                            args.person,
                            args.session,
                            saved_sequences,
                            duration_seconds,
                        )

                        print(
                            f"Saved sequence "
                            f"{saved_sequences}/"
                            f"{args.sequences} | "
                            f"frames={len(current_sequence)} | "
                            f"time={duration_seconds:.2f}s | "
                            f"{output_file.name}"
                        )

                    recording = False
                    current_sequence = []
                    index_trajectory = []
                    recording_start_time = None

    except KeyboardInterrupt:
        print("Recording interrupted.")

    except Exception as error:
        print(f"ERROR: {error}")

    finally:
        close_resources(
            camera,
            detector,
        )

    print()
    print("=" * 70)
    print("Dynamic recording finished.")
    print(f"Saved sequences: {saved_sequences}")
    print(f"Output directory: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
