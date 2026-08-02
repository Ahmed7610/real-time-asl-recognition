import argparse
import csv
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


STATIC_CLASSES = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I",
    "K", "L", "M", "N", "O", "P", "Q", "R", "S",
    "T", "U", "V", "W", "X", "Y",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Collect normalized hand landmarks "
            "for static ASL letters."
        )
    )

    parser.add_argument(
        "--letter",
        required=True,
        type=str.upper,
        choices=STATIC_CLASSES,
        help="Static ASL letter to record.",
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=100,
        help="Number of samples to save.",
    )

    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera index, usually 0.",
    )

    parser.add_argument(
        "--person",
        type=str,
        default="ahmed",
        help="Person identifier saved with the data.",
    )

    parser.add_argument(
        "--delay",
        type=int,
        default=3,
        help="Countdown in seconds before recording starts.",
    )

    parser.add_argument(
        "--interval",
        type=int,
        default=5,
        help="Save one sample every N detected frames.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/collected_landmarks",
        help="Directory used to save the CSV file.",
    )

    return parser.parse_args()


def draw_landmarks(frame, raw_landmarks):
    height, width = frame.shape[:2]

    points = []

    for landmark in raw_landmarks:
        x = int(landmark[0] * width)
        y = int(landmark[1] * height)
        points.append((x, y))

    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (5, 9), (9, 10), (10, 11), (11, 12),
        (9, 13), (13, 14), (14, 15), (15, 16),
        (13, 17), (17, 18), (18, 19), (19, 20),
        (0, 17),
    ]

    for start, end in connections:
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


def create_csv_header():
    header = [
        "letter",
        "person",
        "session",
        "sample_number",
        "timestamp",
    ]

    for landmark_index in range(21):
        header.extend(
            [
                f"x{landmark_index}",
                f"y{landmark_index}",
                f"z{landmark_index}",
            ]
        )

    return header


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


def close_resources(camera, detector):
    print()
    print("Closing camera and MediaPipe resources...")

    if camera is not None:
        try:
            if camera.isOpened():
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

        # Allow the OpenCV GUI backend to process
        # the window-close request.
        for _ in range(5):
            cv2.waitKey(1)

    except Exception as error:
        print(f"OpenCV window cleanup warning: {error}")

    # Give the operating system a short time
    # to release the V4L2 camera device.
    time.sleep(1)

    print("Camera cleanup completed.")


def main():
    args = parse_args()

    if args.samples <= 0:
        raise ValueError("--samples must be greater than zero.")

    if args.interval <= 0:
        raise ValueError("--interval must be greater than zero.")

    output_dir = PROJECT_DIR / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_file = output_dir / (
        f"{args.person}_{args.letter}_{session_id}.csv"
    )

    camera = None
    detector = None

    saved_samples = 0
    detected_frames = 0
    recording_started = False

    try:
        camera = cv2.VideoCapture(
            args.camera,
            cv2.CAP_V4L2,
        )

        if not camera.isOpened():
            raise RuntimeError(
                f"Could not open camera index {args.camera}. "
                "Check /dev/video*, close other camera programs, "
                "or try --camera 1."
            )

        configure_camera(camera)

        actual_width = int(
            camera.get(cv2.CAP_PROP_FRAME_WIDTH)
        )

        actual_height = int(
            camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
        )

        actual_fps = camera.get(
            cv2.CAP_PROP_FPS
        )

        print()
        print(
            "Camera configuration: "
            f"{actual_width}x{actual_height} "
            f"at {actual_fps:.1f} FPS"
        )

        detector = create_hand_detector(
            static_image_mode=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        print()
        print("=" * 60)
        print(f"Recording letter: {args.letter}")
        print(f"Target samples: {args.samples}")
        print(f"Output file: {output_file}")
        print()
        print("During recording:")
        print("- Keep the correct hand sign.")
        print("- Move the hand slightly left/right.")
        print("- Move the hand slightly up/down.")
        print("- Change distance and angle slightly.")
        print("- Do not change the actual letter shape.")
        print("- Press Q to stop early.")
        print("=" * 60)

        start_time = time.time()

        with output_file.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(create_csv_header())

            while saved_samples < args.samples:
                success, frame = camera.read()

                if not success or frame is None:
                    raise RuntimeError(
                        "The camera stopped returning frames."
                    )

                # Keep this flip because the previous collected
                # datasets were recorded using the same pipeline.
                frame = cv2.flip(frame, 1)

                elapsed = time.time() - start_time
                remaining = args.delay - int(elapsed)

                if remaining > 0:
                    cv2.putText(
                        frame,
                        f"Show letter {args.letter}",
                        (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2,
                        (0, 255, 255),
                        3,
                    )

                    cv2.putText(
                        frame,
                        f"Starting in {remaining}",
                        (30, 110),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.1,
                        (0, 255, 255),
                        3,
                    )

                else:
                    recording_started = True

                    raw_landmarks = extract_raw_landmarks(
                        frame,
                        detector,
                    )

                    if raw_landmarks is not None:
                        draw_landmarks(
                            frame,
                            raw_landmarks,
                        )

                        detected_frames += 1

                        if detected_frames % args.interval == 0:
                            features = normalize_landmarks(
                                raw_landmarks
                            )

                            if features.shape != (63,):
                                raise ValueError(
                                    "Expected 63 features, "
                                    f"received {features.shape}."
                                )

                            if not np.isfinite(features).all():
                                raise ValueError(
                                    "Landmark features contain "
                                    "NaN or infinite values."
                                )

                            row = [
                                args.letter,
                                args.person,
                                session_id,
                                saved_samples + 1,
                                datetime.now().isoformat(),
                            ]

                            row.extend(
                                features.astype(
                                    np.float32
                                ).tolist()
                            )

                            writer.writerow(row)
                            csv_file.flush()

                            saved_samples += 1

                    cv2.putText(
                        frame,
                        f"Letter: {args.letter}",
                        (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (255, 255, 255),
                        3,
                    )

                    cv2.putText(
                        frame,
                        (
                            f"Saved: {saved_samples}"
                            f"/{args.samples}"
                        ),
                        (30, 95),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 255, 0),
                        3,
                    )

                    if raw_landmarks is None:
                        cv2.putText(
                            frame,
                            "No hand detected",
                            (30, 140),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 0, 255),
                            2,
                        )
                    else:
                        cv2.putText(
                            frame,
                            (
                                "Move hand slightly while "
                                "keeping the sign"
                            ),
                            (30, 140),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.65,
                            (255, 255, 0),
                            2,
                        )

                cv2.putText(
                    frame,
                    "Press Q to stop",
                    (30, frame.shape[0] - 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2,
                )

                cv2.imshow(
                    "Static ASL Data Collector",
                    frame,
                )

                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    print("Recording stopped by user.")
                    break

    except KeyboardInterrupt:
        print()
        print("Recording interrupted with Ctrl+C.")

    except Exception as error:
        print()
        print(f"ERROR: {error}")

    finally:
        close_resources(
            camera,
            detector,
        )

    print()
    print("=" * 60)
    print("Recording finished.")
    print(f"Saved samples: {saved_samples}")
    print(f"CSV file: {output_file}")
    print("=" * 60)

    if not recording_started:
        print("Warning: Recording did not start.")

    if saved_samples == 0:
        try:
            output_file.unlink()
            print(
                "No samples were saved. "
                "The empty CSV file was removed."
            )
        except FileNotFoundError:
            pass

    elif saved_samples < args.samples:
        print(
            "Warning: The recording is incomplete. "
            f"Expected {args.samples}, saved {saved_samples}."
        )


if __name__ == "__main__":
    main()
