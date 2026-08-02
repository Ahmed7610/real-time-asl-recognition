import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_DIR / "Real-Time Sign Language Recognition"
sys.path.insert(0, str(MODEL_DIR))

from asl_landmarks import create_hand_detector, extract_raw_landmarks, normalize_landmarks  # noqa: E402

CLASSES = ["Z", "J", "NONE"]
MIN_FRAMES = 25
MAX_FRAMES = 140
MIN_TRACKING_RATIO = 0.85

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
        description="Collect dynamic ASL sequences in the format expected by train_z_model.py."
    )
    parser.add_argument("--label", required=True, choices=CLASSES)
    parser.add_argument("--sequences", type=int, default=30)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--person", type=str, default="ahmed")
    return parser.parse_args()


def configure_camera(camera):
    camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    camera.set(cv2.CAP_PROP_FPS, 30)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)


def close_resources(camera, detector):
    print("\nClosing camera resources...")
    if camera is not None:
        try:
            camera.release()
        except Exception:
            pass
    if detector is not None:
        try:
            detector.close()
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
    points = [(int(point[0] * width), int(point[1] * height)) for point in landmarks]

    for start, end in HAND_CONNECTIONS:
        cv2.line(frame, points[start], points[end], (0, 255, 0), 2)

    for index, point in enumerate(points):
        if index == 8:
            radius = 9
            color = (255, 0, 255)  # Index fingertip for Z
        elif index == 20:
            radius = 9
            color = (255, 255, 0)  # Pinky fingertip for J
        else:
            radius = 4
            color = (0, 0, 255)

        cv2.circle(frame, point, radius, color, -1)


def save_sequence(output_dir, person, label, sequence):
    sequence_array = np.stack(sequence, axis=0).astype(np.float32)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_file = output_dir / f"{person}_{label}_{timestamp}.npz"

    np.savez_compressed(
        output_file,
        landmarks=sequence_array,
        label=np.asarray(label),
        person=np.asarray(person),
    )
    return output_file


def main():
    args = parse_args()
    output_dir = PROJECT_DIR / "data" / "dynamic_train" / args.label
    output_dir.mkdir(parents=True, exist_ok=True)

    camera = None
    detector = None

    try:
        camera = cv2.VideoCapture(args.camera, cv2.CAP_V4L2)
        if not camera.isOpened():
            raise RuntimeError(f"Could not open camera {args.camera}.")

        configure_camera(camera)
        detector = create_hand_detector(
            static_image_mode=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        print("\n" + "=" * 68)
        print("DYNAMIC SEQUENCE COLLECTOR V2")
        print("=" * 68)
        print(f"Label: {args.label}")
        print(f"Target sequences: {args.sequences}")
        print(f"Output: {output_dir}")
        print("\nControls:")
        print("R : start recording")
        print("S : stop and save")
        print("C : cancel current recording")
        print("Q : quit")
        print(f"\nMinimum valid frames: {MIN_FRAMES}")
        print(f"Maximum frames: {MAX_FRAMES}")
        print(f"Minimum tracking ratio: {MIN_TRACKING_RATIO:.0%}")
        print("=" * 68)

        recording = False
        sequence = []
        total_recording_frames = 0
        tracked_frames = 0
        saved_sequences = 0

        while saved_sequences < args.sequences:
            success, frame = camera.read()
            if not success or frame is None:
                raise RuntimeError("Camera stopped returning frames.")

            frame = cv2.flip(frame, 1)
            landmarks = extract_raw_landmarks(frame, detector)

            if landmarks is not None:
                draw_landmarks(frame, landmarks)

            auto_stop = False
            if recording:
                total_recording_frames += 1
                if landmarks is not None:
                    normalized = normalize_landmarks(landmarks)
                    if normalized.shape == (63,) and np.isfinite(normalized).all():
                        sequence.append(normalized.astype(np.float32))
                        tracked_frames += 1
                if total_recording_frames >= MAX_FRAMES:
                    auto_stop = True

            tracking_ratio = (
                tracked_frames / total_recording_frames
                if total_recording_frames > 0 else 0.0
            )

            status = "RECORDING" if recording else "READY"
            status_color = (0, 0, 255) if recording else (0, 255, 255)

            cv2.putText(frame, f"{status} - LABEL: {args.label}", (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, status_color, 2)
            cv2.putText(frame, f"Saved: {saved_sequences}/{args.sequences}", (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            cv2.putText(frame,
                        f"Valid frames: {len(sequence)} | Tracking: {tracking_ratio:.0%}",
                        (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.60,
                        (255, 255, 255), 2)
            cv2.putText(frame, "R=start | S=save | C=cancel | Q=quit",
                        (20, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.58, (255, 255, 255), 2)

            cv2.imshow("Dynamic Sequence Collector V2", frame)
            key = cv2.waitKey(1) & 0xFF
            if auto_stop:
                print("Maximum recording length reached. Stopping automatically.")
                key = ord("s")

            if key == ord("q"):
                break

            if key == ord("r"):
                if recording:
                    print("Already recording. Press S to save or C to cancel.")
                    continue
                sequence = []
                total_recording_frames = 0
                tracked_frames = 0
                recording = True
                print(f"\nRecording {args.label} started...")

            elif key == ord("c"):
                if not recording:
                    print("No active recording to cancel.")
                    continue
                recording = False
                sequence = []
                total_recording_frames = 0
                tracked_frames = 0
                print("Recording cancelled.")

            elif key == ord("s"):
                if not recording:
                    print("Press R before pressing S.")
                    continue

                recording = False
                tracking_ratio = (
                    tracked_frames / total_recording_frames
                    if total_recording_frames > 0 else 0.0
                )

                if len(sequence) < MIN_FRAMES:
                    print(
                        f"REJECTED: too few valid frames. Got {len(sequence)}, "
                        f"need at least {MIN_FRAMES}."
                    )
                elif tracking_ratio < MIN_TRACKING_RATIO:
                    print(
                        "REJECTED: hand tracking was unstable. "
                        f"Tracking ratio = {tracking_ratio:.1%}"
                    )
                else:
                    output_file = save_sequence(
                        output_dir=output_dir,
                        person=args.person,
                        label=args.label,
                        sequence=sequence,
                    )
                    saved_sequences += 1
                    print(f"SAVED {saved_sequences}/{args.sequences}")
                    print(f"File: {output_file.name}")
                    print(f"Frames: {len(sequence)}")
                    print(f"Tracking ratio: {tracking_ratio:.1%}")

                sequence = []
                total_recording_frames = 0
                tracked_frames = 0

        print("\nCollection completed successfully.")

    except KeyboardInterrupt:
        print("\nInterrupted with Ctrl+C.")
    except Exception as error:
        print(f"\nERROR: {error}")
    finally:
        close_resources(camera, detector)


if __name__ == "__main__":
    main()
