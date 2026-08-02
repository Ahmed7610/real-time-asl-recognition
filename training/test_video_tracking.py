import sys
import time
from pathlib import Path

import cv2


PROJECT_DIR = Path(__file__).resolve().parents[1]

MODEL_DIR = (
    PROJECT_DIR
    / "Real-Time Sign Language Recognition"
)

sys.path.insert(
    0,
    str(MODEL_DIR),
)

from asl_landmarks import (  # noqa: E402
    create_hand_detector,
    extract_raw_landmarks,
)


CONNECTIONS = [
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


def draw_landmarks(frame, landmarks):
    height, width = frame.shape[:2]

    points = []

    for landmark in landmarks:
        point = (
            int(landmark[0] * width),
            int(landmark[1] * height),
        )
        points.append(point)

    for start, end in CONNECTIONS:
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


def close_resources(camera, detector):
    print("Closing camera resources...")

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


def main():
    camera = None
    detector = None

    try:
        camera = cv2.VideoCapture(
            0,
            cv2.CAP_V4L2,
        )

        if not camera.isOpened():
            raise RuntimeError(
                "Could not open camera 0."
            )

        configure_camera(camera)

        detector = create_hand_detector(
            static_image_mode=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        start_time = time.monotonic()
        previous_timestamp = -1

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
                (
                    time.monotonic()
                    - start_time
                )
                * 1000
            )

            if timestamp_ms <= previous_timestamp:
                timestamp_ms = (
                    previous_timestamp + 1
                )

            previous_timestamp = timestamp_ms

            landmarks = extract_raw_landmarks(
                frame,
                detector,
                timestamp_ms=timestamp_ms,
            )

            if landmarks is not None:
                draw_landmarks(
                    frame,
                    landmarks,
                )

                cv2.putText(
                    frame,
                    "Tracking: HAND DETECTED",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (0, 255, 0),
                    2,
                )

            else:
                cv2.putText(
                    frame,
                    "Tracking: NO HAND",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (0, 0, 255),
                    2,
                )

            cv2.putText(
                frame,
                "Press Q to exit",
                (20, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
            )

            cv2.imshow(
                "MediaPipe Video Tracking Test",
                frame,
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

    except KeyboardInterrupt:
        print("Interrupted.")

    except Exception as error:
        print(f"ERROR: {error}")

    finally:
        close_resources(
            camera,
            detector,
        )


if __name__ == "__main__":
    main()
