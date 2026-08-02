"""FastAPI backend for the live ASL web application.

Modes
-----
- Static mode runs continuously.
- Manual dynamic mode starts when the frontend sends ``dynamic_start``.
- J/Z classification runs when the frontend sends ``dynamic_stop``.

Run
---
uvicorn backend.app:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import base64
import copy
import os
import sys
import threading
import time
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel


# -----------------------------------------------------------------------------
# Locate model bundle
# -----------------------------------------------------------------------------
def find_models_dir() -> str:
    """Find the folder containing asl_inference.py and the trained models."""
    env_path = os.environ.get("MODELS_DIR")

    candidates: list[str] = []

    if env_path:
        candidates.append(env_path)

    here = os.path.dirname(
        os.path.abspath(__file__)
    )

    candidates.extend(
        [
            os.path.join(
                here,
                "..",
                "Real-Time Sign Language Recognition",
            ),
            os.path.join(
                here,
                "..",
                "models",
            ),
            os.path.join(
                here,
                "models",
            ),
            os.path.join(
                here,
                "..",
            ),
        ]
    )

    for candidate in candidates:
        if not candidate:
            continue

        inference_file = os.path.join(
            candidate,
            "asl_inference.py",
        )

        if os.path.exists(inference_file):
            return os.path.abspath(candidate)

    raise RuntimeError(
        "Could not find the model bundle. "
        "Set MODELS_DIR to the folder containing "
        "asl_inference.py, StaticModel.keras, "
        "DynamicJModel_clean.keras, and "
        "DynamicZModel_trimmed.keras."
    )


MODELS_DIR = find_models_dir()

if MODELS_DIR not in sys.path:
    sys.path.insert(
        0,
        MODELS_DIR,
    )


from asl_inference import (  # noqa: E402
    SignLanguageTranslator,
    WordBuilder,
)
from asl_landmarks import (  # noqa: E402
    create_hand_detector,
    extract_raw_landmarks,
)


# -----------------------------------------------------------------------------
# Per-session runtime state
# -----------------------------------------------------------------------------
class Session:
    """Runtime state for one browser client."""

    def __init__(
        self,
        translator: SignLanguageTranslator,
        word_builder: WordBuilder,
    ):
        self.translator = translator
        self.word = word_builder
        self.lock = threading.Lock()
        self.last_used = time.time()


class Engine:
    """Loads heavy models once and clones lightweight runtime state per client."""

    def __init__(
        self,
        models_dir: str,
    ):
        self.base = SignLanguageTranslator(
            models_dir,
        )

        self.cfg = self.base.cfg

        self.sessions: dict[str, Session] = {}
        self.lock = threading.Lock()

    def clone_translator(
        self,
    ) -> SignLanguageTranslator:
        """Share loaded Keras models, but create separate runtime state."""
        translator = copy.copy(
            self.base,
        )

        # Keep the same detector mode used by the stable static version.
        translator.detector = create_hand_detector(
            static_image_mode=True,
        )

        translator.reset()

        return translator

    def session(
        self,
        session_id: str = "default",
    ) -> Session:
        with self.lock:
            session = self.sessions.get(
                session_id,
            )

            if session is None:
                translator = (
                    self.base
                    if session_id == "default"
                    else self.clone_translator()
                )

                word_builder = WordBuilder(
                    int(
                        self.cfg["stability_frames"]
                    )
                )

                session = Session(
                    translator,
                    word_builder,
                )

                self.sessions[
                    session_id
                ] = session

            session.last_used = time.time()

            return session

    def reset(
        self,
        session_id: str = "default",
    ) -> None:
        session = self.session(
            session_id,
        )

        with session.lock:
            session.translator.reset()
            session.word.reset()


ENGINE: Engine | None = None


# -----------------------------------------------------------------------------
# Application lifecycle
# -----------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(
    app: FastAPI,
):
    global ENGINE

    print(
        "[startup] loading static + manual J/Z models from: "
        f"{MODELS_DIR}"
    )

    ENGINE = Engine(
        MODELS_DIR,
    )

    print(
        "[startup] ready: static mode + manual J/Z recording"
    )

    yield

    print(
        "[shutdown] bye"
    )


app = FastAPI(
    title="Real-Time Sign Language Recognition API",
    description=(
        "Live ASL fingerspelling with 24 static letters "
        "and manually recorded J/Z sequences."
    ),
    version="2.1.0",
    lifespan=lifespan,
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get(
        "CORS_ORIGINS",
        "*",
    ).split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def require_engine() -> Engine:
    if ENGINE is None:
        raise HTTPException(
            status_code=503,
            detail="Models are still loading.",
        )

    return ENGINE


def decode_base64_image(
    data: str,
) -> np.ndarray:
    """Decode a base64 string or data URL into a BGR OpenCV frame."""
    if (
        "," in data
        and data.strip().lower().startswith("data:")
    ):
        data = data.split(
            ",",
            1,
        )[1]

    try:
        raw = base64.b64decode(
            data,
        )
    except Exception as error:
        raise HTTPException(
            status_code=400,
            detail="Invalid base64 image data.",
        ) from error

    array = np.frombuffer(
        raw,
        dtype=np.uint8,
    )

    frame = cv2.imdecode(
        array,
        cv2.IMREAD_COLOR,
    )

    if frame is None:
        raise HTTPException(
            status_code=400,
            detail="Could not decode image data.",
        )

    return frame


def apply_word_logic(
    session: Session,
    result: dict,
    coords: np.ndarray | None,
) -> None:
    """Update text using static stability or accepted J/Z results."""
    letter = result.get(
        "letter",
        "nothing",
    )

    source = result.get(
        "source",
        "none",
    )

    if source == "dynamic":
        # Dynamic letters are committed only by the dynamic_stop handler.
        result["word"] = session.word.text
        return

    if coords is None:
        session.word.hand_missing()

    elif letter == "nothing":
        # A low-confidence frame does not count as a full hand release.
        pass

    else:
        session.word.update_static(
            letter,
        )

    result["word"] = session.word.text


def run_on_frame(
    session: Session,
    frame_bgr: np.ndarray,
) -> dict:
    """Extract landmarks once, then run static or manual dynamic processing."""
    with session.lock:
        coords = extract_raw_landmarks(
            frame_bgr,
            session.translator.detector,
        )

        result = session.translator.process_landmarks(
            coords,
        )

        apply_word_logic(
            session,
            result,
            coords,
        )

        result["hand_detected"] = (
            coords is not None
        )

        result["landmarks"] = (
            np.round(
                coords[:, :2],
                4,
            ).tolist()
            if coords is not None
            else None
        )

        return result


def run_on_landmarks(
    session: Session,
    coords: np.ndarray | None,
) -> dict:
    """Run inference from already extracted 21x3 landmarks."""
    with session.lock:
        result = session.translator.process_landmarks(
            coords,
        )

        apply_word_logic(
            session,
            result,
            coords,
        )

        result["hand_detected"] = (
            coords is not None
        )

        result["landmarks"] = (
            np.round(
                coords[:, :2],
                4,
            ).tolist()
            if coords is not None
            else None
        )

        return result


# -----------------------------------------------------------------------------
# Request schemas
# -----------------------------------------------------------------------------
class PredictBody(
    BaseModel,
):
    image: str
    session_id: str = "default"


class LandmarksBody(
    BaseModel,
):
    landmarks: list
    session_id: str = "default"


class WordEditBody(
    BaseModel,
):
    action: str
    session_id: str = "default"


# -----------------------------------------------------------------------------
# HTTP routes
# -----------------------------------------------------------------------------
@app.get("/")
def root():
    return {
        "service": (
            "Real-Time Sign Language Recognition API"
        ),
        "version": "2.1.0",
        "mode": "static+manual-jz",
        "docs": "/docs",
        "endpoints": [
            "/health",
            "/config",
            "/predict",
            "/predict/file",
            "/predict/landmarks",
            "/word/edit",
            "/reset",
            "/ws",
        ],
    }


@app.get("/health")
def health():
    engine = ENGINE

    ready = engine is not None

    return {
        "status": (
            "ok"
            if ready
            else "loading"
        ),
        "models_dir": MODELS_DIR,
        "mode": "static+manual-jz",
        "static_classes": (
            engine.cfg["static_classes"]
            if ready
            else []
        ),
        "dynamic_classes": [
            "J",
            "Z",
        ],
    }


@app.get("/config")
def config():
    engine = require_engine()

    cfg = engine.cfg

    return {
        "mode": "static+manual-jz",
        "feature_dim": cfg["feature_dim"],
        "static_classes": cfg["static_classes"],
        "dynamic_classes": [
            "J",
            "Z",
        ],
        "smoothing_window": cfg["smoothing_window"],
        "stability_frames": cfg["stability_frames"],
        "min_confidence": cfg.get(
            "min_confidence",
            0.55,
        ),
    }


@app.post("/predict")
def predict(
    body: PredictBody,
):
    engine = require_engine()

    frame = decode_base64_image(
        body.image,
    )

    return run_on_frame(
        engine.session(
            body.session_id,
        ),
        frame,
    )


@app.post("/predict/file")
async def predict_file(
    file: UploadFile = File(...),
    session_id: str = Form(
        default="default",
    ),
):
    engine = require_engine()

    raw = np.frombuffer(
        await file.read(),
        dtype=np.uint8,
    )

    frame = cv2.imdecode(
        raw,
        cv2.IMREAD_COLOR,
    )

    if frame is None:
        raise HTTPException(
            status_code=400,
            detail="Could not decode uploaded image.",
        )

    return run_on_frame(
        engine.session(
            session_id,
        ),
        frame,
    )


@app.post("/predict/landmarks")
def predict_landmarks(
    body: LandmarksBody,
):
    engine = require_engine()

    array = np.asarray(
        body.landmarks,
        dtype=np.float32,
    )

    if array.size == 63:
        coords = array.reshape(
            21,
            3,
        )

    elif array.shape == (
        21,
        3,
    ):
        coords = array

    elif array.size == 0:
        coords = None

    else:
        raise HTTPException(
            status_code=400,
            detail=(
                "landmarks must be 21x3 "
                "or a flat list of 63 values."
            ),
        )

    return run_on_landmarks(
        engine.session(
            body.session_id,
        ),
        coords,
    )


@app.post("/word/edit")
def word_edit(
    body: WordEditBody,
):
    engine = require_engine()

    session = engine.session(
        body.session_id,
    )

    with session.lock:
        if body.action == "space":
            session.word.text += " "
            session.word.hand_missing()

        elif body.action == "backspace":
            session.word.text = (
                session.word.text[:-1]
            )
            session.word.hand_missing()

        elif body.action == "clear":
            session.word.reset()

        else:
            raise HTTPException(
                status_code=400,
                detail=(
                    "action must be "
                    "space|backspace|clear."
                ),
            )

        return {
            "word": session.word.text,
        }


@app.post("/reset")
def reset(
    session_id: str = "default",
):
    engine = require_engine()

    engine.reset(
        session_id,
    )

    return {
        "status": "reset",
        "session_id": session_id,
    }


# -----------------------------------------------------------------------------
# WebSocket
# -----------------------------------------------------------------------------
@app.websocket("/ws")
async def websocket_stream(
    websocket: WebSocket,
):
    await websocket.accept()

    engine = require_engine()

    session_id = websocket.query_params.get(
        "session_id",
        "default",
    )

    session = engine.session(
        session_id,
    )

    try:
        while True:
            message = await websocket.receive_json()

            message_type = message.get(
                "type",
            )

            # -------------------------------------------------------------
            # Reset current browser session
            # -------------------------------------------------------------
            if message_type == "reset":
                engine.reset(
                    session_id,
                )

                await websocket.send_json(
                    {
                        "word": "",
                        "letter": "nothing",
                        "raw_letter": "nothing",
                        "confidence": 0.0,
                        "stability": 0.0,
                        "source": "none",
                        "motion_state": "static",
                        "dynamic": {},
                        "hand_detected": False,
                        "landmarks": None,
                    }
                )

                continue

            # -------------------------------------------------------------
            # Start manual J/Z recording
            # -------------------------------------------------------------
            if message_type == "dynamic_start":
                with session.lock:
                    result = (
                        session.translator
                        .start_dynamic_recording()
                    )

                    result["word"] = (
                        session.word.text
                    )

                    result["hand_detected"] = False
                    result["landmarks"] = None

                print(
                    "[dynamic] start request | "
                    f"session={session_id}"
                )

                await websocket.send_json(
                    result,
                )

                continue

            # -------------------------------------------------------------
            # Stop manual J/Z recording and classify sequence
            # -------------------------------------------------------------
            if message_type == "dynamic_stop":
                with session.lock:
                    result = (
                        session.translator
                        .finish_dynamic_recording()
                    )

                    letter = result.get(
                        "letter",
                        "nothing",
                    )

                    if letter in {
                        "J",
                        "Z",
                    }:
                        session.word.commit_dynamic(
                            letter,
                        )

                    result["word"] = (
                        session.word.text
                    )

                    result["hand_detected"] = False
                    result["landmarks"] = None

                print(
                    "[dynamic] stop request | "
                    f"session={session_id} | "
                    f"result={result.get('letter')} | "
                    f"debug={result.get('dynamic', {})}"
                )

                await websocket.send_json(
                    result,
                )

                continue

            # -------------------------------------------------------------
            # Normal camera frame
            # -------------------------------------------------------------
            image = message.get(
                "image",
            )

            if not image:
                await websocket.send_json(
                    {
                        "error": (
                            "Missing image field."
                        )
                    }
                )

                continue

            frame = decode_base64_image(
                image,
            )

            result = run_on_frame(
                session,
                frame,
            )

            await websocket.send_json(
                result,
            )

    except WebSocketDisconnect:
        print(
            "[websocket] disconnected | "
            f"session={session_id}"
        )

    except Exception as error:
        print(
            "[websocket] error | "
            f"session={session_id} | "
            f"{type(error).__name__}: {error}"
        )

        try:
            await websocket.send_json(
                {
                    "error": str(error),
                }
            )
        except Exception:
            pass


# -----------------------------------------------------------------------------
# Global error handler
# -----------------------------------------------------------------------------
@app.exception_handler(
    Exception,
)
async def unhandled_exception(
    request,
    error,
):
    print(
        "[api] unhandled error | "
        f"{type(error).__name__}: {error}"
    )

    return JSONResponse(
        status_code=500,
        content={
            "error": str(error),
        },
    )