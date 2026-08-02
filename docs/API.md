# API Documentation — Real-Time Sign Language Recognition

**Base URL (dev):** `http://localhost:8000`
**Interactive docs:** `http://localhost:8000/docs` (Swagger UI, auto-generated) · `http://localhost:8000/redoc`
**Content type:** `application/json` unless noted.

The API classifies American Sign Language (ASL) fingerspelling from a webcam
frame using a single **static-pose MLP** over the **24 static letters**
(`A–I, K–Y`). The motion letters `J`/`Z` are out of scope.

State (temporal smoothing window and the current word) is kept **per
`session_id`**. Send a unique `session_id` per client; omit it to use
`"default"`.

---

## Prediction result object

Every prediction endpoint returns this object:

| Field | Type | Description |
|---|---|---|
| `letter` | string | Smoothed prediction (majority vote over the last *N* frames). `"nothing"` when no hand is present or the frame is below the confidence gate. |
| `raw_letter` | string | The current frame's raw prediction (before smoothing). |
| `confidence` | float | Model confidence for `raw_letter`, `0.0–1.0`. |
| `stability` | float | Fraction of the smoothing window that agrees with `letter`, `0.0–1.0`. |
| `source` | string | `"static"` (a letter was classified) or `"none"` (no hand). |
| `word` | string | The current translated text for this session. |
| `hand_detected` | bool | Whether a hand was found in the frame. |
| `landmarks` | array\|null | `21 × [x, y]` normalized landmark coordinates (`0–1`) for overlay, or `null`. |

```json
{
  "letter": "A", "raw_letter": "A", "confidence": 0.97, "stability": 0.9,
  "source": "static", "word": "CAB",
  "hand_detected": true,
  "landmarks": [[0.51,0.62],[0.55,0.58], "... 21 points ..."]
}
```

---

## Endpoints

### `GET /` — service metadata
Returns the service name, version, and the list of endpoints.

### `GET /health` — liveness & model status
```json
{ "status": "ok", "models_dir": "...", "mode": "static-only", "static_classes": ["A","B", "..."] }
```
`status` is `"ok"` once the model is loaded, else `"loading"`.

### `GET /config` — model configuration
Returns the values the client should mirror.
```json
{
  "mode": "static-only",
  "feature_dim": 63,
  "static_classes": ["A", "..."],
  "smoothing_window": 10, "stability_frames": 12,
  "min_confidence": 0.55
}
```

---

### `POST /predict` — classify one frame (JSON)
JSON body with a base64 image.

| Field | Type | Required | Description |
|---|---|---|---|
| `image` | string | yes | Base64 or `data:` URL of a JPEG/PNG frame. |
| `session_id` | string | no | Client session (default `"default"`). |

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"image":"data:image/jpeg;base64,/9j/4AAQ...","session_id":"demo"}'
```

**Returns:** a [prediction result object](#prediction-result-object).

### `POST /predict/file` — classify one frame (file upload)
`multipart/form-data` upload — handy for testing with `curl` or Postman.

| Field | Type | Required | Description |
|---|---|---|---|
| `file` | file | yes | A JPEG/PNG image. |
| `session_id` | string (form) | no | Client session. |

```bash
curl -X POST http://localhost:8000/predict/file \
  -F "file=@frame.jpg" -F "session_id=demo"
```

**Returns:** a [prediction result object](#prediction-result-object).

---

### `POST /predict/landmarks` — classify from landmarks
For clients that run MediaPipe in the browser and send landmarks directly (lowest
latency, no image upload).

| Field | Type | Required | Description |
|---|---|---|---|
| `landmarks` | array | yes | `21 × 3` nested, or a flat array of `63`. Empty `[]` = no hand. |
| `session_id` | string | no | Client session. |

```bash
curl -X POST http://localhost:8000/predict/landmarks \
  -H "Content-Type: application/json" \
  -d '{"landmarks": [[0.5,0.6,0.0], "... 21 points ..."], "session_id":"demo"}'
```

**Returns:** a prediction result object (without the `landmarks` echo).

---

### `POST /word/edit` — edit the current word
The alphabet model does not emit space/delete gestures, so the UI drives text
editing directly.

| Field | Type | Description |
|---|---|---|
| `action` | string | `"space"`, `"backspace"`, or `"clear"`. |
| `session_id` | string | Client session. |

```bash
curl -X POST http://localhost:8000/word/edit \
  -H "Content-Type: application/json" \
  -d '{"action":"space","session_id":"demo"}'
```
**Returns:** `{ "word": "CAB " }`

---

### `POST /reset` — clear session state
Clears the landmark buffer, smoothing window, and word for a session.
```bash
curl -X POST "http://localhost:8000/reset?session_id=demo"
```
**Returns:** `{ "status": "reset", "session_id": "demo" }`

---

### `WS /ws` — streaming predictions
WebSocket for low-latency real-time use. Connect to
`ws://localhost:8000/ws?session_id=<id>`.

- **Send** (per frame): `{ "image": "data:image/jpeg;base64,..." }`
- **Send** (reset): `{ "type": "reset" }`
- **Receive:** a prediction result object per frame.

Use a **request→response** cadence (send the next frame only after receiving the
previous result) to self-pace to the server's throughput.

```javascript
const ws = new WebSocket("ws://localhost:8000/ws?session_id=demo");
ws.onopen = () => ws.send(JSON.stringify({ image: dataUrl }));
ws.onmessage = (e) => {
  const r = JSON.parse(e.data);
  console.log(r.letter, r.confidence, r.word);
  ws.send(JSON.stringify({ image: nextFrame() }));
};
```

---

## Errors

| Status | Meaning |
|---|---|
| `400` | Bad request — undecodable image, or `landmarks` not `21×3`/`63`. |
| `500` | Server error — body is `{ "error": "<message>" }`. |

Over WebSocket, errors are delivered in-band as `{ "error": "<message>" }`.

---

## Notes for integrators

- **One session per client.** Sessions are created lazily on first use and share
  the loaded models, so they are cheap. Call `/reset` when a user restarts.
- **Frame rate.** Downscale frames (e.g. width 320 px, JPEG q≈0.6) before sending;
  the models run on landmarks, so full resolution buys almost nothing.
- **Confidence gate.** Frames below `min_confidence` are returned as
  `letter: "nothing"` so a low-confidence guess during a hand transition never
  flashes a wrong letter. Tune `min_confidence` in `config.json`.
- **Out of scope.** The motion letters `J` / `Z` are not modelled in this
  static-only version; every prediction is a single-frame static pose.
- **CORS.** All origins are allowed by default; set the `CORS_ORIGINS` env var
  (comma-separated) to restrict in production.
