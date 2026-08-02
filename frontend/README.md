# Frontend — Real-Time Sign Language Recognition

A zero-build, single-page web client. It captures webcam frames, streams them to
the backend over WebSocket, and renders the predicted letter, confidence, the
hand-landmark skeleton, and the growing translated text.

## Run

The backend must be running first (see [`../backend`](../backend)). Then serve
these static files from any web server — do **not** open `index.html` with
`file://`, because the browser blocks the camera and network calls there.

```bash
# from the frontend/ folder
python -m http.server 5500
# then open http://localhost:5500
```

Click **Start camera**, allow webcam access, and start signing.

## Configure

Open **Settings** in the UI to change:

- **Backend URL** — defaults to `http://localhost:8000`.
- **Send size** — the width (px) frames are downscaled to before upload. Smaller
  = faster / less bandwidth; larger = slightly better detection.

## How it works

1. `getUserMedia` opens the camera; frames are drawn to an offscreen canvas and
   downscaled to a small JPEG.
2. Frames stream over `ws://<backend>/ws` one at a time (request → response), so
   the send rate self-paces to the server's speed.
3. Each response carries `{ letter, confidence, source, stability, word,
   landmarks }`; the skeleton is drawn from `landmarks` on an overlay canvas.
4. **Space / Backspace / Clear** call `POST /word/edit` (the alphabet model has
   no space/delete gesture, so text editing is driven from the UI).

## Notes

- The video and overlay are mirrored (selfie view); landmarks are drawn in raw
  coordinates so the CSS mirror aligns them with the hand.
- The model recognizes the **24 static ASL letters** (`A–I, K–Y`). The motion
  letters `J` and `Z` are out of scope.
- Low-confidence frames (during hand transitions) are shown as "no letter", so a
  wrong letter never flashes while you move between poses.
