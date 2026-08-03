/* Final frontend client
   - Static recognition runs continuously.
   - J/Z uses one-click countdown + automatic 3-second recording.
   - Keeps the working mirror/landmark behavior from the tested version.
*/

const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20],
  [0, 17],
];

const COUNTDOWN_SECONDS = 3;
const DYNAMIC_RECORDING_MS = 3000;
const RESULT_VISIBLE_MS = 2200;
const TARGET_INTERVAL_MS = 70;

const $ = (id) => document.getElementById(id);

const els = {
  video: $("video"),
  overlay: $("overlay"),
  hint: $("videoHint"),

  countdownOverlay: $("countdownOverlay"),
  countdownValue: $("countdownValue"),
  countdownLabel: $("countdownLabel"),

  resultBanner: $("resultBanner"),
  resultIcon: $("resultIcon"),
  resultTitle: $("resultTitle"),
  resultDetails: $("resultDetails"),

  letter: $("letter"),
  source: $("sourceTag"),

  conf: $("confBar"),
  confVal: $("confVal"),
  stab: $("stabBar"),
  stabVal: $("stabVal"),

  fps: $("fps"),
  hand: $("hand"),
  word: $("word"),
  modeText: $("modeText"),

  statusDot: $("statusDot"),
  statusText: $("statusText"),

  start: $("startBtn"),
  stop: $("stopBtn"),
  dynamic: $("dynamicBtn"),

  space: $("spaceBtn"),
  back: $("backBtn"),
  clear: $("clearBtn"),
  copy: $("copyBtn"),
  speak: $("speakBtn"),

  openGuide: $("openGuideBtn"),
  openGuidePreview: $("openGuidePreview"),
  guideModal: $("guideModal"),
  guideModalBackdrop: $("guideModalBackdrop"),
  closeGuide: $("closeGuideBtn"),

  progressWrap: $("dynamicProgressWrap"),
  progressBar: $("dynamicProgressBar"),
  progressText: $("dynamicProgressText"),
  progressTime: $("dynamicProgressTime"),

  url: $("backendUrl"),
  size: $("sendSize"),
};

const SESSION_ID = `web-${Math.random().toString(36).slice(2, 10)}`;

const sendCanvas = document.createElement("canvas");
const sendContext = sendCanvas.getContext("2d");
const overlayContext = els.overlay.getContext("2d");

let stream = null;
let ws = null;
let running = false;
let inFlight = false;
let dynamicBusy = false;

let lastSendTime = 0;
let fpsEma = 0;
let lastRenderedFrame = 0;

let countdownTimer = null;
let recordingTimer = null;
let progressAnimation = null;
let resultTimer = null;


// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------
function setStatus(state, text) {
  const className = {
    on: "dot-on",
    off: "dot-off",
    err: "dot-err",
  }[state] || "dot-off";

  els.statusDot.className = `dot ${className}`;
  els.statusText.textContent = text;
}

function setMode(text) {
  els.modeText.textContent = text;
}

function backendBase() {
  const configured = els.url.value.trim();

  if (!configured || configured.toLowerCase() === "auto") {
    return window.location.origin;
  }

  return configured.replace(/\/+$/, "");
}

function websocketUrl() {
  const base = backendBase();
  const socketBase = base
    .replace(/^https:/, "wss:")
    .replace(/^http:/, "ws:");

  return `${socketBase}/ws?session_id=${encodeURIComponent(SESSION_ID)}`;
}

function setTextActionState() {
  const hasText = (els.word.textContent || "").trim().length > 0;
  els.copy.disabled = !hasText;
  els.speak.disabled = !hasText;
}

function clearTimers() {
  if (countdownTimer) clearInterval(countdownTimer);
  if (recordingTimer) clearTimeout(recordingTimer);
  if (progressAnimation) cancelAnimationFrame(progressAnimation);
  if (resultTimer) clearTimeout(resultTimer);

  countdownTimer = null;
  recordingTimer = null;
  progressAnimation = null;
  resultTimer = null;
}

function hideResultBanner() {
  els.resultBanner.classList.add("hidden");
  els.resultBanner.classList.remove("result-success", "result-rejected");
}

function showResultBanner(letter, confidence) {
  if (resultTimer) clearTimeout(resultTimer);

  const accepted = letter === "J" || letter === "Z";
  const percent = Math.round((confidence || 0) * 100);

  els.resultBanner.classList.remove("hidden");
  els.resultBanner.classList.toggle("result-success", accepted);
  els.resultBanner.classList.toggle("result-rejected", !accepted);

  els.resultIcon.textContent = accepted ? "✓" : "!";
  els.resultTitle.textContent = accepted
    ? `Detected ${letter}`
    : "Motion not recognized";
  els.resultDetails.textContent = accepted
    ? `Confidence ${percent}%`
    : "Try again with a clearer motion";

  resultTimer = setTimeout(() => {
    hideResultBanner();
  }, RESULT_VISIBLE_MS);
}


// -----------------------------------------------------------------------------
// Camera
// -----------------------------------------------------------------------------
async function startCamera() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "user" },
      audio: false,
    });

    els.video.srcObject = stream;
    await els.video.play();

    els.hint.style.display = "none";
    resizeOverlay();

    running = true;
    inFlight = false;
    dynamicBusy = false;

    els.start.disabled = true;
    els.stop.disabled = false;
    els.dynamic.disabled = false;

    [els.space, els.back, els.clear].forEach((button) => {
      button.disabled = false;
    });

    els.dynamic.textContent = "Record J/Z";
    setMode("Connecting to backend");

    await fetch(
      `${backendBase()}/reset?session_id=${encodeURIComponent(SESSION_ID)}`,
      { method: "POST" },
    ).catch(() => {});

    connectWebSocket();
  } catch (error) {
    setStatus("err", "Camera blocked");
    setMode("Camera unavailable");
    alert(`Could not access the webcam: ${error.message}`);
  }
}

function stopCamera() {
  clearTimers();

  running = false;
  inFlight = false;
  dynamicBusy = false;

  if (ws) {
    ws.close();
    ws = null;
  }

  if (stream) {
    stream.getTracks().forEach((track) => track.stop());
    stream = null;
  }

  els.start.disabled = false;
  els.stop.disabled = true;
  els.dynamic.disabled = true;
  els.dynamic.textContent = "Record J/Z";

  [els.space, els.back, els.clear].forEach((button) => {
    button.disabled = true;
  });

  els.progressWrap.classList.add("hidden");
  els.countdownOverlay.classList.add("hidden");
  hideResultBanner();

  overlayContext.clearRect(
    0,
    0,
    els.overlay.width,
    els.overlay.height,
  );

  els.hint.style.display = "grid";
  els.letter.textContent = "–";
  els.source.textContent = "idle";
  els.source.className = "source-tag";
  els.hand.textContent = "no";

  setStatus("off", "Disconnected");
  setMode("Camera stopped");
}

function resizeOverlay() {
  els.overlay.width = els.video.clientWidth;
  els.overlay.height = els.video.clientHeight;
}

window.addEventListener("resize", () => {
  if (running) resizeOverlay();
});


// -----------------------------------------------------------------------------
// WebSocket
// -----------------------------------------------------------------------------
function connectWebSocket() {
  setStatus("off", "Connecting…");

  ws = new WebSocket(websocketUrl());

  ws.onopen = () => {
    setStatus("on", "Live");
    setMode("Static recognition");
    inFlight = false;
    pump();
  };

  ws.onmessage = (event) => {
    inFlight = false;

    let data;

    try {
      data = JSON.parse(event.data);
    } catch {
      setStatus("err", "Invalid response");
      return;
    }

    if (data.error) {
      console.error("Server error:", data.error);
      setStatus("err", "Server error");
      setMode("Backend error");
    } else {
      render(data);
    }

    const delay = Math.max(
      0,
      TARGET_INTERVAL_MS - (performance.now() - lastSendTime),
    );

    setTimeout(pump, delay);
  };

  ws.onerror = () => {
    setStatus("err", "Connection error");
    setMode("Backend connection failed");
  };

  ws.onclose = () => {
    if (running) {
      setStatus("err", "Disconnected");
      setMode("Backend disconnected");
    }
  };
}

function pump() {
  if (
    !running ||
    !ws ||
    ws.readyState !== WebSocket.OPEN ||
    inFlight
  ) {
    return;
  }

  const width = Math.max(
    160,
    Math.min(640, parseInt(els.size.value, 10) || 320),
  );
  const height = Math.round((width * 3) / 4);

  sendCanvas.width = width;
  sendCanvas.height = height;

  // Keep the tested mirrored input required by J/Z models.
  sendContext.save();
  sendContext.clearRect(0, 0, width, height);
  sendContext.translate(width, 0);
  sendContext.scale(-1, 1);
  sendContext.drawImage(els.video, 0, 0, width, height);
  sendContext.restore();

  const image = sendCanvas.toDataURL("image/jpeg", 0.6);

  lastSendTime = performance.now();
  inFlight = true;

  ws.send(JSON.stringify({ image }));
}


// -----------------------------------------------------------------------------
// One-click dynamic sequence
// -----------------------------------------------------------------------------
async function startGuidedDynamicRecording() {
  if (dynamicBusy) return;

  if (!ws || ws.readyState !== WebSocket.OPEN) {
    alert("The backend is not connected.");
    return;
  }

  dynamicBusy = true;
  hideResultBanner();

  els.dynamic.disabled = true;
  els.dynamic.classList.add("is-countdown");
  els.dynamic.textContent = "Get ready…";

  setStatus("on", "Preparing J/Z");
  setMode("J/Z countdown");

  els.countdownOverlay.classList.remove("hidden");
  els.countdownLabel.textContent = "Get ready";

  let remaining = COUNTDOWN_SECONDS;
  els.countdownValue.textContent = remaining;

  countdownTimer = setInterval(() => {
    remaining -= 1;

    if (remaining > 0) {
      els.countdownValue.textContent = remaining;
      return;
    }

    clearInterval(countdownTimer);
    countdownTimer = null;

    els.countdownValue.textContent = "GO";
    els.countdownLabel.textContent = "Draw J or Z now";

    setTimeout(() => {
      els.countdownOverlay.classList.add("hidden");
      beginDynamicRecording();
    }, 450);
  }, 1000);
}

function beginDynamicRecording() {
  if (!ws || ws.readyState !== WebSocket.OPEN) {
    finishDynamicUi();
    return;
  }

  els.dynamic.classList.remove("is-countdown");
  els.dynamic.classList.add("is-recording");
  els.dynamic.textContent = "Recording J/Z…";

  els.progressWrap.classList.remove("hidden");
  els.progressBar.style.width = "0%";
  els.progressText.textContent = "Draw J or Z now";

  setStatus("on", "Recording J/Z");
  setMode("Recording dynamic sign");

  ws.send(JSON.stringify({ type: "dynamic_start" }));

  const startedAt = performance.now();

  function updateProgress(now) {
    const elapsed = now - startedAt;
    const ratio = Math.min(1, elapsed / DYNAMIC_RECORDING_MS);
    const remainingSeconds = Math.max(
      0,
      (DYNAMIC_RECORDING_MS - elapsed) / 1000,
    );

    els.progressBar.style.width = `${ratio * 100}%`;
    els.progressTime.textContent = `${remainingSeconds.toFixed(1)}s`;

    if (ratio < 1) {
      progressAnimation = requestAnimationFrame(updateProgress);
    }
  }

  progressAnimation = requestAnimationFrame(updateProgress);

  recordingTimer = setTimeout(() => {
    stopDynamicRecording();
  }, DYNAMIC_RECORDING_MS);
}

function stopDynamicRecording() {
  if (!ws || ws.readyState !== WebSocket.OPEN) {
    finishDynamicUi();
    return;
  }

  if (progressAnimation) {
    cancelAnimationFrame(progressAnimation);
    progressAnimation = null;
  }

  els.progressBar.style.width = "100%";
  els.progressTime.textContent = "0.0s";
  els.progressText.textContent = "Processing motion…";

  els.dynamic.classList.remove("is-recording");
  els.dynamic.classList.add("is-processing");
  els.dynamic.textContent = "Processing…";

  setStatus("on", "Processing J/Z");
  setMode("Classifying dynamic sign");

  ws.send(JSON.stringify({ type: "dynamic_stop" }));
}

function finishDynamicUi() {
  dynamicBusy = false;

  els.dynamic.disabled = !running;
  els.dynamic.textContent = "Record J/Z";
  els.dynamic.classList.remove(
    "is-countdown",
    "is-recording",
    "is-processing",
  );

  els.progressWrap.classList.add("hidden");
  els.progressBar.style.width = "0%";

  setStatus("on", "Live");
  setMode("Static recognition");
}


// -----------------------------------------------------------------------------
// Rendering
// -----------------------------------------------------------------------------
function render(data) {
  updateFps();

  const detectedLetter =
    data.letter && data.letter !== "nothing"
      ? data.letter
      : "–";

  els.hand.textContent = data.hand_detected ? "yes" : "no";

  const confidence = Math.round((data.confidence || 0) * 100);
  const stability = Math.round((data.stability || 0) * 100);

  els.conf.style.width = `${confidence}%`;
  els.confVal.textContent = `${confidence}%`;

  els.stab.style.width = `${stability}%`;
  els.stabVal.textContent = `${stability}%`;

  els.word.textContent = data.word || "";
  setTextActionState();

  if (data.motion_state === "recording") {
    els.letter.textContent = "–";
    els.source.textContent = `recording J/Z (${data.dynamic?.frames_recorded || 0})`;
    els.source.className = "source-tag source-dynamic";
    setStatus("on", "Recording J/Z");
    setMode("Recording dynamic sign");
  } else if (data.source === "dynamic") {
    els.letter.textContent = detectedLetter;
    els.source.className = "source-tag source-dynamic";

    if (data.letter === "J" || data.letter === "Z") {
      els.source.textContent = `dynamic: ${data.letter}`;
      showResultBanner(data.letter, data.confidence);
    } else {
      els.source.textContent = "motion rejected";
      showResultBanner("nothing", data.confidence);
    }

    console.log("Dynamic result:", data.dynamic);
    finishDynamicUi();
  } else {
    if (!dynamicBusy) {
      els.letter.textContent = detectedLetter;
      els.source.textContent = data.hand_detected ? "static" : "no hand";
      els.source.className =
        "source-tag" +
        (data.hand_detected ? " source-static" : "");

      setStatus("on", "Live");
      setMode("Static recognition");
    }
  }

  drawLandmarks(data.landmarks);
}

function updateFps() {
  const now = performance.now();

  if (lastRenderedFrame) {
    const instantaneousFps = 1000 / (now - lastRenderedFrame);

    fpsEma = fpsEma
      ? fpsEma * 0.8 + instantaneousFps * 0.2
      : instantaneousFps;

    els.fps.textContent = fpsEma.toFixed(0);
  }

  lastRenderedFrame = now;
}

function drawLandmarks(points) {
  const width = els.overlay.width;
  const height = els.overlay.height;

  overlayContext.clearRect(0, 0, width, height);

  if (!points || !points.length) return;

  // Keep the working alignment from the user's tested version.
  const pointX = (point) => (1 - point[0]) * width;
  const pointY = (point) => point[1] * height;

  overlayContext.strokeStyle = "rgba(255,255,255,.72)";
  overlayContext.lineWidth = 2;

  for (const [startIndex, endIndex] of HAND_CONNECTIONS) {
    const start = points[startIndex];
    const end = points[endIndex];

    if (!start || !end) continue;

    overlayContext.beginPath();
    overlayContext.moveTo(pointX(start), pointY(start));
    overlayContext.lineTo(pointX(end), pointY(end));
    overlayContext.stroke();
  }

  overlayContext.fillStyle = "#12c98a";

  for (const point of points) {
    if (!point) continue;

    overlayContext.beginPath();
    overlayContext.arc(
      pointX(point),
      pointY(point),
      4,
      0,
      Math.PI * 2,
    );
    overlayContext.fill();
  }
}


// -----------------------------------------------------------------------------
// Text controls
// -----------------------------------------------------------------------------
async function wordEdit(action) {
  try {
    const response = await fetch(`${backendBase()}/word/edit`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        action,
        session_id: SESSION_ID,
      }),
    });

    const data = await response.json();
    els.word.textContent = data.word || "";
    setTextActionState();
  } catch (error) {
    console.error("Text edit failed:", error);
  }
}

async function copyText() {
  const text = els.word.textContent || "";
  if (!text.trim()) return;

  try {
    await navigator.clipboard.writeText(text);
    const previous = els.copy.textContent;
    els.copy.textContent = "Copied";
    setTimeout(() => {
      els.copy.textContent = previous;
    }, 1200);
  } catch {
    alert("Could not copy the text.");
  }
}

function speakText() {
  const text = els.word.textContent || "";
  if (!text.trim()) return;

  window.speechSynthesis.cancel();

  const utterance = new SpeechSynthesisUtterance(text);
  utterance.lang = "en-US";
  utterance.rate = 0.9;

  window.speechSynthesis.speak(utterance);
}



function openGuideModal() {
  els.guideModal.classList.remove("hidden");
  document.body.classList.add("modal-open");
  els.closeGuide.focus();
}

function closeGuideModal() {
  els.guideModal.classList.add("hidden");
  document.body.classList.remove("modal-open");
}

// -----------------------------------------------------------------------------
// Events
// -----------------------------------------------------------------------------
els.start.addEventListener("click", startCamera);
els.stop.addEventListener("click", stopCamera);
els.dynamic.addEventListener("click", startGuidedDynamicRecording);

els.space.addEventListener("click", () => wordEdit("space"));
els.back.addEventListener("click", () => wordEdit("backspace"));
els.clear.addEventListener("click", () => wordEdit("clear"));

els.copy.addEventListener("click", copyText);
els.speak.addEventListener("click", speakText);

els.openGuide.addEventListener("click", openGuideModal);
els.openGuidePreview.addEventListener("click", openGuideModal);
els.closeGuide.addEventListener("click", closeGuideModal);
els.guideModalBackdrop.addEventListener("click", closeGuideModal);

document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && !els.guideModal.classList.contains("hidden")) {
    closeGuideModal();
  }
});

setStatus("off", "Disconnected");
setMode("Camera stopped");
setTextActionState();
