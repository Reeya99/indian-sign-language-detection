"use strict";

const CFG = window.ISL_CONFIG || { threshold: 0.6, intervalMs: 120, smoothN: 5 };

// MediaPipe hand connections (pairs of landmark indices) for drawing the skeleton.
const HAND_CONNECTIONS = [
  [0,1],[1,2],[2,3],[3,4],            // thumb
  [0,5],[5,6],[6,7],[7,8],            // index
  [5,9],[9,10],[10,11],[11,12],       // middle
  [9,13],[13,14],[14,15],[15,16],     // ring
  [13,17],[17,18],[18,19],[19,20],    // pinky
  [0,17],                             // palm base
];

const video = document.getElementById("video");
const overlay = document.getElementById("overlay");
const octx = overlay.getContext("2d");
const camOff = document.getElementById("camOff");
const toggleCamBtn = document.getElementById("toggleCam");
const toggleTop3 = document.getElementById("toggleTop3");
const top3Card = document.getElementById("top3Card");
const top3List = document.getElementById("top3List");
const statusLine = document.getElementById("statusLine");
const predLabel = document.getElementById("predLabel");
const confBar = document.getElementById("confBar");
const confText = document.getElementById("confText");

// Hidden canvas used to grab frames to send to the server.
const grabCanvas = document.createElement("canvas");
const gctx = grabCanvas.getContext("2d");

let stream = null;
let running = false;
let loopTimer = null;
let inFlight = false;
const labelBuffer = [];   // last N raw top-1 labels for smoothing

// ---- Camera control ------------------------------------------------------
async function startCamera() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 640 }, height: { ideal: 480 } },
      audio: false,
    });
  } catch (err) {
    setStatus("nohand", "Camera permission denied or unavailable.");
    return;
  }
  video.srcObject = stream;
  await video.play();
  // Size canvases to the actual video resolution.
  overlay.width = video.videoWidth;
  overlay.height = video.videoHeight;
  grabCanvas.width = video.videoWidth;
  grabCanvas.height = video.videoHeight;

  camOff.classList.add("hidden");
  running = true;
  toggleCamBtn.textContent = "Stop camera";
  toggleCamBtn.classList.add("stop");
  setStatus("detecting", "Starting…");
  loopTimer = setInterval(tick, CFG.intervalMs);
}

function stopCamera() {
  running = false;
  if (loopTimer) clearInterval(loopTimer);
  loopTimer = null;
  if (stream) stream.getTracks().forEach((t) => t.stop());
  stream = null;
  video.srcObject = null;
  octx.clearRect(0, 0, overlay.width, overlay.height);
  camOff.classList.remove("hidden");
  toggleCamBtn.textContent = "Start camera";
  toggleCamBtn.classList.remove("stop");
  labelBuffer.length = 0;
  resetPrediction();
  setStatus("idle", "Camera is off");
}

toggleCamBtn.addEventListener("click", () => {
  if (running) stopCamera(); else startCamera();
});
toggleTop3.addEventListener("change", () => {
  top3Card.classList.toggle("hidden", !toggleTop3.checked);
});

// ---- Frame loop ----------------------------------------------------------
async function tick() {
  if (!running || inFlight || video.readyState < 2) return;
  inFlight = true;
  try {
    gctx.drawImage(video, 0, 0, grabCanvas.width, grabCanvas.height);
    const dataUrl = grabCanvas.toDataURL("image/jpeg", 0.6);
    const resp = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image: dataUrl }),
    });
    if (!resp.ok) throw new Error("server " + resp.status);
    const data = await resp.json();
    handleResult(data);
  } catch (err) {
    setStatus("nohand", "Connection issue… retrying");
  } finally {
    inFlight = false;
  }
}

// ---- Smoothing: most common label over the last N frames -----------------
function smoothedLabel(current) {
  labelBuffer.push(current);
  while (labelBuffer.length > CFG.smoothN) labelBuffer.shift();
  const counts = {};
  let best = current, bestN = 0;
  for (const l of labelBuffer) {
    counts[l] = (counts[l] || 0) + 1;
    if (counts[l] > bestN) { bestN = counts[l]; best = l; }
  }
  return best;
}

// ---- Handle a /predict response ------------------------------------------
function handleResult(data) {
  if (!data || data.hand === false) {
    labelBuffer.length = 0;
    octx.clearRect(0, 0, overlay.width, overlay.height);
    resetPrediction();
    setStatus("nohand", "No hand detected");
    return;
  }

  drawLandmarks(data.landmarks);
  updateTop3(data.top3);

  const conf = data.confidence || 0;
  const pct = Math.round(conf * 100);
  confBar.style.width = pct + "%";
  confText.textContent = pct + "%";

  const handsTxt = data.num_hands === 2 ? "2 hands" : "1 hand";

  // Hand-count gate: the best guess needs a different number of hands than are
  // visible -> decline (neither right nor wrong). Don't smooth/commit it.
  if (data.gated) {
    labelBuffer.length = 0;
    predLabel.textContent = "—";
    confBar.style.background = "var(--accent)";
    const need = data.required_hands === 2
      ? "this looks like a two-handed sign — show both hands"
      : "this looks like a one-handed sign — show just one hand";
    setStatus("detecting", `Detecting…  (${handsTxt}; ${need})`);
    return;
  }

  const label = smoothedLabel(data.label);
  if (conf >= CFG.threshold) {
    predLabel.textContent = label;
    confBar.style.background = "var(--good)";
    setStatus("good", `Recognized: ${label}  (${handsTxt})`);
  } else {
    predLabel.textContent = "—";
    confBar.style.background = "var(--accent)";
    setStatus("detecting", `Detecting…  (${handsTxt})`);
  }
}

function resetPrediction() {
  predLabel.textContent = "—";
  confBar.style.width = "0%";
  confText.textContent = "0%";
  top3List.innerHTML = "";
}

function updateTop3(top3) {
  if (!top3) return;
  top3List.innerHTML = "";
  top3.forEach((p, i) => {
    const pct = Math.round(p.confidence * 100);
    const li = document.createElement("li");
    li.innerHTML =
      `<span class="top3-rank">${i + 1}</span>` +
      `<span class="top3-name">${p.label}</span>` +
      `<span class="top3-track"><span class="top3-fill" style="width:${pct}%"></span></span>` +
      `<span class="top3-pct">${pct}%</span>`;
    top3List.appendChild(li);
  });
}

// ---- Draw landmarks onto the (non-mirrored) overlay ----------------------
// `hands` is a list of hands, each a list of 21 [x, y] normalized points.
// (A single flat hand is also accepted for backward compatibility.)
const HAND_COLORS = ["rgba(56, 189, 248, 0.9)", "rgba(250, 204, 21, 0.9)"];

function drawLandmarks(hands) {
  octx.clearRect(0, 0, overlay.width, overlay.height);
  if (!hands || !hands.length) return;
  // Normalize to a list-of-hands shape.
  if (typeof hands[0][0] === "number") hands = [hands];
  const W = overlay.width, H = overlay.height;

  hands.forEach((landmarks, hi) => {
    octx.lineWidth = 3;
    octx.strokeStyle = HAND_COLORS[hi % HAND_COLORS.length];
    for (const [a, b] of HAND_CONNECTIONS) {
      octx.beginPath();
      octx.moveTo(landmarks[a][0] * W, landmarks[a][1] * H);
      octx.lineTo(landmarks[b][0] * W, landmarks[b][1] * H);
      octx.stroke();
    }
    octx.fillStyle = "#f8fafc";
    for (const [x, y] of landmarks) {
      octx.beginPath();
      octx.arc(x * W, y * H, 4, 0, Math.PI * 2);
      octx.fill();
    }
  });
}

// ---- Status helper -------------------------------------------------------
function setStatus(kind, text) {
  statusLine.className = "status status-" + kind;
  statusLine.textContent = text;
}

window.addEventListener("beforeunload", () => { if (running) stopCamera(); });
