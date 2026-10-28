"""Milestone 3 - Flask backend for live ISL recognition.

Loads model.h5 + classes.json once at startup. The browser posts webcam
frames (base64 JPEG) to /predict about every 120 ms; the server decodes the
frame, applies the SAME mirror + preprocessing as keypoint generation, runs
MediaPipe + the model, and returns {hand, label, confidence, landmarks, top3}.

Mirror consistency (important):
  The browser sends the RAW (un-mirrored) frame. The flip happens HERE, once,
  via cv2.flip(frame, 1) gated by config.MIRROR_FRAMES, and MediaPipe + feature
  extraction both run on that same flipped frame - identical to
  dataset_keypoint_generation.py. The displayed video is mirrored in CSS, so
  the returned landmark coordinates (which are in flipped-frame space) line up
  with what the user sees.

Run (venv active):
    python app.py
then open http://127.0.0.1:5050  (change with env var ISL_PORT)
"""
import base64
import csv
import json
import os
import threading

import cv2
import numpy as np
import mediapipe as mp
from flask import Flask, jsonify, render_template, request

import config
from preprocessing import (
    CSV_COLUMNS,
    NUM_FEATURES,
    landmarks_to_xy,
    preprocess_hands,
)

# ---- Load model + classes once ------------------------------------------
if not (os.path.isfile(config.MODEL_PATH) and os.path.isfile(config.CLASSES_JSON)):
    raise SystemExit(
        "model.h5 / classes.json not found. Run train_model.py first "
        "(Milestones 1-2)."
    )

import tensorflow as tf  # imported after the file check to fail fast

model = tf.keras.models.load_model(config.MODEL_PATH, compile=False)
with open(config.CLASSES_JSON) as f:
    CLASSES = json.load(f)
print(f"Loaded model and {len(CLASSES)} classes: {CLASSES}")

# Required hand count per class (1 or 2). Missing file -> assume 1 for all.
try:
    with open(config.HAND_COUNTS_JSON) as f:
        _hc = json.load(f)
    REQUIRED_HANDS = np.array([int(_hc.get(c, 1)) for c in CLASSES], dtype=np.int32)
except FileNotFoundError:
    REQUIRED_HANDS = np.ones(len(CLASSES), dtype=np.int32)
print("Two-handed classes:",
      [c for c, r in zip(CLASSES, REQUIRED_HANDS) if r == 2])

# Warm up so the first real request isn't slow.
model(np.zeros((1, NUM_FEATURES), dtype=np.float32), training=False)

# ---- MediaPipe Hands (streaming mode) -----------------------------------
# Not thread-safe, and TF inference shares this lock too, so concurrent
# requests from the single client are serialized.
hands = mp.solutions.hands.Hands(
    static_image_mode=False,       # streaming/tracking mode for live video
    max_num_hands=2,               # two-handed ISL signs need both hands
    min_detection_confidence=0.5,  # lower so the second hand is found too
    min_tracking_confidence=0.5,
)
_lock = threading.Lock()        # guards MediaPipe + TF inference
_file_lock = threading.Lock()   # guards appends to my_keypoints.csv

app = Flask(__name__)


def decode_frame(data_url):
    """Decode a 'data:image/jpeg;base64,...' string to a BGR image."""
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]
    raw = base64.b64decode(data_url)
    arr = np.frombuffer(raw, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


@app.route("/")
def index():
    return render_template("index.html", classes=CLASSES)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    image = data.get("image")
    if not image:
        return jsonify({"error": "no image"}), 400

    frame = decode_frame(image)
    if frame is None:
        return jsonify({"error": "could not decode frame"}), 400

    # THE flip: once, here, matching keypoint generation exactly.
    if config.MIRROR_FRAMES:
        frame = cv2.flip(frame, 1)

    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    with _lock:
        result = hands.process(rgb)

        if not result.multi_hand_landmarks:
            return jsonify({"hand": False})

        hand_list = result.multi_hand_landmarks[:2]
        hands_xy = [landmarks_to_xy(hl, w, h) for hl in hand_list]  # pixel
        features = preprocess_hands(hands_xy)          # 84, for the model
        # Normalized 0..1 coords per hand for drawing on the browser overlay.
        draw = [[[round(lm.x, 5), round(lm.y, 5)] for lm in hl.landmark]
                for hl in hand_list]
        probs = model(
            np.expand_dims(features, 0), training=False
        ).numpy()[0]

    # Top-3 for the debug panel (model's natural, ungated guesses).
    top_idx = np.argsort(probs)[::-1][:3]
    top3 = [
        {"label": CLASSES[i], "confidence": round(float(probs[i]), 4)}
        for i in top_idx
    ]
    best_i = int(top_idx[0])

    # Hand-count gate (reject-on-mismatch): if the best guess needs a different
    # number of hands than are visible, we DON'T show a letter - we decline
    # (neither right nor wrong). We never force a different class. So a
    # two-handed sign simply won't register with one hand, and vice-versa.
    n_hands = len(draw)
    required = int(REQUIRED_HANDS[best_i])
    gated = (required != n_hands)

    return jsonify({
        "hand": True,
        "num_hands": n_hands,
        "label": CLASSES[best_i],
        "confidence": round(float(probs[best_i]), 4),
        "required_hands": required,
        "gated": gated,
        "landmarks": draw,
        "top3": top3,
    })


@app.route("/record", methods=["POST"])
def record():
    """Collect-samples mode: extract one row and append it to my_keypoints.csv.

    Uses the exact same flip + preprocessing as /predict and keypoint
    generation, so recorded samples are directly compatible.
    """
    data = request.get_json(silent=True) or {}
    image = data.get("image")
    label = data.get("label")
    if not image or not label:
        return jsonify({"error": "need image and label"}), 400
    if label not in CLASSES:
        return jsonify({"error": f"unknown label {label}"}), 400

    frame = decode_frame(image)
    if frame is None:
        return jsonify({"error": "could not decode frame"}), 400

    if config.MIRROR_FRAMES:
        frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    with _lock:
        result = hands.process(rgb)
        if not result.multi_hand_landmarks:
            return jsonify({"saved": False, "hand": False})
        hand_list = result.multi_hand_landmarks[:2]
        hands_xy = [landmarks_to_xy(hl, w, h) for hl in hand_list]  # pixel
        features = preprocess_hands(hands_xy)
        draw = [[[round(lm.x, 5), round(lm.y, 5)] for lm in hl.landmark]
                for hl in hand_list]

    with _file_lock:
        new_file = not os.path.isfile(config.MY_KEYPOINT_CSV)
        with open(config.MY_KEYPOINT_CSV, "a", newline="") as f:
            writer = csv.writer(f)
            if new_file:
                writer.writerow(CSV_COLUMNS)
            writer.writerow([label] + features.tolist())

    return jsonify({
        "saved": True,
        "hand": True,
        "landmarks": draw,
    })


if __name__ == "__main__":
    # Port 5000 is often taken on Windows (or blocked), so default to 5050.
    # Override with:  set ISL_PORT=8000   (PowerShell: $env:ISL_PORT="8000")
    port = int(os.environ.get("ISL_PORT", "5050"))
    print(f"\n  Open:  http://127.0.0.1:{port}   (or http://localhost:{port})\n")
    # threaded=True by default in Flask 3; the lock keeps MediaPipe/TF safe.
    app.run(host="127.0.0.1", port=port, debug=False)
