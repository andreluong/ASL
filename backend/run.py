from __future__ import annotations

import base64
import os
import re
from collections import deque
from typing import Tuple

import cv2
import mediapipe as mp
import numpy as np
import pickle
import torch
from flask import Flask, jsonify, request
from flask_cors import CORS
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from model import ASLSignLSTM


app = Flask(__name__)
CORS(app)


LABELS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + list("0123456789")
LABEL_TO_INDEX = {label: idx for idx, label in enumerate(LABELS)}

# MediaPipe Tasks: HandLandmarker (Tasks API replaces legacy mp.solutions.hands)
BaseOptions = mp_python.BaseOptions
HandLandmarker = mp_vision.HandLandmarker
HandLandmarkerOptions = mp_vision.HandLandmarkerOptions
VisionRunningMode = mp_vision.RunningMode

_default_model_path = os.path.join(
    os.path.dirname(__file__), "hand_landmarker.task"
)
HAND_LANDMARKER_MODEL_PATH = os.getenv(
    "HAND_LANDMARKER_MODEL_PATH", _default_model_path
)

HAND_LANDMARKER: HandLandmarker | None = None
if os.path.exists(HAND_LANDMARKER_MODEL_PATH):
    try:
        _options = HandLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path=HAND_LANDMARKER_MODEL_PATH
            ),
            running_mode=VisionRunningMode.IMAGE,
            num_hands=1,
        )
        HAND_LANDMARKER = HandLandmarker.create_from_options(_options)
    except Exception:
        HAND_LANDMARKER = None

# Optional: load a custom classifier that takes 63-dim landmark vector (21 landmarks * 3 coords)
LANDMARK_CLASSIFIER = None
_landmark_model_path = os.getenv("ASL_LANDMARK_MODEL_PATH")
if _landmark_model_path and os.path.exists(_landmark_model_path):
    try:
        import joblib
        LANDMARK_CLASSIFIER = joblib.load(_landmark_model_path)
    except Exception:
        pass

# ── LSTM model ────────────────────────────────────────────────────────────────

LSTM_MODEL = None
LSTM_LABELS = None
SEQUENCE_LENGTH = 30  # number of frames per prediction

_lstm_model_path = os.path.join(os.path.dirname(__file__), "asl_lstm.pt")
_label_map_path = os.path.join(os.path.dirname(__file__), "label_map.pkl")

if os.path.exists(_lstm_model_path) and os.path.exists(_label_map_path):
    try:
        with open(_label_map_path, "rb") as f:
            label_map = pickle.load(f)
        LSTM_LABELS = label_map["idx_to_label"]
        LSTM_MODEL = ASLSignLSTM(num_classes=len(LSTM_LABELS))
        LSTM_MODEL.load_state_dict(torch.load(_lstm_model_path, map_location="cpu"))
        LSTM_MODEL.eval()
        print(f"LSTM model loaded with {len(LSTM_LABELS)} classes")
    except Exception as e:
        print(f"Failed to load LSTM model: {e}")

# Per-session frame buffer — keyed by session_id from the frontend
# Each value is a deque of landmark vectors (np.ndarray of shape 63)
frame_buffers: dict[str, deque] = {}


# ── Shared helpers ────────────────────────────────────────────────────────────

def get_hand_landmarks_vector(img_bgr: np.ndarray) -> np.ndarray | None:
    """
    Run MediaPipe Hands on the image and return a 63-dim vector (21 landmarks * x,y,z)
    for the first detected hand, or None if no hand is detected.
    """
    if HAND_LANDMARKER is None:
        return None

    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    results = HAND_LANDMARKER.detect(mp_image)
    if not results.hand_landmarks:
        return None

    hand_landmarks = results.hand_landmarks[0]
    vec: list[float] = []
    for lm in hand_landmarks:
        vec.extend([lm.x, lm.y, lm.z])
    return np.array(vec, dtype=np.float32)


def classify_from_landmarks(landmarks: np.ndarray) -> Tuple[float, str]:
    """
    Classify ASL letter/number from a 63-dim hand landmark vector.
    """
    if landmarks is None or len(landmarks) < 63:
        return 0.0, "?"

    if LANDMARK_CLASSIFIER is not None:
        try:
            X = landmarks.reshape(1, -1)
            if hasattr(LANDMARK_CLASSIFIER, "predict_proba"):
                probs = LANDMARK_CLASSIFIER.predict_proba(X)[0]
                idx = int(np.argmax(probs))
                confidence = float(probs[idx])
                label = LABELS[idx] if idx < len(LABELS) else "?"
                return confidence, label
            pred = LANDMARK_CLASSIFIER.predict(X)[0]
            label = LABELS[int(pred)] if 0 <= int(pred) < len(LABELS) else "?"
            return 0.9, label
        except Exception:
            pass

    # Placeholder
    r = float(np.mean(landmarks) + np.std(landmarks))
    idx = int(abs(r) * 1e6) % len(LABELS)
    confidence = 0.5 + (abs(r) % 0.5)
    return min(1.0, confidence), LABELS[idx]


def predict_sign_from_image(img_bgr: np.ndarray) -> Tuple[float, str]:
    landmarks = get_hand_landmarks_vector(img_bgr)
    return classify_from_landmarks(landmarks)


def score_sign_from_image(
    img_bgr: np.ndarray, target_label: str
) -> Tuple[float, str]:
    confidence, predicted_label = predict_sign_from_image(img_bgr)
    if target_label not in LABEL_TO_INDEX:
        return 0.0, predicted_label
    score = confidence if predicted_label == target_label else 0.0
    return score, predicted_label


def decode_data_url_to_image(data_url: str) -> np.ndarray | None:
    match = re.match(r"^data:image/[^;]+;base64,(.+)$", data_url)
    if not match:
        return None
    try:
        image_bytes = base64.b64decode(match.group(1))
    except (base64.binascii.Error, ValueError):
        return None
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


# ── Letter / number endpoints ─────────────────────────────────────────────────

@app.route("/api/score-sign", methods=["POST"])
def score_sign():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "JSON body required"}), 400

    image_data = data.get("image")
    target_label = data.get("target")

    if not image_data or not target_label:
        return jsonify({"error": "`image` and `target` are required"}), 400

    img_bgr = decode_data_url_to_image(image_data)
    if img_bgr is None:
        return jsonify({"error": "Invalid or unreadable image data"}), 400

    score, predicted_label = score_sign_from_image(img_bgr, target_label)
    return jsonify({"score": float(score), "predicted_label": predicted_label, "target": target_label})


@app.route("/api/predict-sign", methods=["POST"])
def predict_sign():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "JSON body required"}), 400

    image_data = data.get("image")
    if not image_data:
        return jsonify({"error": "`image` is required"}), 400

    img_bgr = decode_data_url_to_image(image_data)
    if img_bgr is None:
        return jsonify({"error": "Invalid or unreadable image data"}), 400

    confidence, predicted_label = predict_sign_from_image(img_bgr)
    return jsonify({"predicted_label": predicted_label, "confidence": float(confidence)})


# ── Word endpoints ────────────────────────────────────────────────────────────

@app.route("/api/predict-word-snapshot", methods=["POST"])
def predict_word_snapshot():
    """
    Call this repeatedly with one frame at a time. The buffer fills up to
    SEQUENCE_LENGTH frames, then inference runs once and the result is locked
    until /api/clear-buffer is called.

    Response JSON:
      {
        "ready": bool,           -- true once inference has run
        "locked": bool,          -- true if buffer already full, ignoring new frames
        "frames_collected": int,
        "frames_needed": int,
        "predicted_label": str,  -- only present when ready=true
        "confidence": float      -- only present when ready=true
      }
    """
    if LSTM_MODEL is None:
        return jsonify({"error": "LSTM model not loaded"}), 503

    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "JSON body required"}), 400

    session_id = data.get("session_id", "default")
    image_data = data.get("image")

    if not image_data:
        return jsonify({"error": "`image` is required"}), 400

    if session_id not in frame_buffers:
        frame_buffers[session_id] = deque(maxlen=SEQUENCE_LENGTH)

    buf = frame_buffers[session_id]

    # Buffer already full — result is locked, don't process new frames
    if len(buf) >= SEQUENCE_LENGTH:
        return jsonify({
            "ready": False,
            "locked": True,
            "frames_collected": len(buf),
            "frames_needed": SEQUENCE_LENGTH,
        })

    # Extract landmarks from the incoming frame
    img_bgr = decode_data_url_to_image(image_data)
    landmarks = get_hand_landmarks_vector(img_bgr) if img_bgr is not None else None
    if landmarks is None:
        return jsonify({
            "ready": False,
            "locked": False,
            "frames_collected": len(buf),
            "frames_needed": SEQUENCE_LENGTH,
            "no_hand": True
        })
    buf.append(landmarks)

    # Not enough frames yet
    if len(buf) < SEQUENCE_LENGTH:
        return jsonify({
            "ready": False,
            "locked": False,
            "frames_collected": len(buf),
            "frames_needed": SEQUENCE_LENGTH,
        })

    # Buffer just filled — run inference once
    sequence = np.array(buf, dtype=np.float32)       # (SEQUENCE_LENGTH, 63)
    tensor = torch.tensor(sequence).unsqueeze(0)      # (1, SEQUENCE_LENGTH, 63)

    with torch.no_grad():
        logits = LSTM_MODEL(tensor)
        probs = torch.softmax(logits, dim=-1)
        confidence, idx = probs.max(dim=-1)

    return jsonify({
        "ready": True,
        "locked": False,
        "predicted_label": LSTM_LABELS[idx.item()],
        "confidence": round(confidence.item(), 4),
        "frames_collected": SEQUENCE_LENGTH,
        "frames_needed": SEQUENCE_LENGTH,
    })


@app.route("/api/clear-buffer", methods=["POST"])
def clear_buffer():
    """Call this when the user moves to a new question to reset the frame buffer."""
    data = request.get_json(silent=True)
    session_id = data.get("session_id", "default") if data else "default"
    if session_id in frame_buffers:
        del frame_buffers[session_id]
    return jsonify({"status": "cleared"})


# ── Health ────────────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=True)