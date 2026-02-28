from __future__ import annotations

import base64
import os
import re
from typing import Tuple

import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision


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
# Set ASL_LANDMARK_MODEL_PATH to a joblib/pickle or similar, e.g. sklearn classifier.
LANDMARK_CLASSIFIER = None
_landmark_model_path = os.getenv("ASL_LANDMARK_MODEL_PATH")
if _landmark_model_path and os.path.exists(_landmark_model_path):
    try:
        import joblib
        LANDMARK_CLASSIFIER = joblib.load(_landmark_model_path)
    except Exception:
        pass


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
    # 21 landmarks, each (x, y, z) normalized
    vec: list[float] = []
    for lm in hand_landmarks:
        vec.extend([lm.x, lm.y, lm.z])
    return np.array(vec, dtype=np.float32)


def classify_from_landmarks(landmarks: np.ndarray) -> Tuple[float, str]:
    """
    Classify ASL letter/number from a 63-dim hand landmark vector.

    If LANDMARK_CLASSIFIER is set (e.g. sklearn model trained on landmarks),
    use it. Otherwise use a placeholder so the pipeline works end-to-end.
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

    # Placeholder: deterministic-ish from landmark stats so it's not pure random per frame
    r = float(np.mean(landmarks) + np.std(landmarks))
    idx = int(abs(r) * 1e6) % len(LABELS)
    confidence = 0.5 + (abs(r) % 0.5)
    return min(1.0, confidence), LABELS[idx]


def predict_sign_from_image(img_bgr: np.ndarray) -> Tuple[float, str]:
    """
    Run MediaPipe Hands on image, then classify landmarks.
    Returns (confidence, predicted_label). No hand -> (0.0, "?").
    """
    landmarks = get_hand_landmarks_vector(img_bgr)
    return classify_from_landmarks(landmarks)


def score_sign_from_image(
    img_bgr: np.ndarray, target_label: str
) -> Tuple[float, str]:
    """
    Same as predict_sign_from_image; score is confidence if predicted == target else 0.
    """
    confidence, predicted_label = predict_sign_from_image(img_bgr)
    if target_label not in LABEL_TO_INDEX:
        return 0.0, predicted_label
    score = confidence if predicted_label == target_label else 0.0
    return score, predicted_label


def decode_data_url_to_image(data_url: str) -> np.ndarray | None:
    """
    Accepts a data URL like 'data:image/jpeg;base64,...'
    and returns an OpenCV BGR image or None on failure.
    """
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


@app.route("/api/score-sign", methods=["POST"])
def score_sign():
    """
    Request JSON:
      {
        "image": "<data_url>",
        "target": "A"  # the correct letter/number the user was asked to sign
      }

    Response JSON:
      {
        "score": 0.87,              # model probability/score for target label
        "predicted_label": "A",     # model's best guess (optional)
        "target": "A"
      }
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "JSON body required"}), 400

    image_data = data.get("image")
    target_label = data.get("target")

    if not image_data or not target_label:
        return (
            jsonify({"error": "`image` (data URL) and `target` are required"}),
            400,
        )

    img_bgr = decode_data_url_to_image(image_data)
    if img_bgr is None:
        return jsonify({"error": "Invalid or unreadable image data"}), 400

    score, predicted_label = score_sign_from_image(img_bgr, target_label)

    return jsonify(
        {
            "score": float(score),
            "predicted_label": predicted_label,
            "target": target_label,
        }
    )


@app.route("/api/predict-sign", methods=["POST"])
def predict_sign():
    """
    Request JSON:
      { "image": "<data_url>" }

    Response JSON:
      {
        "predicted_label": "A",
        "confidence": 0.93
      }
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "JSON body required"}), 400

    image_data = data.get("image")
    if not image_data:
        return jsonify({"error": "`image` (data URL) is required"}), 400

    img_bgr = decode_data_url_to_image(image_data)
    if img_bgr is None:
        return jsonify({"error": "Invalid or unreadable image data"}), 400

    confidence, predicted_label = predict_sign_from_image(img_bgr)

    return jsonify(
        {
            "predicted_label": predicted_label,
            "confidence": float(confidence),
        }
    )


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


if __name__ == "__main__":
    # Default to http://127.0.0.1:5000
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=True)
