# Tests the model on a camera feed that shows its predicted character/word in real time
# Uses pre-trained MediaPipe hand landmarker to detect and crop hand region before passing to model

import cv2
import torch
import torch.nn as nn
import numpy as np
import mediapipe as mp
import urllib.request, os
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from torchvision import transforms, models
from collections import deque

# Configuration
MODEL_PATH           = "backend/ml/asl_model.pth"
HAND_LANDMARKER_URL  = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
HAND_LANDMARKER_FILE = "backend/ml/hand_landmarker.task"
IMG_SIZE             = 224
CONFIDENCE           = 0.7 # Min confidence to show prediction
SMOOTH_FRAMES        = 10  # Num of frames to smooth predictions over
DEVICE               = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
GESTURES   = checkpoint["gestures"]

model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(model.last_channel, len(GESTURES))
)
model.load_state_dict(checkpoint["model_state"])
model.eval().to(DEVICE)

print(f"- Model loaded | Gestures: {GESTURES}")

# Transform
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

if not os.path.exists(HAND_LANDMARKER_FILE):
    print("Downloading MediaPipe hand landmarker model...")
    urllib.request.urlretrieve(HAND_LANDMARKER_URL, HAND_LANDMARKER_FILE)
    print("- Downloaded hand_landmarker.task")

base_options  = mp_python.BaseOptions(model_asset_path=HAND_LANDMARKER_FILE)
hand_options  = mp_vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_tracking_confidence=0.6,
    running_mode=mp_vision.RunningMode.VIDEO,
)
hand_landmarker = mp_vision.HandLandmarker.create_from_options(hand_options)

# Connection pairs for drawing skeleton
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17)
]

# Prediction smoothing
prediction_buffer = deque(maxlen=SMOOTH_FRAMES)

def smooth_prediction(label):
    prediction_buffer.append(label)
    return max(set(prediction_buffer), key=prediction_buffer.count)

def crop_hand(frame, landmarks, padding=40):
    h, w = frame.shape[:2]
    xs = [lm.x * w for lm in landmarks]
    ys = [lm.y * h for lm in landmarks]

    x1 = max(0, int(min(xs)) - padding)
    y1 = max(0, int(min(ys)) - padding)
    x2 = min(w, int(max(xs)) + padding)
    y2 = min(h, int(max(ys)) + padding)
    
    # Force square crop
    side = max(x2 - x1, y2 - y1)
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    x1 = max(0, cx - side // 2)
    y1 = max(0, cy - side // 2)
    x2 = min(w, cx + side // 2)
    y2 = min(h, cy + side // 2)

    return frame[y1:y2, x1:x2], (x1, y1, x2, y2)

def draw_skeleton(frame, landmarks):
    h, w = frame.shape[:2]
    points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, points[a], points[b], (0, 180, 180), 2)
    for pt in points:
        cv2.circle(frame, pt, 3, (0, 255, 255), -1)

def predict(crop):
    tensor = transform(crop).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = model(tensor)
        probs  = torch.softmax(output, dim=1)
        conf, idx = probs.max(1)
    return GESTURES[idx.item()], conf.item()

def draw_overlay(frame, label, confidence, bbox):
    x1, y1, x2, y2 = bbox
    color = (0, 200, 0) if confidence >= CONFIDENCE else (0, 100, 255)

    # Bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Label background
    text  = f"{label}  {confidence*100:.0f}%"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
    cv2.rectangle(frame, (x1, y1 - th - 12), (x1 + tw + 8, y1), color, -1)
    cv2.putText(frame, text, (x1 + 4, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)



# Main
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Unable to open web camera")

print("- Starting webcam — press Q to quit")
frame_ts = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame    = cv2.flip(frame, 1)
    rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result   = hand_landmarker.detect_for_video(mp_image, frame_ts)
    frame_ts += 1

    if result.hand_landmarks:
        for landmarks in result.hand_landmarks:
            draw_skeleton(frame, landmarks)

            crop, bbox = crop_hand(frame, landmarks)
            if crop.size > 0:
                label, confidence = predict(crop)
                label = smooth_prediction(label)
                draw_overlay(frame, label, confidence, bbox)
    else:
        prediction_buffer.clear()
        cv2.putText(frame, "No hand detected", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 100, 255), 2)

    # Instructions
    cv2.putText(frame, "Press Q to quit", (20, frame.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

    cv2.imshow("ASL Gesture Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print("exit")