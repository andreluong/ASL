# Download dataset and process it into skeleton images for training

import cv2
import mediapipe as mp
import numpy as np
import os
from pathlib import Path
import kagglehub
import random

MAX_IMAGES_PER_CLASS = 150
IMG_SIZE             = 224

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
HAND_CONNECTIONS = mp_hands.HAND_CONNECTIONS

def draw_simple_skeleton(canvas, hand_landmarks):
    h, w = canvas.shape[:2]
    points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks.landmark]
    for a, b in HAND_CONNECTIONS:
        cv2.line(canvas, points[a], points[b], (255, 255, 255), 2)
    for pt in points:
        cv2.circle(canvas, pt, 3, (255, 255, 255), -1)

def render_skeleton_on_black_bg(image_path, output_path, img_size=(IMG_SIZE, IMG_SIZE)):
    img = cv2.imread(str(image_path))
    if img is None:
        return False
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.3
    ) as hands:
        results = hands.process(img_rgb)
        
        # No hands detected
        if not results.multi_hand_landmarks:
            return False
        
        # Add black canvas
        canvas = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        
        for hand_landmarks in results.multi_hand_landmarks:
            draw_simple_skeleton(canvas, hand_landmarks)
        
        canvas = cv2.resize(canvas, img_size)
        cv2.imwrite(str(output_path), canvas)
        return True

def process_dataset(input_root, output_root):
    input_root = Path(input_root)
    output_root = Path(output_root)
    
    skipped = 0
    processed = 0
    
    for class_dir in sorted(input_root.iterdir()):
        if not class_dir.is_dir():
            continue
        
        out_class_dir = output_root / class_dir.name
        out_class_dir.mkdir(parents=True, exist_ok=True)

        all_images = list(class_dir.glob("*.*"))
        random.shuffle(all_images)
        
        count = 0
        for img_path in all_images:
            if count >= MAX_IMAGES_PER_CLASS:
                break
            out_path = out_class_dir / img_path.name
            success = render_skeleton_on_black_bg(img_path, out_path)
            if success:
                processed += 1
                count += 1
            else:
                skipped += 1
    
    print(f"Processed: {processed}; Skipped (no hand detected): {skipped}")

path = kagglehub.dataset_download("grassknoted/asl-alphabet")
print("Path to dataset files:", path)
input_root = os.path.join(path, "asl_alphabet_train", "asl_alphabet_train")
process_dataset(input_root, "ml/asl_alphabet_skeleton")