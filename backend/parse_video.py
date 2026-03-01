import argparse
import numpy as np
import pickle
import cv2 as cv
import os
from pathlib import Path
from hand_cm import Hand_Video_cm

TARGET_FPS = 8

def extract_from_video(video_path):
    cap = cv.VideoCapture(video_path)
    fps = cap.get(cv.CAP_PROP_FPS)
    frame_interval = max(1, int(fps / TARGET_FPS))

    captured_frames = []
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            captured_frames.append(frame)
        frame_count += 1
    cap.release()

    def get_vec(landmark_result):
        if not landmark_result.hand_landmarks:
            return np.zeros(63, dtype=np.float32)
        hand_landmarks = landmark_result.hand_landmarks[0]
        vec = []
        for lm in hand_landmarks:
            vec.extend([lm.x, lm.y, lm.z])
        return np.array(vec, dtype=np.float32)

    with Hand_Video_cm(None) as hand_cm:
        raw_results = [hand_cm.detect(frame) for frame in captured_frames]

    sequence = [get_vec(r) for r in raw_results]

    return np.array(sequence, dtype=np.float32)

def augment(sequence):
    # Time stretch first
    if len(sequence) > 10:
        keep = np.random.choice(len(sequence), int(len(sequence) * 0.9), replace=False)
        sequence = sequence[sorted(keep)]
    # Then generate noise matching the new shape
    noise = np.random.normal(0, 0.01, sequence.shape)
    return sequence + noise

def build_dataset(videos_dir="videos_top10", output_path="landmarks.pkl"):
    data = []
    videos_path = Path(videos_dir)

    for label_dir in sorted(videos_path.iterdir()):
        if not label_dir.is_dir():
            continue
        label = label_dir.name
        videos = list(label_dir.glob("*.mp4"))
        print(f"Processing '{label}' ({len(videos)} videos)...")

        for video_path in videos:
            sequence = extract_from_video(str(video_path))
            if sequence is None or len(sequence) == 0:
                continue

            # Original
            data.append({"landmarks": sequence, "label": label})
            # Augmented copies
            aug1 = augment(sequence)
            data.append({"landmarks": aug1, "label": label})

    with open(output_path, "wb") as f:
        pickle.dump(data, f)

    print(f"\nSaved {len(data)} samples to {output_path}")


if __name__ == "__main__":
    build_dataset()