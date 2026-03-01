import argparse
from hand_cm import Hand_Video_cm
import cv2 as cv
import os

TARGET_FPS = 8

def main(argv=None):
    p = argparse.ArgumentParser(description="Description")
    p.add_argument('--video', help='input video path,', default=None)
    p.add_argument('--label', help='label', default=None)
    args = p.parse_args(argv)

    video = args.video
    label = args.label

    if video is None:
        assert ValueError("No video input file")
    if label is None:
        assert ValueError("No label was provided") 
    
    cap = cv.VideoCapture(video)
    fps = cap.get(cv.CAP_PROP_FPS)
    
    # Calculate frame interval for 8 fps capture
    TARGET_FPS = 8
    frame_interval = int(fps / TARGET_FPS)
    
    # Create output directory if it doesn't exist
    captured_frames = []

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        # Save frame at the target rate
        if frame_count % frame_interval == 0:
            captured_frames.append(frame)
        frame_count += 1

    def print_res(landmark_res):
        print(landmark_res)

    with Hand_Video_cm(print_res) as hand_cm:
        test = [hand_cm.detect(frame) for frame in captured_frames]

    print(test)
    
if __name__ == "__main__":
    main()    