import argparse
from hand_cm import Image_Video_cm
from pathlib import Path
import cv2 as cv
import os
import numpy as np
import pickle

TARGET_FPS = 8

def main(argv=None):
    p = argparse.ArgumentParser(description="Description")
    p.add_argument('--input',   help='input video path,', default=None)
    p.add_argument('--output',  help='label', default=None)
    args = p.parse_args(argv)

    input = args.input
    ouput = args.ouput

    if video is None or output is None:
        assert ValueError("Invalid input or output")

    def return_res(landmark_res):
        return landmark_res

    with Image_Video_cm(return_res) as hand_cm:
        # Use rglob for "recursive glob"
        for input in input_root.rglob("*"): 
            if input.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]:
        
            relative_path = input.relative_to(input_root)
            output = output_root / relative_path

            output.parent.mkdir(parents=True, exist_ok=True)

            with Image.open(input) as img:
                
                data = hand_cm.detect(img)
                # Pickling: Writing to a binary file
                with open(output+'savegame.pkl', 'wb') as f:
                    pickle.dump(data, f)

        

if __name__ == "__main__":
    main()    