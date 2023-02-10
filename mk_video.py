import pprint
import cv2, os
import numpy as np
from pathlib import Path
from moviepy.editor import ImageSequenceClip

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", "-d", type=str, help="image data path")
    parser.add_argument("--ext", "-e", type=str, default="jpg", help="image extension")
    parser.add_argument("--output", "-o", type=str, default="./", help="output video path")
    args = parser.parse_args()

    if args.output is None:
        args.output = os.path.join('/'.join(args.data_path.split('/')[:-1]), "video")
    if not os.path.exists(args.output):
        os.makedirs(args.output)
        
    # load images
    images_path = Path(args.data_path)
    print("collecting images from {}".format(images_path)
    image_files = sorted([str(p) for p in images_path.rglob(f"**/*.{arg.ext}}")])
    clip = ImageSequenceClip(image_files, fps=30)

    clip.write_videofile(os.path.join(args.output, "video.mp4"), fps=30)