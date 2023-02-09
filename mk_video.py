import pprint
import cv2, os
import numpy as np
from pathlib import Path
from moviepy.editor import ImageSequenceClip

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, 
                    default="/data/NIA50/50-2/models/DeepFusionMOT/results/nia50_train1st/image", help="image data path")
parser.add_argument("--output", "-o", type=str, default=None, help="output video path")
args = parser.parse_args()

if args.output is None:
    args.output = os.path.join('/'.join(args.data_path.split('/')[:-1]), "video")

if not os.path.exists(args.output):
    os.makedirs(args.output)

# load images
images_path = Path(args.data_path)
image_files = sorted(list(images_path.glob("**/*.jpg")))
image_files = [str(image_file) for image_file in image_files]
clip = ImageSequenceClip(image_files, fps=30)

clip.write_videofile("/data/NIA50/50-2/models/DeepFusionMOT/results/nia50_train1st/video/nia50_train1st.mp4")