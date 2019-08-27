import moviepy.editor
import os
from pathlib import Path
from PIL import Image

import torch
import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('--base_path', type=str, default='/data/ml_workshop_small')

args = parser.parse_args()

for x in Path(args.base_path).glob('**/Infant/resized.mp4'):
    path_str = str(x)
    target_path = '/'.join(path_str.split('/')[:-1]) + '/Targets.pth'
    folder_path = '/'.join(path_str.split('/')[:-1]) + '/Frames/'

    if not os.path.exists(folder_path):
        os.mkdir(folder_path)

    print(path_str)

    video = moviepy.editor.VideoFileClip(path_str)
    target = torch.load(target_path)
    video.fps = target.size(0) / video.duration
    for i in range(target.size(0)):
        frame = video.get_frame(i * video.fps)
        frame = Image.fromarray(frame.astype('uint8'), 'RGB')
        frame.save(folder_path + str(i) + '.png')



