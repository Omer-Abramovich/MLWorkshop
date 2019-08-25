# Preprocess data 2
import os
import sys
import math
import time
import copy
from pathlib import Path

import moviepy.editor
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

from torchvision import datasets, models, transforms
import torchvision.transforms as transforms

import torchvision

import matplotlib.pyplot as plt

import argparse

for x in Path('/mnt/data/ml_workshop_processed/').glob('**/Infant/source.mp4'):
    path_str = str(x)
    print(path_str)
    target_path = '/'.join(path_str.split('/')[:-1]) + '/resized.mp4'
    
    if not os.path.exists(target_path):
        video = moviepy.editor.VideoFileClip(path_str)
        clip_resized = video.resize(height=256)
        clip_resized.write_videofile(target_path)