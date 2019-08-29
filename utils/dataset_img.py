import torch
import os
from pathlib import Path
import moviepy.editor
from PIL import Image
from torchvision import datasets, models, transforms
import math


class VideoDataset(torch.utils.data.Dataset):

    def __init__(self, base_path, args, transform=None):
        self.base_path = base_path
        self.args = args
        self.transform = transform

        self.image_paths = []
        self.index2video = []
        self.targets = []
        self.audios = []

        self.save = True
        # if os.path.exists('video_files.pth'):
        #     self.audio_files = torch.load('audio_files.pth')
        #     self.target_files = torch.load('target_files.pth')
        #     self.save = False

        video_index = 0
        if self.save:
            print('Scanning for files...')
            for x in Path(base_path).glob('**/Infant/resized.mp4'):
                path_str = str(x)
                print(path_str)
                target_path = '/'.join(path_str.split('/')[:-1]) + '/Targets.pth'
                audio_path = '/'.join(path_str.split('/')[:-1]) + '/audio.pth'
                frames_path = '/'.join(path_str.split('/')[:-1]) + '/Frames'

                target = torch.load(target_path)
                audio = torch.load(audio_path).float()

                audio_trans = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Scale((target.size(0), self.args.crop_size)),
                    transforms.ToTensor()
                ])

                audio = audio_trans(audio.mean(0))

                self.audios.append(audio)
                self.targets.append(target)
                for img_path in os.listdir(frames_path):
                    self.image_paths.append(img_path)
                    self.index2video.append(video_index)

                video_index+=1

            torch.save(self.image_paths, 'image_paths.pth')
            torch.save(self.audios, 'audios.pth')
            torch.save(self.targets, 'targets.pth')
            torch.save(self.index2video, 'index2video.pth')

    def __getitem__(self, index):
        self.load_video_if_needed(index)
        if self.video_index > 0:
            frame_no = index - self.cumulative_lengths[self.video_index - 1]
        else:
            frame_no = index
        frame = self.video.get_frame(frame_no / self.video.fps)
        frame = Image.fromarray(frame.astype('uint8'), 'RGB')

        if self.transform:
            frame = self.transform(frame)

        target = self.targets[frame_no]
        audio = torch.zeros(1, self.args.crop_size, self.args.crop_size)
        half = int(self.args.crop_size / 2)
        if frame_no < half:
            audio[:, half - frame_no:] = self.audio[:, :half + frame_no]
        if frame_no >= half and frame_no + half < self.audio.size(1):
            audio = self.audio[:, frame_no - half:frame_no + half]
        if frame_no + half >= self.audio.size(1):
            delta = self.audio.size(1) - frame_no
            audio[:, :half + delta] = self.audio[:, frame_no - half:]

        return frame, audio, target

    def __len__(self):
        return int(self.cumulative_lengths[-1])
