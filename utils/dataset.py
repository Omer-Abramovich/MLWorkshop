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

        self.video_files = []
        self.target_files = []
        self.audio_files = []

        self.cumulative_lengths = []
        self.video_index = None

        self.video = None
        self.targets = None

        self.save = True
        if os.path.exists('cumulative_lengths.pth'):
            self.cumulative_lengths = torch.load('cumulative_lengths.pth')
            self.video_files = torch.load('video_files.pth')
            self.audio_files = torch.load('audio_files.pth')
            self.target_files = torch.load('target_files.pth')
            self.save = False

        if self.save:
            print('Scanning for files...')
            for x in Path(base_path).glob('**/Infant/resized.mp4'):
                path_str = str(x)
                print(path_str)
                target_path = '/'.join(path_str.split('/')[:-1]) + '/Targets.pth'
                audio_path = '/'.join(path_str.split('/')[:-1]) + '/audio.pth'

                self.video_files.append(path_str)
                self.audio_files.append(audio_path)
                self.target_files.append(target_path)

                video = moviepy.editor.VideoFileClip(path_str)
                target = torch.load(target_path)
                video.fps = target.size(0) / video.duration
                frames = math.floor(video.duration * video.fps)
                if len(self.cumulative_lengths) > 0:
                    self.cumulative_lengths.append(self.cumulative_lengths[-1] + frames)
                else:
                    self.cumulative_lengths.append(frames)
            torch.save(self.cumulative_lengths, 'cumulative_lengths.pth')
            torch.save(self.video_files, 'video_files.pth')
            torch.save(self.audio_files, 'audio_files.pth')
            torch.save(self.target_files, 'target_files.pth')

    def load_video_if_needed(self, index):
        if not (self.video_index is not None and index < self.cumulative_lengths[self.video_index] and \
                (self.video_index == 0 or index > self.cumulative_lengths[self.video_index - 1])):
            for self.video_index in range(len(self.cumulative_lengths)):
                if index < self.cumulative_lengths[self.video_index] and (
                        self.video_index == 0 or index >= self.cumulative_lengths[self.video_index - 1]):
                    # print('Loading video',self.video_index)
                    self.video = moviepy.editor.VideoFileClip(self.video_files[self.video_index])
                    self.targets = torch.load(self.target_files[self.video_index]).float()
                    self.video.fps = self.targets.size(0) / self.video.duration

                    audio_trans = transforms.Compose([
                        transforms.ToPILImage(),
                        transforms.Scale((self.targets.size(0), self.args.crop_size)),
                        transforms.ToTensor()
                    ])

                    a = torch.load(self.audio_files[self.video_index]).float()
                    self.audio = audio_trans(a.mean(0))
                    break

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
