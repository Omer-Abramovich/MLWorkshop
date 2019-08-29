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
        if os.path.exists('image_paths.pth'):
            self.image_paths = torch.load('image_paths.pth')
            self.audios = torch.load('audios.pth')
            self.targets = torch.load('targets.pth')
            self.index2video = torch.load('index2video.pth')

            self.save = False

        video_index = 0
        if self.save:
            print('Scanning for files...')
            for x in Path(base_path).glob('**/Infant/resized.mp4'):
                path_str = str(x)
                print(path_str)
                target_path = '/'.join(path_str.split('/')[:-1]) + '/Targets.pth'
                audio_path = '/'.join(path_str.split('/')[:-1]) + '/audio.pth'
                frames_path = '/'.join(path_str.split('/')[:-1]) + '/Frames'

                target = torch.load(target_path).float()
                audio = torch.load(audio_path).float()

                audio_trans = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Scale((target.size(0), self.args.crop_size)),
                    transforms.ToTensor()
                ])

                audio = audio_trans(audio.mean(0))

                self.audios.append(audio)
                self.targets.append(target)
                for i in range(target.size(0)):
                    img_path = frames_path + '/' + str(i) + '.png'

                    self.image_paths.append(img_path)
                    self.index2video.append(video_index)

                video_index+=1

            torch.save(self.image_paths, 'image_paths.pth')
            torch.save(self.audios, 'audios.pth')
            torch.save(self.targets, 'targets.pth')
            torch.save(self.index2video, 'index2video.pth')

    def __getitem__(self, index):
        frame_no = int(self.image_paths[index].split('/')[-1].split('.')[0])
        frame = Image.open(self.image_paths[index])

        if self.transform:
            frame = self.transform(frame)

        video_audio = self.audios[self.index2video[index]]
        target = self.targets[self.index2video[index]][frame_no]

        audio = torch.zeros(1, self.args.crop_size, self.args.crop_size)
        half = int(self.args.crop_size / 2)
        if frame_no < half:
            audio[:, half - frame_no:] = video_audio[:, :half + frame_no]
        if frame_no >= half and frame_no + half < video_audio.size(1):
            audio = video_audio[:, frame_no - half:frame_no + half]
        if frame_no + half >= video_audio.size(1):
            delta = video_audio.size(1) - frame_no
            audio[:, :half + delta] = video_audio[:, frame_no - half:]

        return frame, audio, target

    def __len__(self):
        return len(self.image_paths)
