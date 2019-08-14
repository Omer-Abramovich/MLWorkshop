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


parser = argparse.ArgumentParser(description='Cnn v.1')
parser.add_argument('--base_path', type=str, default='/data/ml_workshop_small')
parser.add_argument('--video_path', type=str, default='**/Infant/*.mp4')
parser.add_argument('--delimiter', type=str, default='/')
parser.add_argument('--target_path', type=str, default='/Targets.pth')
parser.add_argument('--scale_size', type=int, default=256)
parser.add_argument('--crop_size', type=int, default=224)
parser.add_argument('--train_portion', type=float, default=0.8)
parser.add_argument('--validation_portion', type=float, default=0.1)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--gamma', type=float, default=0.1)
parser.add_argument('--step_size', type=int, default=7)
parser.add_argument('--GPU_device', type=str, default='cuda:0')
parser.add_argument('--use_cuda', action='store_true')
parser.add_argument('--shuffle', action='store_true')
parser.add_argument('--no_pin_memory', action='store_false')
parser.add_argument('--number_of_classes', type=int, default=15)
parser.add_argument('--number_of_workers', type=int, default = 4)
parser.add_argument('--epochs', type=int, default = 1)
parser.add_argument('--save_path', type=str, default = 'testSaving.pth' )

args = parser.parse_args()



class VideoDataset(torch.utils.data.Dataset):
    
    def __init__(self, base_path, transform=None):
        self.base_path = base_path
        self.transform = transform
        
        self.video_files = []
        self.target_files = []
        
        self.cumulative_lengths = []
        self.video_index = None
        
        self.video = None
        self.targets = None
        
        for x in Path(base_path).glob(args.video_path):
            path_str = str(x)
            print(path_str)
            target_path = '/'.join(path_str.split('/')[:-1]) + args.target_path
            
            self.video_files.append(path_str)
            self.target_files.append(target_path)
            
            video = moviepy.editor.VideoFileClip(path_str)
            frames = math.floor(video.duration * video.fps)
            if len(self.cumulative_lengths)>0:
                self.cumulative_lengths.append(self.cumulative_lengths[-1] + frames)                
            else:
                self.cumulative_lengths.append(frames)
    
    def load_video_if_needed(self, index):
        if not (self.video_index is not None and index < self.cumulative_lengths[self.video_index] and \
                (self.video_index == 0 or index > self.cumulative_lengths[self.video_index-1])):
            for self.video_index in range(len(self.cumulative_lengths)):
                if index < self.cumulative_lengths[self.video_index] and (self.video_index == 0 or index > self.cumulative_lengths[self.video_index-1]):
                    print('Loading video',self.video_index)
                    self.video = moviepy.editor.VideoFileClip(self.video_files[self.video_index])
                    self.targets = torch.load(self.target_files[self.video_index]).float()
                    break
    
    def __getitem__(self, index):
        self.load_video_if_needed(index)
        if self.video_index > 0:
            frame_no = index - self.cumulative_lengths[self.video_index-1]
        else:
            frame_no = index    
        
        frame = self.video.get_frame(frame_no / self.video.fps)
        frame = Image.fromarray(frame.astype('uint8'), 'RGB')
        
        if self.transform:
            frame = self.transform(frame)
        
        target = self.targets[frame_no]
        
        return frame, target
        
        
        
    def __len__(self):
        return int(self.cumulative_lengths[-1])
        




normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
transform = transforms.Compose([
    transforms.Scale(args.scale_size),
    #transforms.RandomRotation(10),
    transforms.RandomCrop(args.crop_size),
    transforms.ToTensor(),
    normalize
])
print('Creating dataset')
dataset = VideoDataset(args.base_path, transform)



train_len = int(args.train_portion * len(dataset))
val_len = int(args.validation_portion * len(dataset))
test_len = len(dataset) - train_len - val_len

train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_len, val_len, test_len])

train_dataset.indices.sort()
val_dataset.indices.sort()
test_dataset.indices.sort()

train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=args.shuffle,
        num_workers=args.number_of_workers, pin_memory=args.no_pin_memory)

val_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=args.shuffle,
        num_workers=args.number_of_workers, pin_memory=args.no_pin_memory)

test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=args.shuffle,
        num_workers=args.number_of_workers, pin_memory=args.no_pin_memory)

device = torch.device(args.GPU_device if (args.use_cuda and torch.cuda.is_available()) else 'cpu')


print('Creating model')
NUM_CLASSES = args.number_of_classes

model_ft = models.resnet18(pretrained=True)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, NUM_CLASSES)

if torch.cuda.device_count() > 1 and args.use_cuda:
    print ("Let's use", torch.cuda.device_count(), "GPUs!")
    # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    model_ft = nn.DataParallel(model_ft)

model_ft = model_ft.to(device)

act = nn.Sigmoid().to(device)

criterion = nn.BCELoss().to(device)

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), args.learning_rate, args.momentum)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, args.step_size, args.gamma)

dataloaders = {
    'train': train_loader,
    'val': val_loader,
    'test': test_loader
}

dataset_sizes = {
    'train': len(train_loader),
    'val': len(val_loader),
    'test': len(test_loader),
}

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            batch_no = 0
            
            # Iterate over data.
            for data in dataloaders[phase]:
                #print('loaded data')
                inputs, labels = data
                
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = act(model(inputs))
                    
                    preds = torch.round(outputs)
                    loss = criterion(outputs, labels).mean()
                    #print('batch loss', loss.item())

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        #print('trained')

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                batch_no += 1
                
                

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            
            

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

print('Starting training')
trained_model = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, args.epochs)
torch.save(trained_model.state_dict(), args.save_path)