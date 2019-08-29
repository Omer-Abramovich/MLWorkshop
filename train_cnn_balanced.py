import os
import time
import copy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

from torchvision import models
import torchvision.transforms as transforms

import argparse

import utils.dataset_img

parser = argparse.ArgumentParser(description='Cnn v.1')
parser.add_argument('--base_path', type=str, default='/data/ml_workshop_small')
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
parser.add_argument('--number_of_workers', type=int, default=4)
parser.add_argument('--epochs', type=int, default=1)
parser.add_argument('--save_path', type=str, default='testSaving.pth')
parser.add_argument('--max_epoch_size', type=int, default=0)
parser.add_argument('--frames', action='store_true')
parser.add_argument('--audio', action='store_true')

args = parser.parse_args()

if not args.frames and not args.audio:
    print("Atleast one input type (frames/audio) must be set!")
    exit(0)

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
transform = transforms.Compose([
    transforms.Scale(args.scale_size),
    # transforms.RandomRotation(10),
    transforms.RandomCrop(args.crop_size),
    transforms.ToTensor(),
    normalize
])

print('Creating dataset')
dataset = utils.dataset_img.VideoDataset(args.base_path, args, transform)

train_len = int(args.train_portion * len(dataset))
val_len = int(args.validation_portion * len(dataset))
test_len = len(dataset) - train_len - val_len

train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_len, val_len, test_len])

if os.path.exists('train_dataset_indices.pth'):
    train_dataset.indices = torch.load('train_dataset_indices.pth')
    val_dataset.indices = torch.load('val_dataset_indices.pth')
    test_dataset.indices = torch.load('test_dataset_indices.pth')
else:
    train_dataset.indices.sort()
    val_dataset.indices.sort()
    test_dataset.indices.sort()

    torch.save(train_dataset.indices, 'train_dataset_indices.pth')
    torch.save(val_dataset.indices, 'val_dataset_indices.pth')
    torch.save(test_dataset.indices, 'test_dataset_indices.pth')

print('Counting targets')

all_targets = torch.cat(dataset.targets, 0)

print(all_targets.size())

counts = all_targets.sum(0).float()

print(counts)

wights = torch.zeros(all_targets.size(1), 2)
wights[:, 0] = (1 / (all_targets.size(0) - counts))
wights[:, 1] = (1 / counts)

norm = wights.sum(1).unsqueeze(1).expand((wights.size(0), 2))

wights = wights / norm

print(wights)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.batch_size, shuffle=args.shuffle,
    num_workers=args.number_of_workers, pin_memory=args.no_pin_memory)

val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=args.batch_size, shuffle=args.shuffle,
    num_workers=args.number_of_workers, pin_memory=args.no_pin_memory)

test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=args.batch_size, shuffle=args.shuffle,
    num_workers=args.number_of_workers, pin_memory=args.no_pin_memory)

device = torch.device(args.GPU_device if (args.use_cuda and torch.cuda.is_available()) else 'cpu')

wights = wights.to(device)

print('Creating model')
NUM_CLASSES = args.number_of_classes

model_ft = models.resnet18(pretrained=True)

if args.frames and args.audio:
    model_ft.conv1 = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
elif args.audio:
    model_ft.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, NUM_CLASSES)

if torch.cuda.device_count() > 1 and args.use_cuda:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
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
    'train': len(train_dataset),
    'val': len(val_dataset),
    'test': len(test_dataset),
}


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    idx = 0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            running_images = 0
            batch_no = 0

            # Iterate over data.
            for data in dataloaders[phase]:
                # print('loaded data')
                frame, audio, labels = data

                if args.frames:
                    frame = frame.to(device)
                if args.audio:
                    audio = audio.to(device)

                if args.frames and args.audio:
                    inputs = torch.cat((frame, audio), 1)
                elif args.frames:
                    inputs = frame
                elif args.audio:
                    inputs = audio

                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = act(model(inputs))

                    preds = torch.round(outputs)
                    loss = criterion(outputs, labels)

                    merged_weights = (labels == 0).float() * wights[:, 0] + (labels == 1).float() * wights[:, 1]
                    loss = loss * merged_weights
                    loss = loss.mean()
                    # print('batch loss', loss.item())

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        # print('trained')

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                running_images += labels.size(0)
                batch_no += 1

                if batch_no == args.max_epoch_size:
                    break

                if idx % 10 == 0:
                    epoch_loss = running_loss / running_images
                    epoch_acc = running_corrects.double() / running_images / NUM_CLASSES

                    print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                        phase, epoch_loss, epoch_acc))

                    if phase == 'train' and idx % 1000 == 0:
                        torch.save(model, args.save_path + '.' + str(idx))
                idx += 1

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

            if phase == 'train':
                torch.save(model, args.save_path + '.epoch.' + str(epoch))

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
