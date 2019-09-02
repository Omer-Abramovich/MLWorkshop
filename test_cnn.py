import os

import torch
import torch.nn

from torchvision import transforms

import argparse
import utils.dataset

parser = argparse.ArgumentParser(description='Cnn_test v.1')
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
parser.add_argument('--max_epoch_size', type=int, default=0)
parser.add_argument('--frames', action='store_true')
parser.add_argument('--audio', action='store_true')
parser.add_argument('--load_path', type=str, default='/mnt/models/cnn_audio_final.pth.epoch.9')
parser.add_argument('--save_path', type=str, default='confusion_matrix_audio.pth')
args = parser.parse_args()

device = torch.device(args.GPU_device if (args.use_cuda and torch.cuda.is_available()) else 'cpu')

model = torch.load(args.load_path)
act = torch.nn.Sigmoid().to(device)

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
transform = transforms.Compose([
    transforms.Scale(args.scale_size),
    # transforms.RandomRotation(10),
    transforms.RandomCrop(args.crop_size),
    transforms.ToTensor(),
    normalize
])

dataset = utils.dataset.VideoDataset(args.base_path, args, transform)

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

NUM_CLASSES = args.number_of_classes

train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=args.shuffle,
        num_workers=args.number_of_workers, pin_memory=args.no_pin_memory)

test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=args.batch_size, shuffle=args.shuffle,
    num_workers=args.number_of_workers, pin_memory=args.no_pin_memory)


def test_model(model):
    class_correct = torch.zeros(NUM_CLASSES)
    class_positives = torch.zeros(NUM_CLASSES)
    class_correct_positives = torch.zeros(NUM_CLASSES)
    class_correct_negatives = torch.zeros(NUM_CLASSES)
    class_negatives = torch.zeros(NUM_CLASSES)
    class_false_positives = torch.zeros(NUM_CLASSES)
    class_false_negatives = torch.zeros(NUM_CLASSES)

    confusion_matrix = torch.zeros(NUM_CLASSES, 2, 2)

    total_frames = 0
    with torch.no_grad():
        for data in test_loader:
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

            outputs = act(model(inputs))
            preds = torch.round(outputs)
            preds = preds.cpu()

            for cls in range(NUM_CLASSES):
                for i in range(preds.size(0)):
                    confusion_matrix[cls, labels[i, cls].long(), preds[i,cls].long()] += 1

            for label in range(NUM_CLASSES):
                for i in range(preds.size(0)):
                    class_correct[label] += (preds[i,label] == labels[i,label])
                    if labels[i,label] == 1:
                        class_positives[label] += 1
                        class_correct_positives[label] += (preds[i,label] == 1)
                        class_false_negatives[label] += (preds[i,label] == 0)
                    else:
                        class_negatives[label] += 1
                        class_correct_negatives[label] += (preds[i,label] == 0)
                        class_false_positives[label] += (preds[i,label] == 1)
            total_frames += args.batch_size

            print("frames", total_frames)

    torch.save(confusion_matrix, args.save_path)

    for label in range(NUM_CLASSES):
        print('Accuracy of label %1d : %2d %%' % (
            label, 100 * class_correct[label] / total_frames))

    for label in range(NUM_CLASSES):
        print('Correct Positives for label %1d : %2d %%' % (
            label, 100 * class_correct_positives[label] / max(1,class_positives[label])))

    for label in range(NUM_CLASSES):
        print('False Positives for label %1d : %2d %%' % (
            label, 100 * class_false_positives[label] / max(1,class_negatives[label])))

    for label in range(NUM_CLASSES):
        print('Correct negatives for label %1d : %2d %%' % (
            label, 100 * class_correct_negatives[label] / max(1,class_negatives[label])))

    for label in range(NUM_CLASSES):
        print('False Negatives for label %1d : %2d %%' % (
            label, 100 * class_false_negatives[label] / max(1,class_positives[label])))




test_model(model)
