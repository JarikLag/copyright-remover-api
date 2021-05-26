import pandas as pd
import numpy as np
import os
import torch
import timm
import torchvision.transforms as transforms
import torch.nn as nn

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, precision_score

import application.image_transformer.core as transformer
import application.matting.matting as matting
import application.detector.core as detector


class AuthorDataset(Dataset):
    def __init__(self, data, path, transform=None):
        super().__init__()
        self.data = data.values
        self.path = path
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_name, label = self.data[index]
        img_path = os.path.join(self.path, img_name)
        image = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def action(load_model_path=None):
    labels = pd.read_csv('labels.csv')
    train_path = 'test_r'
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image_transform = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize(mean, std)])

    train, valid = train_test_split(labels, stratify=labels.label, test_size=0.2, random_state=42)
    train_data = AuthorDataset(train, train_path, image_transform)
    valid_data = AuthorDataset(valid, train_path, image_transform)

    num_epochs = 25
    num_classes = 3
    batch_size = 4
    learning_rate = 0.001
    weight_decay = 0.0005
    momentum = 0.9
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=0)
    valid_loader = DataLoader(dataset=valid_data, batch_size=batch_size, shuffle=False, num_workers=0)

    # model = timm.create_model('resnest14d', pretrained=False).to(device)
    model = timm.create_model('cspdarknet53', pretrained=False, num_classes=3).to(device)
    # model = timm.create_model('efficientnet_b1_pruned', pretrained=False, num_classes=3).to(device)
    criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    epoch_last = 1

    if load_model_path is not None:
        checkpoint = torch.load(load_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # epoch_last = checkpoint['epoch']

    train_losses = []
    valid_losses = []
    for epoch in range(epoch_last, num_epochs + 1):
        train_loss = 0.0
        valid_loss = 0.0

        # training-the-model
        model.train()
        for data, target in train_loader:
            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * data.size(0)

        model.eval()
        for data, target in valid_loader:
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            loss = criterion(output, target)
            valid_loss += loss.item() * data.size(0)

        # calculate-average-losses
        train_loss = train_loss / len(train_loader.sampler)
        valid_loss = valid_loss / len(valid_loader.sampler)
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        # print-training/validation-statistics
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch, train_loss, valid_loss))

        if epoch % 25 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, f'../drive/MyDrive/class_dark_val/model_{epoch}.pth')
            model.eval()  # it-disables-dropout

    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels_l in valid_loader:
            images = images.to(device)
            labels_l = labels_l.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels_l.size(0)
            correct += (predicted == labels_l).sum().item()
    print('Test Accuracy of the model: {} %'.format(100 * correct / total))
    return


def calc_stats(model_name, load_model_path):
    labels = pd.read_csv('labels.csv')
    train_path = 'test_r'
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image_transform = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize(mean, std)])
    batch_size = 4
    _, valid = train_test_split(labels, stratify=labels.label, test_size=0.2, random_state=42)
    valid_data = AuthorDataset(valid, train_path, image_transform)

    valid_loader = DataLoader(dataset=valid_data, batch_size=batch_size, shuffle=False, num_workers=0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = timm.create_model(model_name, pretrained=False, num_classes=3).to(device)
    if load_model_path is not None:
        checkpoint = torch.load(load_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
    true_labels = []
    pred_labels = []
    with torch.no_grad():
        for images, labels_l in valid_loader:
            true_labels.extend(labels_l.numpy())
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            pred_labels.extend(predicted.cpu().numpy())
        print(f"Precision micro: {precision_score(true_labels, pred_labels, average='micro')}")
        print(f"Precision macro: {precision_score(true_labels, pred_labels, average='macro')}")
        print(f"Recall micro: {recall_score(true_labels, pred_labels, average='micro')}")
        print(f"Recall macro: {recall_score(true_labels, pred_labels, average='macro')}")


def create_val_dataset():
    import shutil
    global_root = '/media/jariklag/762821B428217471/Users/jarik/Рабочий стол/oasis_dataset'
    classes = ['car', 'wedding_cake', 'yacht', 'plane', 'helicopter']
    for clazz in classes:
        imgs_root = f'{global_root}/{clazz}/img'
        imgs = os.listdir(imgs_root)
        labels_root = f'{global_root}/{clazz}/label'
        for i, img in enumerate(imgs):
            try:
                if i % 5 == 0:
                    shutil.move(f'{imgs_root}/{img}', f'{global_root}/val_img/{img}')
                    shutil.move(f'{labels_root}/{img}', f'{global_root}/val_label/{img}')
                    print(f'Done with {clazz}/{img}, idx is {i}')
            except Exception as ex:
                print(f'Exception with file {clazz}/{img}, idx is {i}: {ex}')


def prepare_dataset():
    size = 256, 256
    global_root = '/'
    save_root = '/'
    labels = {
        'car': 'car',
        'helicopter': 'airplane',
        'plane': 'airplane',
        'wedding_cake': 'cake',
        'yacht': 'boat'
    }
    classes = os.listdir(global_root)
    for clazz in classes:
        cur_class_path = f'{global_root}/{clazz}'
        cur_class_save = f'{save_root}/{clazz}'
        img_names = os.listdir(cur_class_path)
        for img in img_names:
            try:
                image = Image.open(f'{cur_class_path}/{img}')
                image_np = detector.detect_and_crop_specified_labels(image, [labels[clazz]])
                if image_np is None:
                    print(f'Cannot find {clazz} on {img}')
                    continue
                image_np = transformer.make_thumbnail(image_np, size)
                trimap = transformer.generate_trimap(image_np, '../models')
                matting_trimap = matting.trimap_for_matting(trimap)
                matting_img = matting.image_for_matting(image_np)
                fg, _, alpha = matting.perform_matting(matting_img, matting_trimap, '../models/fba_matting.pth')
                fixed_image = transformer.swap_bg(fg[:, :, ::-1] * 255, alpha)
                segmentation = transformer.trimap_to_segmentation(trimap, labels[clazz])
                segmentation = transformer.make_image_square(segmentation, size, 'L')
                sq_image = transformer.make_image_square(fixed_image, size, 'RGB')
                segmentation.save(f'{cur_class_save}/label/{img}')
                sq_image.save(f'{cur_class_save}/img/{img}')
                print(f'Done with {clazz}/{img}')
            except Exception as ex:
                print(f'Exception on {clazz}/{img}: {ex}')
                continue


if __name__ == '__main__':
    pass
