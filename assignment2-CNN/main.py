# License: BSD
# Author: Sasank Chilamkurthy

from __future__ import print_function, division
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from XKNet import resnet18, resnet34, resnet14
from PIL import Image

plt.ion()   # interactive mode

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        # transforms.RandomResizedCrop(224),    # Crop a random portion of image and resize it to a given size.
        transforms.RandomHorizontalFlip(p=0.1),
        transforms.RandomVerticalFlip(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        # transforms.Resize(256),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

# 用于读取 test 数据集
class FlameSet(torch.utils.data.Dataset):
    def __init__(self, root):
        # 所有图片的绝对路径
        imgs = os.listdir(root)
        self.imgs = [os.path.join(root, k) for k in imgs]
        self.transforms = data_transforms['val']
        self.name = ['test/' + x for x in imgs]

    def __getitem__(self, index):
        img_path = self.imgs[index]
        pil_img = Image.open(img_path)
        data = self.transforms(pil_img)
        return data, self.name[index]

    def __len__(self):
        return len(self.imgs)

data_dir = 'F:\\2021研一上学期\\深度学习\\ImageData2'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=64, shuffle=True, num_workers=0)
               for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
# class_names = image_datasets['train'].classes
data_test = FlameSet(os.path.join(data_dir, 'test'))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# set random seed
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # random.seed(seed)
    # torch.backends.cudnn.deterministic = True

def train_model(model, loss_fn, optimizer, scheduler, num_epochs=25):
    since = time.time()
    train_ls, train_acc, val_ls, val_acc = [], [], [], []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        t = time.time()
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode (不启用 Batch Normalization 和 Dropout)

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = loss_fn(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f} Time: {:.0f}m {:.0f}s'.format(
                phase, epoch_loss, epoch_acc, (time.time()-t) // 60, (time.time()-t) % 60))

            if phase == 'train':
                train_ls.append(epoch_loss)
                train_acc.append(epoch_acc.item())
            else:
                val_ls.append(epoch_loss)
                val_acc.append(epoch_acc.item())

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

    # plot loss and accuracy curve
    plot(num_epochs, train_ls, val_ls, train_acc, val_acc)
    plt.savefig('./' + time.strftime("%H", time.localtime()) + '.png')
    plt.close()

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, time_elapsed, best_acc, train_ls, val_ls, train_acc, val_acc

def plot(epochs, train_loss, val_loss, train_acc, val_acc) -> None:
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()  # 镜像对称 生成ax2次坐标；
    ax1.plot(range(epochs), train_loss, label='train', linewidth=1.5)
    ax1.plot(range(epochs), val_loss, label='vallidate', linewidth=1.5)
    ax2.plot(range(epochs), train_acc, '--', label='train', linewidth=1.5)
    ax2.plot(range(epochs), val_acc, '--', label='validate', linewidth=1.5)
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('Loss')
    ax2.set_ylabel('Accuracy')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    setup_seed(20211108)
    saveyes = 0 # 是否写入结果、保存图片和模型
    predictyes = 0  # 是否对 test 数据集进行预测

    # for norm_layer in [nn.LayerNorm, nn.InstanceNorm2d]:
    # for model in [resnet14, resnet34]:
    for _ in [1]:
        model_ft = resnet18()
        # model_ft = resnet18(norm_layer=norm_layer)
        # model_ft = model()
        # model_ft = torchvision.models.resnet18()

        num_ftrs = model_ft.fc.in_features

        model_ft.fc = nn.Linear(num_ftrs, 100)

        model_ft = model_ft.to(device)

        criterion = nn.CrossEntropyLoss()

        optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

        # Decay LR by a factor of 0.1 every 7 epochs
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

        model_ft, total_time, acc, train_ls, val_ls, train_acc, val_acc = train_model(
            model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=25)

        if saveyes:
            with open('./result.txt', 'a') as f:
                f.writelines([
                    '\n', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                    '\n', str(total_time),
                    '\t', str(acc.item()), '\n'])
            torch.save(model_ft.state_dict(), './model/best_model_' + time.strftime("%H-%M", time.localtime()) + '.pkl')

            df = pd.DataFrame({'train_loss': train_ls,
                               'val_loss': val_ls,
                               'train_acc': train_acc,
                               'val_acc': val_acc})
            df.to_csv('./result/' + time.strftime("%H-%M", time.localtime()) + '.csv', index=False, sep=',')

        # predict images'label in test set
        if predictyes:
            for x, name in data_test:
                # name = test_list[i]
                x = x.to(device)
                x = x.view([1, 3, 64, 64])
                torch.no_grad()
                logits = model_ft(x)
                pred = logits.argmax(dim=1).item()
                with open('./test.txt', 'a') as f:
                    f.writelines([name, ' ', str(pred), '\n'])

