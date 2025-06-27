import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm
import numpy as np
from PIL import Image
from torchvision.datasets import ImageFolder
import albumentations as A
from albumentations.pytorch import ToTensorV2
import random
import csv

# 配置
train_dir = 'train'
test_dir = 'test'
batch_size = 32
epochs = 32
lr = 0.001
num_workers = 2
save_path = 'resnet100.pth'

if torch.cuda.is_available():
    print(f"CUDA 可用，当前设备：{torch.cuda.get_device_name(0)}")
    device = torch.device('cuda')
else:
    print("CUDA 不可用，使用 CPU 训练")
    device = torch.device('cpu')

class AlbumentationsDataset(ImageFolder):
    def __init__(self, root, transform=None):
        super().__init__(root)
        self.transform = transform

    def __getitem__(self, index):
        path, target = self.samples[index]
        image = Image.open(path).convert('RGB')
        image = np.array(image)
        if self.transform:
            image = self.transform(image=image)['image']
        return image, target

def main():
    # 数据增强与预处理
    train_transform = A.Compose([
        A.Resize(224, 224),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.CoarseDropout(max_holes=2, max_height=32, max_width=32, p=0.3),
        A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    test_transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    # 数据集
    train_dataset = AlbumentationsDataset('train', transform=train_transform)
    test_dataset = AlbumentationsDataset('test', transform=test_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # 类别数
    num_classes = len(train_dataset.classes)
    print('类别:', train_dataset.classes)

    # 模型（升级为ResNet50，使用ImageNet预训练权重）
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    # 损失与优化器
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    # 训练与验证（含早停）
    best_acc = 0.0
    patience = 5
    counter = 0
    history = []  # 用于保存loss和acc
    for epoch in range(epochs):
        # 每个epoch动态采样每类400张
        sampled_indices = []
        class_to_indices = {i: [] for i in range(len(train_dataset.classes))}
        for idx, (_, label) in enumerate(train_dataset.samples):
            class_to_indices[label].append(idx)
        for label, indices in class_to_indices.items():
            if len(indices) >= 400:
                sampled = random.sample(indices, 400)
            else:
                sampled = random.choices(indices, k=400)  # 不足则有放回采样
            sampled_indices.extend(sampled)
        epoch_train_dataset = Subset(train_dataset, sampled_indices)
        epoch_train_loader = DataLoader(epoch_train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        model.train()
        running_loss = 0.0
        train_bar = tqdm(epoch_train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]')
        for imgs, labels in train_bar:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
            train_bar.set_postfix(loss=loss.item())
        epoch_loss = running_loss / len(epoch_train_loader.dataset)
        scheduler.step()

        # 验证
        model.eval()
        correct = 0
        total = 0
        val_bar = tqdm(test_loader, desc=f'Epoch {epoch+1}/{epochs} [Val]')
        with torch.no_grad():
            for imgs, labels in val_bar:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        acc = correct / total
        print(f'Epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.4f} | Val Acc: {acc:.4f}')
        history.append({'epoch': epoch+1, 'loss': epoch_loss, 'val_acc': acc})
        # 保存最优模型
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), 'best_resnet50.pth')
            counter = 0
            print('保存最优模型！')
        else:
            counter += 1
            if counter >= patience:
                print('早停！')
                break

    # 保存loss和精度到csv
    with open('train_history.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['epoch', 'loss', 'val_acc'])
        writer.writeheader()
        for row in history:
            writer.writerow(row)

    print('训练完成，最优验证准确率：', best_acc)

if __name__ == '__main__':
    main() 