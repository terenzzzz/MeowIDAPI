import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import accuracy_score
import os
import time
import copy
from tqdm import tqdm  # 导入 tqdm
from PIL import Image


# 自定义ImageFolder，跳过损坏的图像文件
class CustomImageFolder(datasets.ImageFolder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, index):
        try:
            return super().__getitem__(index)
        except (OSError, IOError, Image.UnidentifiedImageError) as e:
            print(f"Skipping corrupted file at index {index}: {self.imgs[index][0]}")
            # 递归获取下一个有效文件
            return self.__getitem__((index + 1) % len(self.imgs))


# 定义数据预处理和加载
data_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(30),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
# 加载训练、验证和测试数据集
train_dataset = CustomImageFolder(root=r'C:\Users\Terence\Desktop\split_catBreedDataSet\train', transform=data_transforms)
val_dataset = CustomImageFolder(root=r'C:\Users\Terence\Desktop\split_catBreedDataSet\val', transform=data_transforms)
test_dataset = CustomImageFolder(root=r'C:\Users\Terence\Desktop\split_catBreedDataSet\test', transform=data_transforms)

# 创建数据加载器
batch_size = 32  # 或者更大，试试不同的值
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 使用预训练模型 (比如 ResNet18)
model = models.resnet18(pretrained=True)

# 修改最后一层以适应你的数据集
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(train_dataset.classes))  # 分类数量

# 将模型移到GPU（如果可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")  # 打印当前使用的设备
model = model.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()  # 分类问题使用交叉熵损失
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# 训练和验证函数
def train_model(model, criterion, optimizer, num_epochs=25, model_save_path='best_model.pth'):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        # 每个epoch有训练和验证两个阶段
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # 设置模型为训练模式
                dataloader = train_loader
            else:
                model.eval()  # 设置模型为评估模式
                dataloader = val_loader

            running_loss = 0.0
            running_corrects = 0

            # 使用 tqdm 包装 dataloader 以添加进度条
            with tqdm(total=len(dataloader), desc=f'{phase.capitalize()} Progress', unit='batch') as pbar:
                # 遍历数据
                for inputs, labels in dataloader:
                    inputs, labels = inputs.to(device), labels.to(device)

                    # 将梯度清零
                    optimizer.zero_grad()

                    # 向前传播
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # 训练阶段，反向传播
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # 统计损失和准确率
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                    # 更新进度条
                    pbar.set_postfix(loss=running_loss / (pbar.n + 1), acc=running_corrects.double() / ((pbar.n + 1) * dataloader.batch_size))
                    pbar.update(1)

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # 记录最佳模型并保存
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                # 保存最佳模型
                torch.save(model.state_dict(), model_save_path)
                print(f"Best model saved with accuracy: {best_acc:.4f}")

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60}m {time_elapsed % 60}s')
    print(f'Best val Acc: {best_acc:.4f}')

    # 加载最佳模型权重
    model.load_state_dict(best_model_wts)
    return model

# 训练模型
best_model = train_model(model, criterion, optimizer, num_epochs=25)

# 测试模型
def test_model(model, test_loader):
    model.eval()  # 设置模型为评估模式
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 计算准确率
    accuracy = accuracy_score(all_labels, all_preds)
    print(f'Test Accuracy: {accuracy:.4f}')

# 测试模型
test_model(best_model, test_loader)
