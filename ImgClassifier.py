# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 15:39:46 2025

@author: Terence
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import torch.nn.functional as F
import os

class ImgClassifier:
    def __init__(self, model_path, dataset_dir, device=None):
        """
        初始化 ImgClassifier 类，加载训练好的模型。
        
        :param model_path: 训练好的模型权重文件路径
        :param dataset_dir: 数据集根目录，包含各个类别的子文件夹
        :param device: 计算设备，默认会自动选择 GPU 或 CPU
        """
        # 选择设备，如果没有 GPU，则使用 CPU
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # 创建模型结构
        self.model = models.resnet18(pretrained=False)  # 不加载预训练权重
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, len(os.listdir(dataset_dir)))  # 根据类别数量调整最后一层

        # 加载训练好的模型权重
        self.model.load_state_dict(torch.load(model_path))
        self.model = self.model.to(self.device)  # 移动模型到设备
        self.model.eval()  # 设置模型为评估模式

        # 定义数据预处理（和训练时相同）
        self.data_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # 加载类别标签（通过文件夹名来获取类别名称）
        self.idx_to_label = {i: folder for i, folder in enumerate(sorted(os.listdir(dataset_dir)))}

    def predict_image(self, image_path, top_k=4):
        """
        对单张图像进行预测，并返回概率最高的几个类别
        
        :param image_path: 图像文件的路径
        :param top_k: 返回前 K 个类别及其概率
        :return: 预测的类别名称和对应的概率
        """
        # 打开图像并进行预处理
        image = Image.open(image_path)
        image = self.data_transforms(image)
        
        # 增加 batch 维度：模型需要的是 [batch_size, channels, height, width]
        image = image.unsqueeze(0).to(self.device)

        # 进行前向传播并获取预测结果
        with torch.no_grad():
            outputs = self.model(image)

        # 获取输出的概率分布
        probabilities = F.softmax(outputs, dim=1)

        # 获取 top_k 的概率和对应的类别索引
        top_probs, top_indices = torch.topk(probabilities, top_k)

        # 将概率和索引转换为列表
        top_probs = top_probs.squeeze().cpu().numpy()
        top_indices = top_indices.squeeze().cpu().numpy()

        # 获取预测类别的名称
        top_classes = [self.idx_to_label[idx] for idx in top_indices]

        # 返回类别和对应的概率
        return list(zip(top_classes, top_probs))

# 使用示例
if __name__ == "__main__":
    # 初始化 ImgClassifier 类
    model_path = 'best_model.pth'  # 你保存的训练模型路径
    dataset_dir = r'C:\Users\Terence\Desktop\split_catBreedDataSet\train'  # 你的数据集根目录
    classifier = ImgClassifier(model_path, dataset_dir)

    # 预测单张图片
    image_path = r'C:\Users\Terence\Desktop\test.jpg'
    predicted_classes = classifier.predict_image(image_path, top_k=4)
    
    print(f"Top 4 predicted classes for image {image_path}:")
    for i, (cls, prob) in enumerate(predicted_classes, 1):
        print(f"{i}. {cls}: {prob:.4f}")

