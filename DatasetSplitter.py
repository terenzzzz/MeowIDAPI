# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 20:16:52 2025

@author: Terence
"""

import os
import shutil
from sklearn.model_selection import train_test_split
from tqdm import tqdm  # 导入 tqdm 库

class DatasetSplitter:
    def __init__(self, source_dir, target_dir, test_size=0.2, val_size=0.1, random_state=42, min_samples=2):
        """
        初始化数据集划分器

        :param source_dir: 原始数据集所在的目录，包含子文件夹，每个子文件夹表示一个类别
        :param target_dir: 划分后的目标目录，存储 train, val 和 test 子文件夹
        :param test_size: 测试集的比例
        :param val_size: 验证集的比例
        :param random_state: 随机种子，确保可重复划分
        :param min_samples: 最小样本数，少于该数量的类别将跳过或单独处理
        """
        self.source_dir = source_dir
        self.target_dir = target_dir
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        self.min_samples = min_samples

    def _create_dirs(self, class_names):
        """
        创建目标目录和类别子目录
        :param class_names: 类别名称列表
        """
        os.makedirs(self.target_dir, exist_ok=True)
        for split in ['train', 'val', 'test']:
            split_dir = os.path.join(self.target_dir, split)
            os.makedirs(split_dir, exist_ok=True)
            for class_name in class_names:
                os.makedirs(os.path.join(split_dir, class_name), exist_ok=True)

    def _split_class_images(self, class_dir):
        """
        将某个类别的图像文件划分为训练集、验证集和测试集
        :param class_dir: 类别目录
        :return: 划分后的图像文件路径
        """
        images = [os.path.join(class_dir, img) for img in os.listdir(class_dir)]
        
        # 检查图片数量是否足够
        if len(images) < self.min_samples:
            print(f"警告：类别 {class_dir} 中的图片数量少于 {self.min_samples} 张，跳过此类别。")
            return [], [], []

        train_imgs, test_imgs = train_test_split(images, test_size=self.test_size, random_state=self.random_state)
        train_imgs, val_imgs = train_test_split(train_imgs, test_size=self.val_size, random_state=self.random_state)
        return train_imgs, val_imgs, test_imgs

    def _move_images(self, image_list, target_dir):
        """
        将图像移动到目标目录
        :param image_list: 要移动的图像路径列表
        :param target_dir: 目标目录
        """
        for img_path in image_list:
            shutil.move(img_path, os.path.join(target_dir, os.path.basename(img_path)))
            
    def _copy_images(self, image_list, target_dir):
        """
        将图像复制到目标目录
        :param image_list: 要复制的图像路径列表
        :param target_dir: 目标目录
        """
        for img_path in image_list:
            shutil.copy(img_path, os.path.join(target_dir, os.path.basename(img_path)))

    def split(self):
        """
        划分数据集并将图像复制到目标目录
        """
        # 获取所有类别
        class_names = os.listdir(self.source_dir)
        self._create_dirs(class_names)
    
        # 使用 tqdm 显示进度条
        for class_name in tqdm(class_names, desc="Processing classes", unit="class"):
            class_dir = os.path.join(self.source_dir, class_name)
            if os.path.isdir(class_dir):
                # 对每个类别的图像进行划分
                train_imgs, val_imgs, test_imgs = self._split_class_images(class_dir)
    
                if train_imgs and val_imgs and test_imgs:
                    # 将划分后的图像复制到目标文件夹
                    self._copy_images(train_imgs, os.path.join(self.target_dir, 'train', class_name))
                    self._copy_images(val_imgs, os.path.join(self.target_dir, 'val', class_name))
                    self._copy_images(test_imgs, os.path.join(self.target_dir, 'test', class_name))
    
        print("数据集划分并复制完成！")

# 通过 if __name__ == "__main__": 确保只有在直接运行此文件时才会执行
if __name__ == "__main__":
    source_dir = r'C:\Users\Terence\Desktop\images'  # 使用原始字符串
    target_dir = r'C:\Users\Terence\Desktop\split_catBreedDataSet'  # 使用原始字符串

    # 创建 DatasetSplitter 实例
    splitter = DatasetSplitter(source_dir, target_dir, test_size=0.2, val_size=0.1)

    # 开始划分数据集
    splitter.split()
