import os
from PIL import Image, ImageOps
import numpy as np
import random
from tqdm import tqdm

class ImgProcessor:

    def _resize(self, image, resize_size=(224, 224)):
        """
        等比例缩放图像到目标尺寸，同时保持纵横比
        :param image: 输入图像
        :return: 调整后的图像
        """
        # 获取原始图像尺寸
        width, height = image.size

        # 计算缩放比例
        ratio = min(resize_size[0] / width, resize_size[1] / height)

        # 计算新的尺寸
        new_width = int(width * ratio)
        new_height = int(height * ratio)

        # 等比例缩放图像
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # 创建一个新的空白图像，背景为透明
        new_image = Image.new("RGB", resize_size, (0, 0, 0))  # 白色背景

        # 计算填充的位置，将缩放后的图像粘贴到背景图上
        paste_x = (resize_size[0] - new_width) // 2
        paste_y = (resize_size[1] - new_height) // 2
        new_image.paste(image, (paste_x, paste_y))

        return new_image



    def process(self, image_path):
        """
        处理单张图像，调整其大小并保存为原格式
        :param image_path: 图像文件的路径
        :return: 无返回，直接保存覆盖原文件
        """
        try:
            # 打开图像
            image = Image.open(image_path)
    
            # 调整图像大小
            image = self._resize(image)
    
            # 保存为原格式
            image.save(image_path)
    
        except Exception as e:
            print(f"无法处理图像 {image_path}: {e}")

    def process_dataset(self, dataset_path):
        """
        处理整个数据集，处理每个品种文件夹中的图像
        :param dataset_path: 数据集的根目录，包含各个品种的文件夹
        """
        # 遍历数据集目录中的每个品种文件夹
        for breed_folder in tqdm(os.listdir(dataset_path), desc="Processing dataset", unit="folder"):
            breed_folder_path = os.path.join(dataset_path, breed_folder)

            # 确保是文件夹
            if os.path.isdir(breed_folder_path):
                # 遍历每个品种文件夹中的图像文件
                for image_name in os.listdir(breed_folder_path):
                    image_path = os.path.join(breed_folder_path, image_name)

                    # 确保是图像文件（例如，可以根据文件扩展名来判断）
                    if image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                        # 处理图像
                        self.process(image_path)

# 通过 if __name__ == "__main__": 确保只有在直接运行此文件时才会执行
if __name__ == "__main__":
    dataset_path = r'C:\Users\Terence\Desktop\images'  # 数据集路径

    # 创建 ImgProcessor 实例
    imgProcessor = ImgProcessor()

    # 处理整个数据集
    imgProcessor.process_dataset(dataset_path)

    print("数据集处理完成！")
