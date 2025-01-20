# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 15:43:14 2025

@author: Terence
"""

import torch

# 检查 PyTorch 是否能够识别到 GPU
print(torch.cuda.is_available())  # 如果是 True，说明 GPU 可用
print(torch.cuda.current_device())  # 查看当前 GPU 设备编号

# 查看当前 GPU 的详细信息
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.get_device_name(device))
print(f"Total memory: {torch.cuda.get_device_properties(device).total_memory / 1024**3} GB")
