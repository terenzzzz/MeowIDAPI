# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 18:00:28 2025

@author: Terence
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from ImgClassifier import ImgClassifier
import os
from tempfile import NamedTemporaryFile
import numpy as np  # 导入 numpy

app = Flask(__name__)
CORS(app)  # 启用 CORS，允许跨域请求

# 允许的文件扩展名
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# 初始化 ImgClassifier 类
model_path = 'best_model.pth'  # 你保存的训练模型路径
dataset_dir = r'C:\Users\Terence\Desktop\split_catBreedDataSet\train'  # 你的数据集根目录
classifier = ImgClassifier(model_path, dataset_dir)

# 用于将 numpy 数值类型转换为可以 JSON 序列化的类型
def convert_to_serializable(obj):
    if isinstance(obj, np.generic):
        return obj.item()  # 转换为标准的 Python 类型
    raise TypeError(f"Type {type(obj)} not serializable")

@app.route('/predictBreed', methods=['POST'])
def predictBreed():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
        
        if file and allowed_file(file.filename):
            # 创建一个临时文件来保存上传的图像
            with NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
                file.save(temp_file.name)
                temp_file_path = temp_file.name  # 获取临时文件的路径
                
                # 确保文件完全写入
                temp_file.close()
                
                # 使用分类器进行预测
                predicted_classes = classifier.predict_image(temp_file_path, top_k=4)  # 调用分类器预测
                
                # 删除临时文件
                try:
                    os.remove(temp_file_path)
                except Exception as e:
                    print(f"Error deleting temp file: {e}")
                
            # 将预测结果中的英文类别、中文类别以及概率格式化，并确保概率为 Python 原生 float 类型
            predicted_classes_serializable = [
                [cls_en, cls_cn, round(float(prob), 4)]  # 将 prob 转换为 Python 原生 float 类型
                for (cls_en, cls_cn, prob) in predicted_classes
            ]
            
            # 返回预测结果
            return jsonify({
                'message': 'File successfully uploaded and processed',
                'predictions': predicted_classes_serializable
            }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='localhost', port=5000)

