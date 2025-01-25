# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 18:00:28 2025

@author: Terence
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from predict import get_cat_breed
import os

app = Flask(__name__)
CORS(app)  # 启用 CORS，允许跨域请求

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'jpg', 'jpeg', 'png'}

@app.route('/predictBreed', methods=['POST'])
def predictBreed():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        # 验证文件名是否合法
        if allowed_file(file.filename):  # 假设你已经定义了 allowed_file 函数
            img_bytes = file.read()
            prediction_name = get_cat_breed(img_bytes)

            # 返回预测结果
            return jsonify({
                'message': 'File successfully uploaded and processed',
                'prediction': prediction_name
            }), 200

        else:
            return jsonify({"error": "Unsupported file type"}), 400

    except Exception as e:
        return jsonify({"error": str(e)}), 500



if __name__ == '__main__':
    # 如果在开发环境中，使用 Flask 自带的服务器
    if os.getenv('FLASK_ENV') == 'development':
        app.run(host='localhost', port=5000)
    else:
        # 在生产环境中，通过 gunicorn 启动 Flask
        app.run(host='0.0.0.0', port=5000)  # 这里可以选择其他端口

