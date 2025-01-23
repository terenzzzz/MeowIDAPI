import torch
import io
from PIL import Image
from torchvision.models import ResNet50_Weights
import random
import numpy as np

model_path = "model.pt"

class_names = {
    0: {'en': 'Abyssinian', 'zh': '阿比西尼亚猫'},
    1: {'en': 'American Curl', 'zh': '美国卷耳猫'},
    2: {'en': 'American Shorthair', 'zh': '美国短毛猫'},
    3: {'en': 'Balinese', 'zh': '巴厘猫'},
    4: {'en': 'Bengal', 'zh': '孟加拉猫'},
    5: {'en': 'Birman', 'zh': '缅甸猫'},
    6: {'en': 'Bombay', 'zh': '孟买猫'},
    7: {'en': 'British Shorthair', 'zh': '英国短毛猫'},
    8: {'en': 'Burmese', 'zh': '缅甸猫'},
    9: {'en': 'Cornish Rex', 'zh': '康沃尔卷毛猫'},
    10: {'en': 'Devon Rex', 'zh': '德文卷毛猫'},
    11: {'en': 'Egyptian Mau', 'zh': '埃及猫'},
    12: {'en': 'Exotic Shorthair', 'zh': '异国短毛猫'},
    13: {'en': 'Extra-Toes Cat - Hemingway Polydactyl', 'zh': '海明威多趾猫'},
    14: {'en': 'Havana', 'zh': '哈瓦那猫'},
    15: {'en': 'Himalayan', 'zh': '喜马拉雅猫'},
    16: {'en': 'Japanese Bobtail', 'zh': '日本短尾猫'},
    17: {'en': 'Korat', 'zh': '科拉特猫'},
    18: {'en': 'Maine Coon', 'zh': '缅因猫'},
    19: {'en': 'Manx', 'zh': '曼岛猫'},
    20: {'en': 'Nebelung', 'zh': '尼贝龙猫'},
    21: {'en': 'Norwegian Forest Cat', 'zh': '挪威森林猫'},
    22: {'en': 'Oriental Short Hair', 'zh': '东方短毛猫'},
    23: {'en': 'Persian', 'zh': '波斯猫'},
    24: {'en': 'Ragdoll', 'zh': '布偶猫'},
    25: {'en': 'Russian Blue', 'zh': '俄罗斯蓝猫'},
    26: {'en': 'Scottish Fold', 'zh': '苏格兰折耳猫'},
    27: {'en': 'Selkirk Rex', 'zh': '塞尔凯克卷毛猫'},
    28: {'en': 'Siamese', 'zh': '暹罗猫'},
    29: {'en': 'Siberian', 'zh': '西伯利亚猫'},
    30: {'en': 'Snowshoe', 'zh': '雪鞋猫'},
    31: {'en': 'Sphynx', 'zh': '斯芬克斯猫'},
    32: {'en': 'Tonkinese', 'zh': '东奇尼猫'},
    33: {'en': 'Toyger tiger cat', 'zh': '玩具虎猫'},
    34: {'en': 'Turkish Angora', 'zh': '土耳其安哥拉猫'}
}

def get_cat_breed(img_bytes):
    # Set random seed for reproducibility
    random_seed = 2023
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.load(model_path, map_location=torch.device(dev))
    model.to(dev)
    model.eval()

    tensor = transform_image(img_bytes).to(dev)
    outputs = model.forward(tensor)
    probabilities = torch.nn.functional.softmax(outputs, dim=1)  # Calculate probabilities
    top_probs, top_indices = torch.topk(probabilities, 4)  # Get top 4 probabilities and indices

    # Map indices to class names and convert to a list of tuples
    results = [{
        'en': class_names[int(idx)]['en'],
        'zh': class_names[int(idx)]['zh'],
        'probability': float(prob)
    } for idx, prob in zip(top_indices[0], top_probs[0])]
    return results
      
def transform_image(im):  
    weights = ResNet50_Weights.DEFAULT
    preprocess = weights.transforms()
    image = Image.open(io.BytesIO(im)).convert('RGB')
    return preprocess(image).unsqueeze(0)

# 从本地加载图片并进行预测
def predict_local_image(image_path):
    # 读取本地图片为 bytes
    with open(image_path, "rb") as f:
        img_bytes = f.read()
    # 调用 get_cat_breed 方法获取预测结果
    prediction_results = get_cat_breed(img_bytes)
    return prediction_results


# Example usage
if __name__ == "__main__":
    image_path = r"C:\Users\Terence\Desktop\test.jpg"  # Replace with your image path
    results = predict_local_image(image_path)
    print("Top 4 Predictions with Probabilities:")
    for result in results:
        print(f"{result['zh']} ({result['en']}): {result['probability']:.4f}")
