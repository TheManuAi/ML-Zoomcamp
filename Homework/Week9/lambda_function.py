# Docker commands:
# Build: docker build -t hair-classifier .
# Check size: docker images hair-classifier
# Run: docker run -it --rm -p 8080:8080 hair-classifier
# Test (in another terminal): curl -XPOST "http://localhost:8080/2015-03-31/functions/function/invocations" -d '{"url": "https://habrastorage.org/webt/yf/_d/ok/yf_dokzqy3vcritme8ggnzqlvwa.jpeg"}'

import numpy as np
import onnxruntime as rt
from io import BytesIO
from urllib import request
from PIL import Image

MODEL_FILE = "hair_classifier_v1.onnx"

session = rt.InferenceSession(MODEL_FILE)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img

def prepare_image(img, target_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img

def preprocess(url):
    img = download_image(url)
    target_size = (200, 200)
    img = prepare_image(img, target_size)
    
    x = np.array(img, dtype='float32')
    x = x / 255.0
    
    mean = np.array([0.485, 0.456, 0.406], dtype='float32')
    std = np.array([0.229, 0.224, 0.225], dtype='float32')
    x = (x - mean) / std
    
    X = np.array([x])
    X = X.transpose(0, 3, 1, 2)
    return X

def predict(url):
    X = preprocess(url)
    preds = session.run([output_name], {input_name: X})
    return float(preds[0][0][0])

def lambda_handler(event, context):
    url = event['url']
    result = predict(url)
    return result

if __name__ == "__main__":
    url = "https://habrastorage.org/webt/yf/_d/ok/yf_dokzqy3vcritme8ggnzqlvwa.jpeg"
    print(predict(url))
