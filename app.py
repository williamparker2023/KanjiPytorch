import torch
from flask import Flask, render_template, request, jsonify
from model import KanjiModel  # Replace with your model definition
from PIL import Image
import io
import base64
import json
import numpy as np
import torchvision.transforms as transforms

app = Flask(__name__)

# Load your model
model = KanjiModel()  # Replace with your model class
model.load_state_dict(torch.load("kanji_model.pth", weights_only=True))
model.eval()

with open("labelUnicode.json", "r") as f:
    data = json.load(f)
unicode_array = data["class_names"]

if torch.cuda.is_available():
    model.cuda()

# Preprocess the image
def preprocess_image(image_data):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])
    
    image = Image.open(io.BytesIO(image_data))
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    image_data = base64.b64decode(data['image'].split(',')[1])  # Extract image data from base64
    image = preprocess_image(image_data)

    if torch.cuda.is_available():
        image = image.cuda()

    with torch.no_grad():
        output = model(image)
        _, top_indices = torch.topk(output, 10)  # Get the top 10 predictions
        top_indices = top_indices[0].cpu().numpy()

    # Get the kanji characters for the top predictions
    predictions = [chr(int(unicode_array[i][2:], 16)) for i in top_indices]

    return jsonify({'predictions': predictions})

if __name__ == '__main__':
    app.run(debug=True)
