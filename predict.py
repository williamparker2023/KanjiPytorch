import torch
from PIL import Image
import torchvision.transforms as transforms
from model import KanjiModel  # Replace with your model definition
import json
import numpy as np

with open("labelUnicode.json", "r") as f:
    data = json.load(f)

unicode_array = data["class_names"]

# Load the model
model = KanjiModel()  # Replace with your model class
model.load_state_dict(torch.load("kanji_model.pth", weights_only=True))  # Set weights_only=True
model.eval()

if torch.cuda.is_available():
    model.cuda()

# Preprocess input image
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # Replace with your training normalization
    ])
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image

# Predict Kanji
image_path = "bruhRain.png"  # Replace with your image path
image = preprocess_image(image_path)

if torch.cuda.is_available():
    image = image.cuda()

with torch.no_grad():
    output = model(image)
    top_k = 10  # Number of top predictions you want
    top_k_values, top_k_indices = torch.topk(output, top_k)

# Output the top 10 predicted kanji
for i in range(top_k):
    predicted_index = top_k_indices[0][i].item()  # Get the index of the i-th top prediction
    predicted_unicode = unicode_array[predicted_index]
    predicted_kanji = chr(int(predicted_unicode[2:], 16))  # Convert Unicode to Kanji character

    print(f"Rank {i + 1}:")
    print(f"Predicted Kanji Label: {predicted_index}")
    print(f"Predicted Kanji Unicode: {predicted_unicode}")
    print(f"Predicted Kanji Character: {predicted_kanji}")
    print("-" * 30)  # Print separator
