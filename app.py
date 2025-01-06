from flask import Flask, render_template, request, jsonify
import torch
from torchvision import transforms
from PIL import Image
import io

app = Flask(__name__)

# Load the pre-trained model
model = torch.load('models/kanji_model.pth')  # Update with the actual model path
model.eval()

@app.route('/')
def index():
    return render_template('index.html')  # The homepage of your app

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the image from the form
        file = request.files['image']
        image = Image.open(file.stream)

        # Preprocess the image
        preprocess = transforms.Compose([
            transforms.Resize(64),  # Resize image to match model input
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Example normalization
        ])
        image = preprocess(image).unsqueeze(0)

        # Make the prediction
        with torch.no_grad():
            output = model(image)
            _, predicted = torch.max(output, 1)

        # Return the prediction
        return jsonify({'prediction': str(predicted.item())})

if __name__ == '__main__':
    app.run(debug=True)
