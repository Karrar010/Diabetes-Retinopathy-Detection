import os
from flask import Flask, request, render_template, send_file
import torch
import torchvision.transforms as transforms
from PIL import Image
import io
import base64
import matplotlib.pyplot as plt
import numpy as np
from torchvision import models
import torch.nn as nn

# Import the grad_cam function from your notebook
from hac_functions import grad_cam, load_model

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Define the same transform as in the notebook
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_prediction(image_tensor, model):
    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor.unsqueeze(0))
        _, predicted = torch.max(outputs, 1)
    return predicted.item()

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error='No file selected')
        
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', error='No file selected')
        
        if file and allowed_file(file.filename):
            # Read and process the image
            image = Image.open(file).convert('RGB')
            image_tensor = transform(image)
            
            # Get prediction and Grad-CAM
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = load_model()  # Load your trained model
            model.to(device)
            
            prediction = get_prediction(image_tensor, model)
            
            # Generate Grad-CAM visualization
            plt.figure(figsize=(8, 8))
            grad_cam(model, image_tensor)
            
            # Save the plot to a bytes buffer
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            plt.close()
            buf.seek(0)
            
            # Convert to base64 for displaying in HTML
            img_str = base64.b64encode(buf.getvalue()).decode()
            
            severity_descriptions = {
                0: "No DR (No diabetic retinopathy)",
                1: "Mild DR (Mild nonproliferative DR)",
                2: "Moderate DR (Moderate nonproliferative DR)",
                3: "Severe DR (Severe nonproliferative DR)",
                4: "Proliferative DR"
            }
            
            return render_template('index.html',
                                 prediction=severity_descriptions[prediction],
                                 grad_cam_image=img_str)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
