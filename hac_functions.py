import torch
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision import models
import torch.nn as nn

def load_model():
    # Initialize the model
    model = models.resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 5)  # 5 classes for DR severity
    
    # Load the trained weights
    model.load_state_dict(torch.load('best_model.pth', map_location=torch.device('cpu')))
    return model

def grad_cam(model, image_tensor, class_idx=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    image_tensor = image_tensor.unsqueeze(0).to(device)

    feature_maps = None
    gradients = None

    def forward_hook(module, input, output):
        nonlocal feature_maps
        feature_maps = output

    def backward_hook(module, grad_in, grad_out):
        nonlocal gradients
        gradients = grad_out[0]

    layer = model.layer4[-1]
    layer.register_forward_hook(forward_hook)
    layer.register_backward_hook(backward_hook)

    output = model(image_tensor)
    if class_idx is None:
        class_idx = torch.argmax(output).item()

    model.zero_grad()
    output[0, class_idx].backward()

    weights = gradients.mean(dim=[2, 3], keepdim=True)
    cam = (weights * feature_maps).sum(dim=1, keepdim=True)
    cam = F.relu(cam)

    cam = cam.detach().squeeze().cpu().numpy()
    cam = cv2.resize(cam, (224, 224))
    cam = (cam - cam.min()) / (cam.max() - cam.min())

    image = image_tensor.cpu().squeeze().permute(1, 2, 0).numpy()
    image = (image - image.min()) / (image.max() - image.min())

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam_image = heatmap + np.float32(image)
    cam_image = cam_image / cam_image.max()

    plt.imshow(cam_image)
    plt.title(f"Grad-CAM for Class {class_idx}")
    plt.axis("off") 