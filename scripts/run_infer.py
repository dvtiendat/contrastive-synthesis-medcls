import torch
from torchvision import models, transforms
from PIL import Image
import argparse

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models.simclr import LinearEvaluation, SimCLR

# Load pretrained model
def load_model(weights_path):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'DEVICE: {DEVICE}')

    simclr_model = SimCLR()  # Khởi tạo mô hình SimCLR phù hợp
    pretrained_model_path = "D:\THO\Bach_Khoa\Computer Vision\model\Pretrain\simclr_pretrain_ACGAN_Imagenet.pth"
    simclr_model.load_state_dict(torch.load(pretrained_model_path, map_location=DEVICE, weights_only=True))
    simclr_model.eval()

    eval_model = LinearEvaluation(simclr_model, 4).to(DEVICE)
    eval_model.load_state_dict(torch.load(weights_path, map_location=DEVICE, weights_only=True))
    eval_model.eval()
    return eval_model

# Preprocess input image
def preprocess_image(image_path):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path).convert('RGB')
    return preprocess(image).unsqueeze(0)

# Run inference
def run_inference(model, input_tensor):
    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)
    return predicted.item()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on an image using a pretrained model.")
    parser.add_argument("--weights", required=True, help="Path to the pretrained weights file.")
    parser.add_argument("--image", required=True, help="Path to the input image file.")
    args = parser.parse_args()

    model = load_model(args.weights)
    input_tensor = preprocess_image(args.image)
    prediction = run_inference(model, input_tensor)

    print(f"Predicted class: {prediction}")