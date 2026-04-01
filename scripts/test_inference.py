import os
import sys
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from src.inference.predictor import load_model, predict

model_name = "resnet34"

model_files = {
    "resnet34": "lycopene_resnet34_best.pth",
    "resnet50": "lycopene_resnet50_best.pth",
    "efficientnet": "lycopene_efficientnet_best.pth",
    "customcnn": "lycopene_customcnn_best.pth",
}

model_path = os.path.join(BASE_DIR, "weights", model_files[model_name])
image_path = os.path.join(BASE_DIR, "data", "sample", "2.jpg")

LOWER_THRESH = np.array([0, 70, 0], dtype=np.uint8)
UPPER_THRESH = np.array([255, 255, 255], dtype=np.uint8)

model, device = load_model(model_name, model_path)

result = predict(
    image_path=image_path,
    model=model,
    device=device,
    lower_thresh=LOWER_THRESH,
    upper_thresh=UPPER_THRESH
)

print(f"Model: {model_name}")
print(f"Prediction: {result}")