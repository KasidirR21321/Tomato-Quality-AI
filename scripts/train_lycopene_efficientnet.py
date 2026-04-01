import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import cv2

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, "src")
sys.path.append(SRC_DIR)

from data_loader.data_loader import load_data
from models.efficientnet_model import ModifiedEfficientNetB0
from training.train import train_model
from evaluation.evaluate import evaluate_model_and_save_results
from utils.gradcam_efficientnet import apply_grad_cam_efficientnet

TARGET_NAME = "Lycopene"

processed_data_dir = os.path.join(BASE_DIR, "data", "processed", TARGET_NAME)
base_results_dir = os.path.join(BASE_DIR, "results", TARGET_NAME)

os.makedirs(base_results_dir, exist_ok=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

train_loader, val_loader, test_loader = load_data(processed_data_dir, device, batch_size=32)

model = ModifiedEfficientNetB0(num_outputs=1, dropout_rate=0.5).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

results_dir = train_model(
    model,
    criterion,
    optimizer,
    train_loader,
    val_loader,
    num_epochs=500,
    base_results_dir=base_results_dir,
    patience=30
)

evaluate_model_and_save_results(model, criterion, test_loader, results_dir)

n = 10
model.eval()

counter = 0
for i, (images, _) in enumerate(test_loader):
    if counter >= n:
        break

    for j, image in enumerate(images):
        if counter >= n:
            break

        image = image.unsqueeze(0).to(device)

        if image.shape[1] != 3:
            image = image.permute(0, 3, 1, 2)

        output = model(image)
        _, predicted = torch.max(output, 1)
        target_class = predicted.item()

        superimposed_img = apply_grad_cam_efficientnet(model, image, target_class, device)

        grad_cam_path = os.path.join(results_dir, f'grad_cam_result_image_{counter + 1}.png')
        plt.imsave(grad_cam_path, superimposed_img)
        print(f"Grad-CAM image saved to {grad_cam_path}")

        counter += 1

    original_image = images[0].cpu().detach().numpy()
    original_image = original_image.transpose(1, 2, 0)

    if original_image.shape[2] != 3:
        print("Warning: The image does not have 3 channels.")
    else:
        if original_image.max() <= 1:
            original_image = (original_image * 255).astype('uint8')

        original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        original_image_display = original_image_rgb.astype('float32') / 255.0

        original_image_save_path = os.path.join(results_dir, f'corrected_original_image_{i+1}.png')
        plt.imsave(original_image_save_path, original_image_display)
        print(f"Corrected original image saved to {original_image_save_path}")

print(f"{n} pairs of Grad-CAM and original images have been saved to {results_dir}")