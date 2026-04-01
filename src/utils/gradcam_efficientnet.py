import torch
import numpy as np
import cv2

def apply_grad_cam_efficientnet(model, image, target_class, device):
    model.eval()
    image = image.to(device)

    if image.shape[1] == 3:
        image_rgb = image
    else:
        image_rgb = torch.flip(image, [1])

    outputs = model(image)
    model.zero_grad()

    one_hot_output = torch.zeros((1, outputs.size()[-1]), dtype=torch.float).to(device)
    one_hot_output[0][target_class] = 1
    outputs.backward(gradient=one_hot_output)

    gradients = model.get_activations_gradient()
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
    activations = model.get_activations().detach()

    for i in range(activations.shape[1]):
        activations[:, i, :, :] *= pooled_gradients[i]

    heatmap = torch.mean(activations, dim=1).squeeze().cpu()
    heatmap = np.maximum(heatmap.numpy(), 0)
    heatmap /= np.max(heatmap)

    heatmap = cv2.resize(heatmap, (image.shape[3], image.shape[2]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    original_image = image.squeeze().cpu().detach().numpy().transpose(1, 2, 0)
    original_image = (original_image * 255).astype(np.uint8)
    superimposed_img = heatmap * 0.4 + original_image
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)

    return superimposed_img