import torch
import numpy as np

from src.preprocessing.image_processor import preprocess_data


def load_model(model_name, model_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if model_name == "resnet34":
        from src.models.resnet34_model import ModifiedResNet34
        model = ModifiedResNet34(num_outputs=1, dropout_rate=0.5)

    elif model_name == "resnet50":
        from src.models.resnet50_model import ModifiedResNet50
        model = ModifiedResNet50(num_outputs=1, dropout_rate=0.5)

    elif model_name == "efficientnet":
        from src.models.efficientnet_model import ModifiedEfficientNetB0
        model = ModifiedEfficientNetB0(num_outputs=1, dropout_rate=0.5)

    elif model_name == "customcnn":
        from src.models.custom_cnn_model import CustomModel
        model = CustomModel(num_outputs=1)

    else:
        raise ValueError("Invalid model_name")

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    model = model.to(device)
    model.eval()

    return model, device


def preprocess_image(image_path, lower_thresh, upper_thresh):
    img_flat = preprocess_data(
        image_path=image_path,
        target_size=(224, 224),
        lower_thresh=lower_thresh,
        upper_thresh=upper_thresh,
        save_path=None
    )

    img = img_flat.reshape(224, 224, 3)
    img = np.transpose(img, (2, 0, 1))

    image_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0)

    return image_tensor


def predict(image_path, model, device, lower_thresh, upper_thresh):
    image = preprocess_image(image_path, lower_thresh, upper_thresh)
    image = image.to(device)

    with torch.no_grad():
        output = model(image)
        prediction = output.squeeze().item()

    return prediction