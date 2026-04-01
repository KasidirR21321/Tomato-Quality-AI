from fastapi import FastAPI, UploadFile, File, Form
import numpy as np
import tempfile
import os

from src.inference.predictor import load_model, predict

app = FastAPI(title="Tomato Quality AI API")

MODEL_PATHS = {
    "resnet34": "weights/lycopene_resnet34_best.pth",
    "resnet50": "weights/lycopene_resnet50_best.pth",
    "efficientnet": "weights/lycopene_efficientnet_best.pth",
    "customcnn": "weights/lycopene_customcnn_best.pth",
}

LOWER_THRESH = np.array([0, 70, 0], dtype=np.uint8)
UPPER_THRESH = np.array([255, 255, 255], dtype=np.uint8)

loaded_models = {}


@app.on_event("startup")
def startup_event():
    for model_name, model_path in MODEL_PATHS.items():
        try:
            model, device = load_model(model_name, model_path)
            loaded_models[model_name] = {
                "model": model,
                "device": device
            }
            print(f"Loaded {model_name} successfully")
        except Exception as e:
            print(f"Failed to load {model_name}: {e}")


@app.get("/")
def root():
    return {"message": "Tomato Quality AI API is running"}


@app.get("/models")
def get_models():
    return {
        "available_models": list(MODEL_PATHS.keys()),
        "loaded_models": list(loaded_models.keys())
    }


@app.post("/predict")
async def predict_image(
    model_name: str = Form(...),
    file: UploadFile = File(...)
):
    if model_name not in MODEL_PATHS:
        return {
            "error": f"Invalid model_name: {model_name}",
            "available_models": list(MODEL_PATHS.keys())
        }

    if model_name not in loaded_models:
        return {
            "error": f"Model {model_name} was not loaded"
        }

    temp_path = None

    try:
        file_bytes = await file.read()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(file_bytes)
            temp_path = tmp.name

        model = loaded_models[model_name]["model"]
        device = loaded_models[model_name]["device"]

        prediction = predict(
            image_path=temp_path,
            model=model,
            device=device,
            lower_thresh=LOWER_THRESH,
            upper_thresh=UPPER_THRESH
        )

        return {
            "model": model_name,
            "filename": file.filename,
            "prediction": float(prediction)
        }

    except Exception as e:
        return {
            "error": str(e)
        }

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)