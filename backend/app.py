"""Optional inference service. Set MODEL_PATH to the notebook's best.pt."""
import os
from io import BytesIO
from pathlib import Path

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

from voc_detector import load_detector, predict_image

app = FastAPI(title="VOC Faster R-CNN inference")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in os.getenv("SITE_ORIGINS", "http://localhost:8000").split(",")],
    allow_methods=["POST"],
    allow_headers=["*"],
)
MODEL_PATH = Path(os.getenv("MODEL_PATH", "best.pt"))
model = None


@app.on_event("startup")
def load_model():
    global model
    if MODEL_PATH.is_file():
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = load_detector(MODEL_PATH, device)


@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    if model is None:
        raise HTTPException(503, "Trained checkpoint is unavailable.")
    if image.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(400, "Upload a JPG, PNG, or WebP image.")
    data = await image.read(10 * 1024 * 1024 + 1)
    if len(data) > 10 * 1024 * 1024:
        raise HTTPException(413, "Image exceeds 10 MB.")
    try:
        with Image.open(BytesIO(data)) as photo:
            return predict_image(model, photo, threshold=0.01)
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(400, "Could not read the image.") from exc
