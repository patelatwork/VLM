"""
serve_api.py

Minimal FastAPI server exposing the fine-tuned VLM as a REST endpoint.

Run:
    uvicorn serve_api:app --host 0.0.0.0 --port 8000

Then:
    curl -X POST http://localhost:8000/ask \
        -F "image=@sample.jpg" \
        -F "question=What is in this image?"

Environment variables (set before starting, or edit the defaults below):
    PROJECTOR_CKPT   path to projector.pt
    LORA_DIR         path to LoRA adapter dir (optional)
    DEVICE           "cuda" or "cpu"
"""

import io
import os

import torch
from fastapi import FastAPI, File, Form, UploadFile
from PIL import Image

from inference import load_model, answer

PROJECTOR_CKPT = os.environ.get("PROJECTOR_CKPT", "checkpoints/stage2/projector.pt")
LORA_DIR = os.environ.get("LORA_DIR", "checkpoints/stage2/lora_adapter")
DEVICE = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

app = FastAPI(title="TinyVLM API")

# Load once at startup, reused across requests
model = load_model(PROJECTOR_CKPT, LORA_DIR, DEVICE)


@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE}


@app.post("/ask")
async def ask(image: UploadFile = File(...), question: str = Form(...)):
    image_bytes = await image.read()
    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    tmp_path = "/tmp/_upload.jpg"
    pil_image.save(tmp_path)

    response_text = answer(model, tmp_path, question, DEVICE)
    return {"question": question, "answer": response_text}
