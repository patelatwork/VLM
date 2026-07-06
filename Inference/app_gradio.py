"""
app_gradio.py

Quick shareable demo UI (useful for showing off results, e.g. in a Kaggle
notebook or a portfolio demo link).

Run:
    python app_gradio.py
"""

import os

import gradio as gr
import torch

from inference import load_model, answer

PROJECTOR_CKPT = os.environ.get("PROJECTOR_CKPT", "checkpoints/stage2/projector.pt")
LORA_DIR = os.environ.get("LORA_DIR", "checkpoints/stage2/lora_adapter")
DEVICE = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

model = load_model(PROJECTOR_CKPT, LORA_DIR, DEVICE)


def infer(image, question):
    image.save("/tmp/_gradio_upload.jpg")
    return answer(model, "/tmp/_gradio_upload.jpg", question, DEVICE)


demo = gr.Interface(
    fn=infer,
    inputs=[gr.Image(type="pil", label="Image"), gr.Textbox(label="Question")],
    outputs=gr.Textbox(label="Answer"),
    title="TinyVLM Demo",
)

if __name__ == "__main__":
    demo.launch(share=True)
