"""
app_gradio.py

Demo UI. Pulls weights from the Hub when HF_REPO_ID is set, so this file can be
dropped into a Space alongside model.py / dataset.py / inference.py unchanged.

Env:
    HF_REPO_ID       e.g. dhruvpatel93/tinyvlm-vqa   (preferred)
    PROJECTOR_CKPT   local path, used when HF_REPO_ID is unset
    LORA_DIR         local path, used when HF_REPO_ID is unset
"""

import os
import sys
from pathlib import Path

import gradio as gr
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "Model"))

from inference import load_model, answer   # noqa: E402

DEVICE = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
REPO_ID = os.environ.get("HF_REPO_ID")

if REPO_ID:
    from huggingface_hub import hf_hub_download, snapshot_download

    projector_ckpt = hf_hub_download(repo_id=REPO_ID, filename="projector.pt")
    lora_dir = str(Path(snapshot_download(repo_id=REPO_ID, allow_patterns=["lora_adapter/*"])) / "lora_adapter")
else:
    projector_ckpt = os.environ.get("PROJECTOR_CKPT", "checkpoints/stage2/projector.pt")
    lora_dir = os.environ.get("LORA_DIR", "checkpoints/stage2/lora_adapter")
    if not Path(projector_ckpt).exists():
        # In a Space this always means HF_REPO_ID was never set, so the local
        # branch ran and reported a training-machine path that was never going
        # to exist. Say that, rather than raising FileNotFoundError on it.
        raise RuntimeError(
            f"No weights: HF_REPO_ID is unset and {projector_ckpt!r} does not exist.\n"
            "In a Space, set the HF_REPO_ID variable (Settings > Variables and secrets) "
            "to the repo holding projector.pt + lora_adapter/, e.g. you/tinyvlm-vqa. "
            "`python deploy_space.py --space_id ... --model_repo ...` sets it for you.\n"
            "Locally, set PROJECTOR_CKPT and LORA_DIR instead."
        )

model = load_model(projector_ckpt, lora_dir, DEVICE)


def infer(image, question, mode, max_new_tokens, temperature):
    if image is None:
        return "Upload an image first."
    if not (question or "").strip():
        question = "Describe this image."
    # Pass the PIL image straight through -- the old version wrote every
    # request to a fixed /tmp path, which races between concurrent users.
    return answer(
        model,
        image,
        question.strip(),
        DEVICE,
        max_new_tokens=int(max_new_tokens),
        short_answer=(mode == "Short answer (VQA)"),
        temperature=float(temperature),
    )


demo = gr.Interface(
    fn=infer,
    inputs=[
        gr.Image(type="pil", label="Image"),
        gr.Textbox(label="Question", placeholder="Describe this image."),
        gr.Radio(
            ["Descriptive", "Short answer (VQA)"],
            value="Descriptive",
            label="Answer style",
            info="Short answer appends the VQA hint the model was trained with.",
        ),
        gr.Slider(16, 256, value=128, step=16, label="Max new tokens"),
        gr.Slider(0.0, 1.2, value=0.0, step=0.1, label="Temperature (0 = greedy)"),
    ],
    outputs=gr.Textbox(label="Answer", lines=6),
    title="TinyVLM",
    description="SigLIP2 + Qwen2.5-0.5B, LLaVA-style. Ask about an image.",
)

if __name__ == "__main__":
    demo.launch(share=True)
