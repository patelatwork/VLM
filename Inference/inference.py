"""
inference.py

Load base models + trained projector + LoRA adapter and answer questions
about an image.

Two things here are load-bearing:

*   The prompt is built by `dataset.build_prompt`, the same function the
    training data used. In the first version the training code and the
    inference code each rendered the chat template their own way; they happened
    to agree for Qwen, but nothing enforced it.

*   `pool_stride` is recovered from the projector checkpoint's own shape rather
    than passed in. A mismatch here silently changes how many visual tokens the
    model expects and produces garbage, so it should not be a flag a human can
    get wrong.

Usage:
    python inference.py --image cat.jpg --question "What is in this image?" \
        --projector_ckpt ckpt/stage2/projector.pt --lora_dir ckpt/stage2/lora_adapter
"""

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image

for _p in (Path(__file__).resolve().parent.parent / "Model", Path(__file__).resolve().parent):
    sys.path.insert(0, str(_p))

from model import TinyVLM, IMAGE_TOKEN            # noqa: E402
from dataset import build_prompt                  # noqa: E402


def infer_pool_stride(state_dict, vision_dim: int = 768) -> int:
    """The projector's first Linear has in_features = vision_dim * stride^2."""
    w = state_dict.get("net.0.weight")
    if w is None:
        return 2
    stride = round((w.shape[1] / vision_dim) ** 0.5)
    if vision_dim * stride * stride != w.shape[1]:
        raise ValueError(
            f"cannot derive pool_stride: projector expects {w.shape[1]} input features, "
            f"which is not vision_dim({vision_dim}) * k^2"
        )
    return stride


def load_model(projector_ckpt: str, lora_dir: str = None, device: str = "cuda", dtype=torch.float32):
    sd = torch.load(projector_ckpt, map_location="cpu")
    pool_stride = infer_pool_stride(sd)

    model = TinyVLM(freeze_vision=True, freeze_llm=True, pool_stride=pool_stride, dtype=dtype)
    model.projector.load_state_dict(sd)

    if lora_dir:
        from peft import PeftModel
        model.llm = PeftModel.from_pretrained(model.llm, lora_dir)
        model.llm = model.llm.merge_and_unload()   # fold LoRA in; faster inference

    model = model.to(device).eval()
    print(f"loaded: pool_stride={pool_stride}, visual tokens={model.num_visual_tokens}")
    return model


@torch.no_grad()
def answer(
    model,
    image,
    question: str,
    device: str = "cuda",
    max_new_tokens: int = 128,
    short_answer: bool = False,
    temperature: float = 0.0,
    repetition_penalty: float = 1.05,
) -> str:
    """`image` may be a path or a PIL.Image."""
    if not isinstance(image, Image.Image):
        image = Image.open(image)
    image = image.convert("RGB")

    pixel_values = model.image_processor(images=image, return_tensors="pt")["pixel_values"].to(device)
    placeholder = IMAGE_TOKEN * model.num_visual_tokens
    text = build_prompt(model.tokenizer, question, placeholder, short_answer=short_answer)
    enc = model.tokenizer(text, return_tensors="pt", add_special_tokens=False)

    gen = dict(
        max_new_tokens=16 if short_answer else max_new_tokens,
        repetition_penalty=repetition_penalty,
    )
    if temperature and temperature > 0:
        gen.update(do_sample=True, temperature=temperature, top_p=0.9)
    else:
        gen.update(do_sample=False)

    out = model.generate(
        input_ids=enc["input_ids"].to(device),
        attention_mask=enc["attention_mask"].to(device),
        pixel_values=pixel_values,
        **gen,
    )
    # inputs_embeds (not input_ids) went into generate(), so HF returns only
    # the newly generated tokens.
    return model.tokenizer.decode(out[0], skip_special_tokens=True).strip()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("--question", required=True)
    p.add_argument("--projector_ckpt", required=True)
    p.add_argument("--lora_dir", default=None)
    p.add_argument("--short", action="store_true", help="append the short-answer hint (VQA mode)")
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()

    m = load_model(a.projector_ckpt, a.lora_dir, a.device)
    print(answer(m, a.image, a.question, a.device, a.max_new_tokens,
                 short_answer=a.short, temperature=a.temperature))
