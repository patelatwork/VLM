"""
inference.py

Load the base model + trained projector + LoRA adapter, and answer a
question about a given image.

Usage:
    python inference.py \
        --image path/to/image.jpg \
        --question "What is in this image?" \
        --projector_ckpt checkpoints/stage2/projector.pt \
        --lora_dir checkpoints/stage2/lora_adapter
"""

import argparse

import torch
from PIL import Image
from peft import PeftModel

from model import TinyVLM, IMAGE_TOKEN


def load_model(projector_ckpt: str, lora_dir: str = None, device: str = "cuda"):
    model = TinyVLM(freeze_vision=True, freeze_llm=True)
    model.projector.load_state_dict(torch.load(projector_ckpt, map_location="cpu"))

    if lora_dir:
        model.llm = PeftModel.from_pretrained(model.llm, lora_dir)

    model = model.to(device)
    model.eval()
    return model


def answer(model, image_path: str, question: str, device: str = "cuda", max_new_tokens: int = 64) -> str:
    image = Image.open(image_path).convert("RGB")
    pixel_values = model.image_processor(images=image, return_tensors="pt")["pixel_values"].to(device)

    messages = [{"role": "user", "content": f"{IMAGE_TOKEN} {question}"}]
    input_ids = model.tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            pixel_values=pixel_values,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    # output_ids only contains newly generated tokens since we passed inputs_embeds,
    # not input_ids, to generate() -- decode directly
    return model.tokenizer.decode(output_ids[0], skip_special_tokens=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--projector_ckpt", required=True)
    parser.add_argument("--lora_dir", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    model = load_model(args.projector_ckpt, args.lora_dir, args.device)
    result = answer(model, args.image, args.question, args.device)
    print(result)
