"""
train_stage2_lora.py

Stage 2: load the Stage 1 projector, keep vision encoder frozen, attach LoRA
adapters to the LLM, and train on instruction-style VQA data so the model
learns to actually reason over visual tokens and answer conversationally.

Usage:
    python train_stage2_lora.py \
        --data data/vqa_train.jsonl \
        --image_root data/images \
        --projector_ckpt checkpoints/stage1/projector.pt \
        --output_dir checkpoints/stage2 \
        --epochs 2 --batch_size 4 --lr 2e-4
"""

import argparse
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from peft import LoraConfig, get_peft_model

from model import TinyVLM
from dataset import VQADataset, make_collate_fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--image_root", required=True)
    parser.add_argument("--projector_ckpt", required=True)
    parser.add_argument("--output_dir", default="checkpoints/stage2")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Vision stays frozen; LLM base weights frozen too -- LoRA adapters do the learning
    model = TinyVLM(freeze_vision=True, freeze_llm=True)
    model.projector.load_state_dict(torch.load(args.projector_ckpt, map_location="cpu"))
    model.projector.requires_grad_(True)  # keep fine-tuning the projector jointly

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Qwen2 attention proj names
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model.llm = get_peft_model(model.llm, lora_config)
    model.llm.print_trainable_parameters()

    model = model.to(args.device)
    model.train()

    dataset = VQADataset(args.data, args.image_root, model.tokenizer, model.image_processor)
    collate_fn = make_collate_fn(model)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)

    for epoch in range(args.epochs):
        pbar = tqdm(loader, desc=f"epoch {epoch}")
        for batch in pbar:
            batch = {k: v.to(args.device) for k, v in batch.items()}
            outputs = model(input_ids=batch["input_ids"], pixel_values=batch["pixel_values"], labels=batch["labels"])
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=loss.item())

    # Save LoRA adapter + projector together
    model.llm.save_pretrained(os.path.join(args.output_dir, "lora_adapter"))
    torch.save(model.projector.state_dict(), os.path.join(args.output_dir, "projector.pt"))
    print(f"Saved LoRA adapter + projector to {args.output_dir}")


if __name__ == "__main__":
    main()
