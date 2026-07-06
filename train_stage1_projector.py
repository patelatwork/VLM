"""
train_stage1_projector.py

Stage 1: freeze vision encoder + LLM, train ONLY the projector on
image-caption pairs. This teaches the projector to map visual features
into the LLM's embedding space.

Usage:
    python train_stage1_projector.py \
        --data data/captions_train.jsonl \
        --image_root data/images \
        --output_dir checkpoints/stage1 \
        --epochs 1 --batch_size 8 --lr 1e-4
"""

import argparse
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import TinyVLM
from dataset import VQADataset, make_collate_fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--image_root", required=True)
    parser.add_argument("--output_dir", default="checkpoints/stage1")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    model = TinyVLM(freeze_vision=True, freeze_llm=True).to(args.device)
    model.train()

    dataset = VQADataset(args.data, args.image_root, model.tokenizer, model.image_processor)
    collate_fn = make_collate_fn(model)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    # Only the projector has requires_grad=True at this stage
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)

    print(f"Trainable params (should be projector only): {sum(p.numel() for p in trainable_params):,}")

    step = 0
    for epoch in range(args.epochs):
        pbar = tqdm(loader, desc=f"epoch {epoch}")
        for batch in pbar:
            batch = {k: v.to(args.device) for k, v in batch.items()}
            outputs = model(input_ids=batch["input_ids"], pixel_values=batch["pixel_values"], labels=batch["labels"])
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step += 1
            pbar.set_postfix(loss=loss.item())

    torch.save(model.projector.state_dict(), os.path.join(args.output_dir, "projector.pt"))
    print(f"Saved projector weights to {args.output_dir}/projector.pt")


if __name__ == "__main__":
    main()
