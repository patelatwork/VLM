"""
train.py -- both stages, one script.

    Stage 1  projector alignment    (vision frozen, LLM frozen, projector trains)
    Stage 2  instruction tuning     (vision frozen, LoRA on LLM, projector fine-tunes)

The previous code had a separate script per stage, and the copies in the repo
and in the notebook had already drifted apart (the notebook's stage-1 script had
gradient accumulation, the repo's did not). One file, one `--stage` flag.

Fixes relative to the first version
-----------------------------------
* Loss logging was wrong: `loss = loss / accum_steps` was assigned before
  `set_postfix(loss=loss.item())`, so the reported Stage-1 loss of 0.786 was
  really ~3.14. Here the reported number is always the true per-token loss.
* `num_workers=0` meant image decode was single-threaded and the GPU starved
  (1.33 it/s for a 0.5B model). Now workers + prefetch + persistent workers.
* No LR schedule and no warmup. Now warmup + cosine.
* No gradient clipping, no AMP. Now both (fp16 -- T4 is Turing, no bf16).
* No held-out loss and no sample generations, so a collapsed projector was
  invisible until after the upload. Now both, every `--eval_every` steps.

Usage:
    python train.py --stage 1 --data data/stage1.jsonl --image_root data/images \
        --output_dir ckpt/stage1 --epochs 1 --batch_size 16 --lr 1e-3

    python train.py --stage 2 --data data/stage2.jsonl --image_root data/images \
        --projector_ckpt ckpt/stage1/projector.pt --output_dir ckpt/stage2 \
        --epochs 2 --batch_size 8 --lr 2e-4
"""

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

for _p in (Path(__file__).resolve().parent.parent / "Model", Path(__file__).resolve().parent):
    sys.path.insert(0, str(_p))

from model import TinyVLM, IMAGE_TOKEN            # noqa: E402
from dataset import VLMDataset, make_collate_fn, build_prompt   # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", type=int, choices=[1, 2], required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--image_root", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--projector_ckpt", default=None, help="stage 2: stage-1 projector")

    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--accum_steps", type=int, default=2)
    p.add_argument("--lr", type=float, default=None, help="default: 1e-3 (stage 1) / 2e-4 (stage 2)")
    p.add_argument("--projector_lr", type=float, default=2e-5, help="stage 2 only")
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--max_steps", type=int, default=-1, help="hard budget cap on optimizer steps")

    p.add_argument("--pool_stride", type=int, default=2)
    p.add_argument("--max_length", type=int, default=1024)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)

    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--eval_every", type=int, default=500)
    p.add_argument("--save_every", type=int, default=2000)
    p.add_argument("--eval_frac", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--amp", default="fp16", choices=["fp16", "bf16", "off"])

    a = p.parse_args()
    if a.lr is None:
        a.lr = 1e-3 if a.stage == 1 else 2e-4
    return a


def build_model(args):
    model = TinyVLM(freeze_vision=True, freeze_llm=True, pool_stride=args.pool_stride)

    if args.projector_ckpt:
        sd = torch.load(args.projector_ckpt, map_location="cpu")
        model.projector.load_state_dict(sd)
        print(f"loaded projector from {args.projector_ckpt}")
    elif args.stage == 2:
        print("WARNING: stage 2 without --projector_ckpt -- the projector starts from scratch.")

    model.projector.requires_grad_(True)

    if args.stage == 2:
        from peft import LoraConfig, get_peft_model

        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            # The first run only adapted q/k/v/o. The MLP block is where a
            # frozen LLM actually absorbs a new input modality, so it is
            # included here.
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model.llm = get_peft_model(model.llm, lora_config)

    return model


@torch.no_grad()
def evaluate(model, loader, device, amp_dtype, max_batches=40):
    model.eval()
    total, n_tok = 0.0, 0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
            out = model(**batch)
        n = int((batch["labels"] != -100).sum())
        if torch.isfinite(out.loss) and n:
            total += out.loss.item() * n
            n_tok += n
    model.train()
    model.vision_encoder.eval()
    return total / max(n_tok, 1)


@torch.no_grad()
def sample_generations(model, examples, image_root, device, amp_dtype, n=2):
    """Generate on a couple of held-out images. If these come out identical for
    different images, the projector has collapsed -- stop and investigate."""
    from PIL import Image

    model.eval()
    placeholder = IMAGE_TOKEN * model.num_visual_tokens
    outs = []
    for ex in examples[:n]:
        img = Image.open(Path(image_root) / ex["image"]).convert("RGB")
        px = model.image_processor(images=img, return_tensors="pt")["pixel_values"].to(device)
        q = ex["conversations"][0]["value"].replace("<image>", "").strip()
        text = build_prompt(model.tokenizer, q, placeholder)
        enc = model.tokenizer(text, return_tensors="pt", add_special_tokens=False)
        with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
            ids = model.generate(
                input_ids=enc["input_ids"].to(device),
                attention_mask=enc["attention_mask"].to(device),
                pixel_values=px,
                max_new_tokens=64,
                do_sample=False,
            )
        outs.append((ex["image"], q[:60], model.tokenizer.decode(ids[0], skip_special_tokens=True)))
    model.train()
    model.vision_encoder.eval()
    return outs


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    print(json.dumps(vars(args), indent=2))

    amp_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "off": None}[args.amp]
    if args.device != "cuda":
        amp_dtype = None

    model = build_model(args).to(args.device)
    model.train()
    model.vision_encoder.eval()  # frozen: keep it deterministic
    print(model.trainable_parameter_report())
    print(f"visual tokens per image: {model.num_visual_tokens} (pool_stride={args.pool_stride})")

    full = VLMDataset(
        args.data, args.image_root, model.tokenizer, model.image_processor,
        num_visual_tokens=model.num_visual_tokens,
        image_token=IMAGE_TOKEN, max_length=args.max_length,
    )
    n_eval = max(16, int(len(full) * args.eval_frac))
    idx = list(range(len(full)))
    random.Random(args.seed).shuffle(idx)
    train_ds = Subset(full, idx[n_eval:])
    eval_ds = Subset(full, idx[:n_eval])
    print(f"train {len(train_ds):,} | eval {len(eval_ds):,}")

    collate = make_collate_fn(model.tokenizer.pad_token_id)
    common = dict(collate_fn=collate, num_workers=args.num_workers, pin_memory=True)
    if args.num_workers > 0:
        common.update(persistent_workers=True, prefetch_factor=4)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True, **common)
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False, **common)

    sample_examples = [full.examples[i] for i in idx[:2]]

    # --- optimizer: LoRA and an already-trained projector want different LRs ---
    if args.stage == 1:
        groups = [{"params": [p for p in model.projector.parameters() if p.requires_grad], "lr": args.lr}]
    else:
        groups = [
            {"params": [p for p in model.llm.parameters() if p.requires_grad], "lr": args.lr},
            {"params": [p for p in model.projector.parameters() if p.requires_grad], "lr": args.projector_lr},
        ]
    optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay, betas=(0.9, 0.95))

    steps_per_epoch = math.ceil(len(train_loader) / args.accum_steps)
    total_steps = steps_per_epoch * args.epochs
    if args.max_steps > 0:
        total_steps = min(total_steps, args.max_steps)
    # Floor of 10 for real runs, but never let warmup eat the whole schedule
    # (matters for short smoke runs, where the LR would otherwise never peak).
    warmup = max(1, min(max(10, int(total_steps * args.warmup_ratio)), total_steps // 5))
    from transformers import get_cosine_schedule_with_warmup
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup, total_steps)
    print(f"optimizer steps: {total_steps:,} (warmup {warmup:,})")

    scaler = torch.amp.GradScaler("cuda", enabled=(amp_dtype is torch.float16))

    gstep, micro, running, seen, t0 = 0, 0, 0.0, 0, time.time()
    stop = False
    for epoch in range(args.epochs):
        if stop:
            break
        pbar = tqdm(train_loader, desc=f"stage{args.stage} epoch {epoch}")
        for batch in pbar:
            batch = {k: v.to(args.device, non_blocking=True) for k, v in batch.items()}
            with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
                out = model(**batch)
                loss = out.loss

            if not torch.isfinite(loss):
                # a fully-truncated example can yield NaN; drop the micro-batch
                optimizer.zero_grad(set_to_none=True)
                micro = 0
                continue

            scaler.scale(loss / args.accum_steps).backward()
            # report the TRUE loss, not the accumulation-scaled one
            running += loss.item()
            seen += 1
            micro += 1

            if micro == args.accum_steps:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for g in groups for p in g["params"]], args.max_grad_norm
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                gstep += 1
                micro = 0

                pbar.set_postfix(
                    loss=f"{running / max(seen, 1):.3f}",
                    lr=f"{scheduler.get_last_lr()[0]:.2e}",
                    step=gstep,
                )
                if seen >= 50:
                    running, seen = 0.0, 0

                if gstep % args.eval_every == 0:
                    ev = evaluate(model, eval_loader, args.device, amp_dtype)
                    gens = sample_generations(model, sample_examples, args.image_root, args.device, amp_dtype)
                    tqdm.write(f"\n[step {gstep}] eval_loss={ev:.4f}  elapsed={(time.time()-t0)/60:.1f}m")
                    for img, q, g in gens:
                        tqdm.write(f"   {img} | {q!r}\n     -> {g[:200]!r}")
                    if len({g for _, _, g in gens}) == 1 and len(gens) > 1:
                        tqdm.write("   !! identical outputs on different images -- projector may be collapsing")

                if args.save_every > 0 and gstep % args.save_every == 0:
                    save(model, args, tag=f"step{gstep}")

                if args.max_steps > 0 and gstep >= args.max_steps:
                    stop = True
                    break

    ev = evaluate(model, eval_loader, args.device, amp_dtype)
    print(f"\nfinal eval_loss = {ev:.4f}  ({(time.time()-t0)/60:.1f} min, {gstep} steps)")
    save(model, args)
    (Path(args.output_dir) / "train_meta.json").write_text(
        json.dumps({**vars(args), "final_eval_loss": ev, "steps": gstep,
                    "num_visual_tokens": model.num_visual_tokens}, indent=2)
    )


def save(model, args, tag=None):
    out = Path(args.output_dir) if tag is None else Path(args.output_dir) / tag
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.projector.state_dict(), out / "projector.pt")
    if args.stage == 2:
        model.llm.save_pretrained(out / "lora_adapter")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
