"""
push_to_hub.py

Uploads the trained projector + LoRA adapter + a model card.

The adapter is now ~35 MB rather than 1.1 GB: the first version added an
`<image>` token and called `resize_token_embeddings`, which made PEFT serialise
the entire 151k x 896 embedding matrix into the adapter. Reusing Qwen's
existing `<|image_pad|>` token removes that entirely.

Usage:
    python push_to_hub.py --repo_id you/tinyvlm --projector_ckpt ckpt/stage2/projector.pt \
        --lora_dir ckpt/stage2/lora_adapter
"""

import argparse
import json
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, create_repo

CARD = """---
license: apache-2.0
base_model: Qwen/Qwen2.5-0.5B-Instruct
tags: [vision-language-model, vqa, image-captioning, lora, siglip2, qwen2.5]
---

# {repo_name}

A compact LLaVA-style VLM: SigLIP2 vision encoder -> pixel-shuffle -> MLP
projector -> Qwen2.5-0.5B-Instruct with LoRA.

| | |
|---|---|
| Vision encoder | `google/siglip2-base-patch16-224` (frozen) |
| LLM | `Qwen/Qwen2.5-0.5B-Instruct` (LoRA: q,k,v,o,gate,up,down) |
| Visual tokens | {num_visual_tokens} per image (14x14 patches, 2x2 pixel-shuffle) |
| Image placeholder | `<|image_pad|>` -- an existing Qwen token, so no vocab resize |
| Stage 1 | projector alignment on Flickr8k + Flickr30k captions |
| Stage 2 | LoRA instruction tuning on VQAv2 + LLaVA-ReCap + captions |

## Answer length is prompt-controlled

Stage 2 mixes one-word VQA answers with multi-sentence descriptions, and VQA
examples carry an explicit hint. Append it for short answers, omit it for prose:

```
What color is the bus?
Answer the question using a single word or phrase.     -> "red"

What color is the bus?                                 -> "The bus is red."
```

## Files

- `projector.pt` -- projector weights (`pool_stride` is recoverable from the tensor shapes)
- `lora_adapter/` -- PEFT adapter for the LLM

## Usage

Needs the companion code from the training repo (`Model/model.py`,
`Model/dataset.py`, `Inference/inference.py`):

```python
from huggingface_hub import hf_hub_download, snapshot_download
from inference import load_model, answer

projector = hf_hub_download(repo_id="{repo_id}", filename="projector.pt")
lora = snapshot_download(repo_id="{repo_id}", allow_patterns=["lora_adapter/*"]) + "/lora_adapter"

model = load_model(projector, lora, device="cuda")
print(answer(model, "photo.jpg", "What is in this image?"))
print(answer(model, "photo.jpg", "What color is the car?", short_answer=True))
```

## Limitations

224x224 input, {num_visual_tokens} visual tokens, and a 0.5B LLM. Expect
everyday-scene captioning and simple VQA -- not OCR, fine detail, counting, or
multi-step visual reasoning.

{metrics}
"""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo_id", required=True)
    p.add_argument("--projector_ckpt", required=True)
    p.add_argument("--lora_dir", required=True)
    p.add_argument("--meta", default=None, help="train_meta.json from stage 2")
    p.add_argument("--private", action="store_true")
    a = p.parse_args()

    num_visual_tokens, metrics = 49, ""
    if a.meta and Path(a.meta).exists():
        meta = json.loads(Path(a.meta).read_text())
        num_visual_tokens = meta.get("num_visual_tokens", 49)
        metrics = (
            "## Training\n\n"
            f"- optimizer steps: {meta.get('steps')}\n"
            f"- final held-out loss: {meta.get('final_eval_loss'):.4f}\n"
            f"- batch size {meta.get('batch_size')} x {meta.get('accum_steps')} accum, "
            f"lr {meta.get('lr')}\n"
        )

    api = HfApi()
    create_repo(a.repo_id, private=a.private, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        shutil.copy(a.projector_ckpt, tmp / "projector.pt")
        shutil.copytree(a.lora_dir, tmp / "lora_adapter")
        if a.meta and Path(a.meta).exists():
            shutil.copy(a.meta, tmp / "train_meta.json")
        (tmp / "README.md").write_text(
            CARD.format(
                repo_name=a.repo_id.split("/")[-1],
                repo_id=a.repo_id,
                num_visual_tokens=num_visual_tokens,
                metrics=metrics,
            ),
            encoding="utf-8",
        )
        api.upload_folder(folder_path=str(tmp), repo_id=a.repo_id,
                          commit_message="TinyVLM projector + LoRA adapter")

    print(f"https://huggingface.co/{a.repo_id}")


if __name__ == "__main__":
    main()
