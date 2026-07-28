"""
tools/build_notebook.py

Generates `vlm-train-v2.ipynb` as a SELF-CONTAINED Kaggle notebook: the
`%%writefile` cells are produced from the repo's real .py files at build time,
so the notebook can be uploaded on its own with no git clone and no Kaggle
dataset attachment.

The point of generating them is that they cannot drift. Run 1 kept the model and
training code in both the notebook and the repo, hand-maintained, and they had
already diverged -- the notebook's `train_stage1_projector.py` had gradient
accumulation and the repo's did not. Edit the .py files, re-run this, never edit
the writefile cells by hand.

    python tools/build_notebook.py
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "vlm-train-v2.ipynb"

# repo path -> flat filename inside /kaggle/working
EMBED = [
    (ROOT / "Model" / "model.py", "model.py"),
    (ROOT / "Model" / "dataset.py", "dataset.py"),
    (ROOT / "Training" / "prepare_data.py", "prepare_data.py"),
    (ROOT / "Training" / "train.py", "train.py"),
    (ROOT / "Inference" / "inference.py", "inference.py"),
    (ROOT / "Inference" / "diagnose_grounding.py", "diagnose_grounding.py"),
    (ROOT / "push_to_hub.py", "push_to_hub.py"),
]

CELLS = []


def _src(s: str) -> list[str]:
    """nbformat stores `source` as a list of lines that each KEEP their trailing
    newline; a cell is reconstructed with "".join(source). Splitting without
    keepends produces a file that renders as one long line."""
    return s.strip("\n").splitlines(keepends=True)


def md(s):
    CELLS.append({"cell_type": "markdown", "metadata": {}, "source": _src(s)})


def code(s):
    CELLS.append({"cell_type": "code", "execution_count": None, "metadata": {},
                  "outputs": [], "source": _src(s)})


def embed(src: Path, name: str):
    body = src.read_text(encoding="utf-8").rstrip("\n")
    code(f"%%writefile {name}\n{body}")


md(r"""
# TinyVLM v2 -- SigLIP2 + Qwen2.5-0.5B

Rebuild of the pipeline after diagnosing why run 1 answered with one or two
fixed responses regardless of the image.

## What went wrong in run 1

| Root cause | Evidence | Fix |
|---|---|---|
| Projector alignment corpus ~350x too small | `--max_examples 8000` on `captions.txt` = ~1600 distinct images (5 captions per image, stored consecutively); 500 optimizer steps total | Flickr8k + Flickr30k, ~39k images / ~185k pairs |
| The failure was invisible | `loss = loss/accum_steps` was assigned *before* `set_postfix`, so the logged Stage-1 `0.786` was really ~3.14 -- about the text-only prior | logging reports the true per-token loss |
| Trained to emit exactly one word | VQAv2 targets tokenize to `['red', '<\|im_end\|>']` -- 2 supervised tokens out of 238, 10k times | mix VQA + detailed captions; VQA rows carry an explicit length hint |
| Visual tokens outside the embedding manifold | SigLIP token norm 47.2 vs Qwen embedding norm 0.45 (~104x) | projector ends in a LayerNorm initialised to the LLM's own embedding std -> 1.01x |
| LoRA never touched the MLP block | `target_modules=[q,k,v,o]` | + `gate_proj, up_proj, down_proj` |
| 1.1 GB adapter | adding `<image>` forced `resize_token_embeddings`, so PEFT serialised the whole embedding matrix | reuse Qwen's existing `<\|image_pad\|>`; adapter ~35 MB |
| GPU starved at 1.33 it/s | `num_workers=0` + full-resolution decode every epoch | prep-time resize, worker processes, fp16 AMP |
| No way to notice before upload | no held-out loss, no sample generations | eval + generations every 500 steps, plus a grounding gate between stages |

## Session setup

- **Accelerator:** GPU T4 x2
- **Internet:** On
- **Add data:** the Kaggle dataset `adityajn105/flickr8k`

Rough budget: 1.5-2 h data prep, ~2 h Stage 1, ~1.5 h Stage 2.

> The `%%writefile` cells below are generated from the training repo by
> `tools/build_notebook.py`. Edit the repo and regenerate -- don't hand-edit
> them here, which is how run 1's notebook and repo copies drifted apart.
""")

md("## 1. Environment")

code(r"""
!pip install -q "transformers>=4.45" "peft>=0.11" accelerate datasets huggingface_hub
""")

code(r"""
import torch, transformers, peft
print("torch", torch.__version__, "| cuda", torch.cuda.is_available())
print("transformers", transformers.__version__, "| peft", peft.__version__)
print(torch.cuda.get_device_name(0) if torch.cuda.is_available()
      else "NO GPU -- set Accelerator to GPU T4 x2 in the sidebar")
""")

md("## 2. Project files")

code(r"""
import os
os.chdir("/kaggle/working")
print(os.getcwd())
""")

for src, name in EMBED:
    embed(src, name)

code(r"""
!ls -la /kaggle/working/*.py
""")

md(r"""
## 3. Data

Four corpora, all written into one shared image directory as JPEGs resized to a
256px short side. Run 1 symlinked full-resolution originals and re-decoded them
every epoch.

| corpus | role | rows |
|---|---|---|
| Flickr8k | alignment captions | ~40k (8k images) |
| Flickr30k | alignment captions | ~145k (29k images) |
| VQAv2 | short answers | 40k |
| LLaVA-ReCap-118K | detailed descriptions | 25k |
""")

code(r"""
!ls /kaggle/input
""")

code(r"""
# Flickr8k, from the Kaggle input attached in the sidebar. If this fails, the
# traceback prints the directory tree it searched so you can point it correctly.
!python prepare_data.py --task flickr8k \
    --kaggle_input_dir /kaggle/input \
    --out_jsonl /kaggle/working/data/cap8k.jsonl \
    --out_image_dir /kaggle/working/data/images
""")

code(r"""
# ~29k more distinct images. This is the single biggest change from run 1:
# the projector now sees ~24x more visual variety.
!python prepare_data.py --task flickr30k \
    --out_jsonl /kaggle/working/data/cap30k.jsonl \
    --out_image_dir /kaggle/working/data/images
""")

code(r"""
!python prepare_data.py --task vqav2 --max_examples 40000 \
    --out_jsonl /kaggle/working/data/vqa.jsonl \
    --out_image_dir /kaggle/working/data/images
""")

code(r"""
# Detailed multi-sentence descriptions -- what restores the ability to write
# prose rather than a single word.
!python prepare_data.py --task recap --max_examples 25000 \
    --out_jsonl /kaggle/working/data/recap.jsonl \
    --out_image_dir /kaggle/working/data/images
""")

code(r"""
# Stage 1 = pure captioning. Stage 2 = a deliberate mix of answer styles.
!python prepare_data.py --task mix \
    --inputs /kaggle/working/data/cap8k.jsonl /kaggle/working/data/cap30k.jsonl \
    --out_jsonl /kaggle/working/data/stage1.jsonl

!python prepare_data.py --task mix \
    --inputs /kaggle/working/data/vqa.jsonl /kaggle/working/data/recap.jsonl /kaggle/working/data/cap8k.jsonl \
    --caps 40000 25000 20000 \
    --out_jsonl /kaggle/working/data/stage2.jsonl
""")

code(r"""
import json, collections
for name in ["stage1", "stage2"]:
    rows = [json.loads(l) for l in open(f"/kaggle/working/data/{name}.jsonl", encoding="utf-8")]
    styles = collections.Counter(r.get("style", "caption") for r in rows)
    ans = [len(t["value"].split()) for r in rows for t in r["conversations"] if t["from"] == "gpt"]
    print(f"{name}: {len(rows):,} rows | styles {dict(styles)}")
    print(f"   answer length: mean {sum(ans)/len(ans):.1f} words, min {min(ans)}, max {max(ans)}")
""")

md(r"""
## 4. Stage 1 -- projector alignment

Vision encoder and LLM frozen; only the projector trains. Gradients still flow
*through* the LLM to reach it, so this is not cheap, but no LLM weights move.

Watch the sample generations printed every 500 steps. If two different images
produce the same caption, stop -- Stage 2 will not rescue it.
""")

code(r"""
!python train.py --stage 1 \
    --data /kaggle/working/data/stage1.jsonl \
    --image_root /kaggle/working/data/images \
    --output_dir /kaggle/working/ckpt/stage1 \
    --epochs 2 --batch_size 32 --accum_steps 1 --lr 1e-3 \
    --num_workers 4 --eval_every 500 --save_every 4000
""")

md(r"""
### Grounding gate

Run this before spending GPU time on Stage 2. Three visually different images,
one question. Identical answers means the projector learned nothing, and no
amount of Stage-2 tuning or decoding-parameter fiddling fixes that.
""")

code(r"""
import json
rows = [json.loads(l) for l in open("/kaggle/working/data/stage2.jsonl", encoding="utf-8")][:400]
seen, picks = set(), []
for r in rows:                      # one image from each source corpus
    tag = r["image"].split("_")[0]
    if tag not in seen:
        seen.add(tag); picks.append("/kaggle/working/data/images/" + r["image"])
    if len(picks) == 3:
        break
IMGS = " ".join(picks)
print(IMGS)
""")

code(r"""
!python diagnose_grounding.py \
    --projector_ckpt /kaggle/working/ckpt/stage1/projector.pt \
    --images {IMGS}
""")

md(r"""
## 5. Stage 2 -- LoRA instruction tuning

LoRA on attention **and** MLP projections; the projector keeps training at a much
lower LR so it is refined rather than overwritten.
""")

code(r"""
!python train.py --stage 2 \
    --data /kaggle/working/data/stage2.jsonl \
    --image_root /kaggle/working/data/images \
    --projector_ckpt /kaggle/working/ckpt/stage1/projector.pt \
    --output_dir /kaggle/working/ckpt/stage2 \
    --epochs 3 --batch_size 16 --accum_steps 1 \
    --lr 2e-4 --projector_lr 2e-5 --lora_r 16 --lora_alpha 32 \
    --num_workers 4 --eval_every 500 --save_every 4000
""")

md("### Grounding gate again, now with the adapter")

code(r"""
!python diagnose_grounding.py \
    --projector_ckpt /kaggle/working/ckpt/stage2/projector.pt \
    --lora_dir /kaggle/working/ckpt/stage2/lora_adapter \
    --images {IMGS}
""")

md(r"""
## 6. Qualitative check

Both answer styles on the same images. Descriptive prompts should produce
sentences; the short-answer hint should produce a word or two. If everything
comes back as one word, the Stage-2 style mix did not take.
""")

code(r"""
import sys
sys.path.insert(0, "/kaggle/working")
from inference import load_model, answer
from PIL import Image
import matplotlib.pyplot as plt

model = load_model("/kaggle/working/ckpt/stage2/projector.pt",
                   "/kaggle/working/ckpt/stage2/lora_adapter", "cuda")

fig, axes = plt.subplots(1, len(picks), figsize=(5 * len(picks), 5))
for ax, path in zip(axes if len(picks) > 1 else [axes], picks):
    img = Image.open(path).convert("RGB")
    long_a = answer(model, img, "What is in this image?", "cuda")
    short_a = answer(model, img, "What is in this image?", "cuda", short_answer=True)
    ax.imshow(img); ax.axis("off")
    ax.set_title(f"long: {long_a[:70]}\nshort: {short_a[:40]}", fontsize=8, wrap=True)
    print(f"{path}\n  descriptive: {long_a}\n  short      : {short_a}\n")
plt.tight_layout(); plt.show()
""")

code(r"""
# Exact-match VQA accuracy on the short-answer prompt.
import json, random
rows = [json.loads(l) for l in open("/kaggle/working/data/vqa.jsonl", encoding="utf-8")]
random.Random(1).shuffle(rows)

hits = 0
sample = rows[:150]
for r in sample:
    q = r["conversations"][0]["value"].replace("<image>", "").split("\nAnswer the question")[0].strip()
    gt = r["conversations"][1]["value"].strip().lower()
    pred = answer(model, "/kaggle/working/data/images/" + r["image"], q, "cuda",
                  short_answer=True).strip().lower().rstrip(".")
    hits += (pred == gt)
print(f"exact match on {len(sample)} rows: {hits/len(sample):.1%}")
print("(a question-prior-only model lands near 25-30%; grounded should be clearly above)")
""")

md("## 7. Push weights to the Hub")

code(r"""
import os
try:
    from kaggle_secrets import UserSecretsClient
    os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
except Exception:
    print("No Kaggle secret named HF_TOKEN -- add one under Add-ons > Secrets.")

from huggingface_hub import login
login(token=os.environ["HF_TOKEN"]) if "HF_TOKEN" in os.environ else login()
""")

code(r"""
!python push_to_hub.py \
    --repo_id dhruvpatel93/tinyvlm-vqa \
    --projector_ckpt /kaggle/working/ckpt/stage2/projector.pt \
    --lora_dir /kaggle/working/ckpt/stage2/lora_adapter \
    --meta /kaggle/working/ckpt/stage2/train_meta.json
""")

md(r"""
## 8. The Space does NOT update itself

Pushing weights to the model repo does not rebuild a Space, and restarting the
existing one by hand will crash: it runs run-1 code, and these checkpoints are
not loadable by it. The projector gained an `out_norm`, and its first Linear
went from 768 to 3072 input features (2x2 pixel-shuffle), so `load_state_dict`
raises immediately.

Deploy the code and the weights together, from the training repo:

```bash
python deploy_space.py --space_id <you>/tinyvlm --model_repo dhruvpatel93/tinyvlm-vqa
```

That flattens `model.py`, `dataset.py`, `inference.py` and `app_gradio.py` into
the Space, sets `HF_REPO_ID` so the app pulls weights from the Hub at startup,
and triggers a rebuild. After that first deploy, weights-only changes need just
a push plus a Space restart.
""")

nb = {
    "cells": CELLS,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
OUT.write_text(json.dumps(nb, indent=1), encoding="utf-8")

embedded = sum(len(s.read_text(encoding='utf-8').splitlines()) for s, _ in EMBED)
print(f"wrote {OUT}")
print(f"  {len(CELLS)} cells, {len(EMBED)} embedded modules, {embedded:,} lines of code inlined")
