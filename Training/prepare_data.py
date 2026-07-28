"""
prepare_data.py

Builds the JSONL corpora consumed by dataset.VLMDataset.

Why the corpus changed
----------------------
The first run aligned the projector on `--max_examples 8000` rows of Flickr8k's
captions.txt. That file stores 5 captions per image consecutively, so 8000 rows
is only ~1600 distinct images, and the projector never saw enough visual
variety to learn anything. LLaVA's equivalent stage uses 558k pairs.

Stage 1 here uses Flickr8k + Flickr30k: ~39k distinct images / ~195k caption
pairs. That is ~24x more images, and it fits comfortably in one Kaggle session.

Stage 2 mixes three response styles so the model learns that answer length is
controlled by the prompt rather than baked in:
  * vqav2    -- one-word answers, tagged with the short-answer hint
  * recap    -- detailed multi-sentence descriptions
  * captions -- one-sentence descriptions

Images are resized at prep time (short side 256, JPEG q90). The old pipeline
symlinked full-resolution originals and re-decoded them every epoch, which is a
large part of why Stage 1 only managed 1.33 it/s.

Usage:
    python prepare_data.py --task flickr8k  --kaggle_input_dir /kaggle/input/... \
        --out_jsonl data/cap8k.jsonl --out_image_dir data/images --max_examples 40000
    python prepare_data.py --task flickr30k --out_jsonl data/cap30k.jsonl --out_image_dir data/images
    python prepare_data.py --task vqav2     --out_jsonl data/vqa.jsonl   --out_image_dir data/images --max_examples 30000
    python prepare_data.py --task recap     --out_jsonl data/recap.jsonl --out_image_dir data/images --max_examples 20000
    python prepare_data.py --task mix --inputs data/vqa.jsonl data/recap.jsonl data/cap8k.jsonl \
        --out_jsonl data/stage2.jsonl
"""

import argparse
import csv
import json
import random
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "Model"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from dataset import SHORT_ANSWER_HINT, DESCRIBE_INSTRUCTIONS, IMAGE_MARKER
except ImportError:  # the notebook writes all the .py files flat
    SHORT_ANSWER_HINT = "\nAnswer the question using a single word or phrase."
    IMAGE_MARKER = "<image>"
    DESCRIBE_INSTRUCTIONS = [
        "Describe this image.",
        "What is happening in this image?",
        "Write a short caption for this image.",
        "Describe the image in detail.",
        "What do you see in this picture?",
        "Provide a description of this image.",
    ]

TARGET_SHORT_SIDE = 256
JPEG_QUALITY = 90


def save_resized(img: Image.Image, dst: Path) -> bool:
    """Resize so the short side is TARGET_SHORT_SIDE, save JPEG.
    Returns False if the file already existed."""
    if dst.exists():
        return False
    img = img.convert("RGB")
    w, h = img.size
    scale = TARGET_SHORT_SIDE / min(w, h)
    if scale < 1.0:
        img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.BICUBIC)
    dst.parent.mkdir(parents=True, exist_ok=True)
    img.save(dst, "JPEG", quality=JPEG_QUALITY)
    return True


def turn(human: str, gpt: str):
    return [
        {"from": "human", "value": f"{IMAGE_MARKER}\n{human}"},
        {"from": "gpt", "value": gpt},
    ]


def write_jsonl(records, out_jsonl: str):
    Path(out_jsonl).parent.mkdir(parents=True, exist_ok=True)
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(records):,} examples -> {out_jsonl}")


# --------------------------------------------------------------------------- #
# Flickr8k (Kaggle input)
# --------------------------------------------------------------------------- #

def _find_flickr8k_files(root: Path):
    caps = list(root.rglob("captions.txt")) or list(root.rglob("Flickr8k.token.txt"))
    imgs = [d for d in root.rglob("Images") if d.is_dir()] or [d for d in root.rglob("images") if d.is_dir()]
    if not caps or not imgs:
        tree = "\n".join(str(p) for p in sorted(root.rglob("*"))[:50])
        raise FileNotFoundError(
            f"Could not find captions file and/or Images dir under {root}.\n"
            f"First 50 entries:\n{tree}"
        )
    return caps[0], imgs[0]


def prepare_flickr8k(kaggle_input_dir, out_jsonl, out_image_dir, max_examples=None, seed=0):
    rng = random.Random(seed)
    captions_file, images_dir = _find_flickr8k_files(Path(kaggle_input_dir))
    print(f"captions: {captions_file}\nimages:   {images_dir}")

    out_image_dir = Path(out_image_dir)
    records, saved = [], 0

    with open(captions_file, "r", encoding="utf-8") as f:
        head = f.readline()
        f.seek(0)
        reader = csv.reader(f)
        if head.strip().lower().startswith("image"):
            next(reader)
        for row in reader:
            if len(row) < 2:
                continue
            name, caption = row[0].split("#")[0], ",".join(row[1:]).strip()
            src = images_dir / name
            if not src.exists() or not caption:
                continue
            dst_name = f"f8k_{name}"
            try:
                if save_resized(Image.open(src), out_image_dir / dst_name):
                    saved += 1
            except Exception:
                continue
            records.append(
                {"image": dst_name, "conversations": turn(rng.choice(DESCRIBE_INSTRUCTIONS), caption)}
            )
            if max_examples and len(records) >= max_examples:
                break

    if not records:
        raise RuntimeError("0 Flickr8k records matched -- captions and images dirs don't correspond.")
    print(f"  saved {saved:,} new images")
    write_jsonl(records, out_jsonl)


# --------------------------------------------------------------------------- #
# Flickr30k (Hugging Face, images embedded in parquet)
# --------------------------------------------------------------------------- #

def prepare_flickr30k(out_jsonl, out_image_dir, max_examples=None, seed=0, captions_per_image=5):
    from datasets import load_dataset

    rng = random.Random(seed)
    out_image_dir = Path(out_image_dir)

    # nlphuji/flickr30k ships everything under a single split (named "test"),
    # with a per-row `split` column carrying the real train/val/test label.
    ds = load_dataset("nlphuji/flickr30k", split="test", streaming=True)

    records, n_img = [], 0
    for ex in ds:
        if ex.get("split") == "test":
            continue  # keep the official test images out of training
        name = f"f30k_{ex.get('img_id', n_img)}.jpg"
        try:
            save_resized(ex["image"], out_image_dir / name)
        except Exception:
            continue
        n_img += 1
        caps = ex["caption"]
        if isinstance(caps, str):
            caps = [caps]
        for c in caps[:captions_per_image]:
            c = c.strip()
            if c:
                records.append(
                    {"image": name, "conversations": turn(rng.choice(DESCRIBE_INSTRUCTIONS), c)}
                )
        if max_examples and len(records) >= max_examples:
            break

    print(f"  {n_img:,} images -> {len(records):,} caption pairs")
    write_jsonl(records, out_jsonl)


# --------------------------------------------------------------------------- #
# VQAv2 (short answers)
# --------------------------------------------------------------------------- #

def prepare_vqav2(out_jsonl, out_image_dir, max_examples=30000, split="validation"):
    """lmms-lab/VQAv2 is a script-free parquet mirror with embedded images.
    Only validation/testdev/test exist; validation is the one with answers."""
    from datasets import load_dataset

    out_image_dir = Path(out_image_dir)
    ds = load_dataset("lmms-lab/VQAv2", split=split, streaming=True)

    records = []
    for i, ex in enumerate(ds):
        if i >= max_examples:
            break
        name = f"vqa_{i:06d}.jpg"
        try:
            save_resized(ex["image"], out_image_dir / name)
        except Exception:
            continue
        answer = ex.get("multiple_choice_answer") or ex["answers"][0]["answer"]
        question = ex["question"].strip()
        records.append(
            {
                "image": name,
                # The hint is what stops one-word supervision from destroying the
                # model's ability to produce sentences.
                "conversations": turn(question + SHORT_ANSWER_HINT, str(answer).strip()),
                "style": "short",
            }
        )

    write_jsonl(records, out_jsonl)


# --------------------------------------------------------------------------- #
# LLaVA-ReCap (detailed descriptions)
# --------------------------------------------------------------------------- #

def prepare_recap(out_jsonl, out_image_dir, max_examples=20000, max_words=140):
    """lmms-lab/LLaVA-ReCap-118K: image + a 2-turn conversation whose assistant
    reply is a detailed multi-sentence description. This is what teaches the
    model to write prose instead of a single word."""
    from datasets import load_dataset

    out_image_dir = Path(out_image_dir)
    ds = load_dataset("lmms-lab/LLaVA-ReCap-118K", split="train", streaming=True)

    records = []
    for i, ex in enumerate(ds):
        if len(records) >= max_examples:
            break
        name = f"recap_{i:06d}.jpg"
        try:
            save_resized(ex["image"], out_image_dir / name)
        except Exception:
            continue

        convs = []
        for t in ex["conversations"]:
            val = t["value"].strip()
            if t["from"] in ("gpt", "assistant"):
                words = val.split()
                if len(words) > max_words:  # keep sequence lengths bounded
                    val = " ".join(words[:max_words]).rstrip(",;: ") + "."
            convs.append({"from": t["from"], "value": val})

        if not any(t["from"] in ("gpt", "assistant") and t["value"] for t in convs):
            continue
        records.append({"image": name, "conversations": convs, "style": "long"})

    write_jsonl(records, out_jsonl)


# --------------------------------------------------------------------------- #
# Mix
# --------------------------------------------------------------------------- #

def mix(inputs, out_jsonl, seed=0, caps=None):
    rng = random.Random(seed)
    all_recs = []
    for idx, path in enumerate(inputs):
        recs = [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
        limit = caps[idx] if caps and idx < len(caps) and caps[idx] > 0 else len(recs)
        rng.shuffle(recs)
        recs = recs[:limit]
        print(f"  {path}: {len(recs):,}")
        all_recs.extend(recs)
    rng.shuffle(all_recs)
    write_jsonl(all_recs, out_jsonl)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=["flickr8k", "flickr30k", "vqav2", "recap", "mix"], required=True)
    p.add_argument("--kaggle_input_dir", default="/kaggle/input/flickr8k")
    p.add_argument("--out_jsonl", required=True)
    p.add_argument("--out_image_dir", default="data/images")
    p.add_argument("--max_examples", type=int, default=None)
    p.add_argument("--inputs", nargs="*", default=[], help="mix: input jsonl paths")
    p.add_argument("--caps", nargs="*", type=int, default=None, help="mix: per-input row cap")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    if a.task == "flickr8k":
        prepare_flickr8k(a.kaggle_input_dir, a.out_jsonl, a.out_image_dir, a.max_examples, a.seed)
    elif a.task == "flickr30k":
        prepare_flickr30k(a.out_jsonl, a.out_image_dir, a.max_examples, a.seed)
    elif a.task == "vqav2":
        prepare_vqav2(a.out_jsonl, a.out_image_dir, a.max_examples or 30000)
    elif a.task == "recap":
        prepare_recap(a.out_jsonl, a.out_image_dir, a.max_examples or 20000)
    else:
        mix(a.inputs, a.out_jsonl, a.seed, a.caps)


if __name__ == "__main__":
    main()
