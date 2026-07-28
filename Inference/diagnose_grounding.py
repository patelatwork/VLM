"""
diagnose_grounding.py

The check that would have caught the original failure before it ever reached a
Space: run the SAME question against several DIFFERENT images and compare.

If the outputs are identical across genuinely different images, the LLM is
answering from the question prior alone and the projector carries no usable
signal. No amount of decoding-parameter tuning fixes that -- it means the
alignment stage did not work and has to be redone.

Usage:
    python diagnose_grounding.py \
        --projector_ckpt ckpt/stage2/projector.pt \
        --lora_dir ckpt/stage2/lora_adapter \
        --images a.jpg b.jpg c.jpg
"""

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image

for _p in (Path(__file__).resolve().parent.parent / "Model", Path(__file__).resolve().parent):
    sys.path.insert(0, str(_p))

from inference import load_model, answer   # noqa: E402


@torch.no_grad()
def visual_tokens(model, image, device):
    px = model.image_processor(images=image.convert("RGB"), return_tensors="pt")["pixel_values"].to(device)
    return model.encode_image(px)[0].float()   # [num_visual_tokens, llm_dim]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--projector_ckpt", required=True)
    p.add_argument("--lora_dir", default=None)
    p.add_argument("--images", nargs="+", required=True)
    p.add_argument("--question", default="What is in this image?")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()

    model = load_model(a.projector_ckpt, a.lora_dir, a.device)
    images = [Image.open(i).convert("RGB") for i in a.images]
    names = [Path(i).name for i in a.images]

    blank = Image.new("RGB", (224, 224), (0, 0, 0))
    noise = Image.effect_noise((224, 224), 64).convert("RGB")

    print("=" * 74)
    print("TEST 1 -- same question, different images")
    print("=" * 74)
    outs = []
    for name, img in list(zip(names, images)) + [("<blank>", blank), ("<noise>", noise)]:
        long_a = answer(model, img, a.question, a.device, short_answer=False)
        short_a = answer(model, img, a.question, a.device, short_answer=True)
        outs.append((name, long_a, short_a))
        print(f"  {name:24s} long : {long_a[:120]!r}")
        print(f"  {'':24s} short: {short_a[:60]!r}")

    real_long = [o[1] for o in outs[: len(images)]]
    real_short = [o[2] for o in outs[: len(images)]]
    print(f"\n  distinct long answers  : {len(set(real_long))} / {len(real_long)}")
    print(f"  distinct short answers : {len(set(real_short))} / {len(real_short)}")
    if len(set(real_long)) == 1:
        print("  >>> VERDICT: image completely ignored (pure language prior).")
    elif len(set(real_long)) < len(real_long):
        print("  >>> VERDICT: weak grounding -- some images collapse to the same answer.")
    else:
        print("  >>> Outputs vary with the image. Grounding is working.")

    avg_len = sum(len(o[1].split()) for o in outs[: len(images)]) / max(len(images), 1)
    print(f"  mean long-answer length: {avg_len:.1f} words "
          f"({'TOO SHORT -- collapsed to VQA style' if avg_len < 4 else 'ok'})")

    print()
    print("=" * 74)
    print("TEST 2 -- projector output scale vs. LLM embedding scale")
    print("=" * 74)
    vs = [visual_tokens(model, im, a.device) for im in images]
    emb = model.llm.get_input_embeddings().weight
    emb = emb[: len(model.tokenizer)].float()
    v_norm = vs[0].norm(dim=-1).mean().item()
    t_norm = emb.norm(dim=-1).mean().item()
    print(f"  visual tokens per image : {vs[0].shape[0]}")
    print(f"  visual token  std / norm: {vs[0].std().item():.4f} / {v_norm:.3f}")
    print(f"  text embedding std / norm: {emb.std().item():.4f} / {t_norm:.3f}")
    print(f"  norm ratio (visual/text): {v_norm / t_norm:.2f}x")
    if not 0.33 < v_norm / t_norm < 3.0:
        print("  >>> Visual tokens sit outside the LLM's embedding distribution.")

    print()
    print("=" * 74)
    print("TEST 3 -- do different images produce different visual tokens?")
    print("=" * 74)
    # Deliberately NOT cosine over mean-pooled tokens. Projected visual tokens
    # share a large constant component, so pooled cosine sits at ~0.99 even for
    # a perfectly well-grounded model -- it reports collapse that isn't there.
    # Relative L2 distance over the full token matrix is the honest measure:
    # 0 = identical, ~1.4 = unrelated.
    V = torch.stack([v.flatten() for v in vs])
    rels = []
    for i in range(len(vs)):
        for j in range(i + 1, len(vs)):
            denom = (V[i].norm() + V[j].norm()) / 2
            rel = ((V[i] - V[j]).norm() / denom).item()
            rels.append(rel)
            print(f"  rel_dist({names[i]}, {names[j]}) = {rel:.4f}")

    varying = (V - V.mean(0)).norm() / V.norm()
    print(f"\n  image-dependent fraction of the representation: {varying:.4f}")
    if rels and min(rels) < 0.02:
        print("  >>> Projector has collapsed to a near-constant output.")
    elif rels:
        print(f"  min rel_dist = {min(rels):.4f} -- projector output varies with the image.")


if __name__ == "__main__":
    main()
