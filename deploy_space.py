"""
deploy_space.py

Assembles and uploads the Gradio Space.

A Space is its own git repo. Pushing new weights to the *model* repo does not
rebuild it, and restarting it by hand does not help either -- the Space runs
whatever code it was last given, and run-2 checkpoints are not loadable by
run-1 code (the projector gained an `out_norm` and its first Linear went from
768 to 3072 input features). Code and weights have to move together.

This copies the canonical modules out of the repo rather than keeping a second
flattened copy under version control -- duplicated source is what let the
notebook and repo training scripts drift apart in the first place.

Usage:
    huggingface-cli login          # or export HF_TOKEN
    python deploy_space.py --space_id dhruvpatel93/tinyvlm --model_repo dhruvpatel93/tinyvlm-vqa
"""

import argparse
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, create_repo

ROOT = Path(__file__).resolve().parent

# canonical source -> name in the flattened Space root
FILES = {
    ROOT / "Model" / "model.py": "model.py",
    ROOT / "Model" / "dataset.py": "dataset.py",
    ROOT / "Inference" / "inference.py": "inference.py",
    ROOT / "Inference" / "app_gradio.py": "app.py",
    ROOT / "Inference" / "requirements.txt": "requirements.txt",
}

CARD = """---
title: TinyVLM
emoji: 🖼️
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
---

# TinyVLM

SigLIP2 + Qwen2.5-0.5B, LLaVA-style. Ask a question about an image.

Weights are pulled at startup from [`{model_repo}`](https://huggingface.co/{model_repo})
via the `HF_REPO_ID` variable, so pushing new weights + restarting this Space is
enough for a weights-only change. A change to the *architecture* needs this
Space's code redeployed too -- run `deploy_space.py` from the training repo.

**Answer style** is prompt-controlled: "Descriptive" writes sentences,
"Short answer (VQA)" appends the hint the model was trained with and returns a
word or two.

On free CPU hardware a response takes roughly 10-30 s.
"""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--space_id", required=True, help="e.g. dhruvpatel93/tinyvlm")
    p.add_argument("--model_repo", required=True, help="repo holding projector.pt + lora_adapter/")
    p.add_argument("--private", action="store_true")
    p.add_argument("--dry_run", action="store_true", help="assemble locally, don't upload")
    a = p.parse_args()

    missing = [str(s) for s in FILES if not s.exists()]
    if missing:
        raise SystemExit("missing source files:\n  " + "\n  ".join(missing))

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for src, dst in FILES.items():
            shutil.copy(src, tmp / dst)
        (tmp / "README.md").write_text(CARD.format(model_repo=a.model_repo), encoding="utf-8")

        print("Space contents:")
        for f in sorted(tmp.iterdir()):
            print(f"   {f.name:20s} {f.stat().st_size / 1024:6.1f} KB")

        if a.dry_run:
            out = ROOT / "_space_preview"
            if out.exists():
                shutil.rmtree(out)
            shutil.copytree(tmp, out)
            print(f"\ndry run -- assembled at {out}")
            return

        api = HfApi()
        create_repo(a.space_id, repo_type="space", space_sdk="gradio",
                    private=a.private, exist_ok=True)
        # The app reads this at startup instead of hardcoding the weights repo.
        api.add_space_variable(a.space_id, "HF_REPO_ID", a.model_repo)
        api.upload_folder(folder_path=str(tmp), repo_id=a.space_id, repo_type="space",
                          commit_message="Deploy TinyVLM app + architecture")

    print(f"\nhttps://huggingface.co/spaces/{a.space_id}")
    print("The Space rebuilds automatically on this commit; watch its Logs tab.")


if __name__ == "__main__":
    main()
