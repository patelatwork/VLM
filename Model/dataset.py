"""
dataset.py

JSONL schema (LLaVA-style, one JSON object per line):

  {"image": "images/0001.jpg",
   "conversations": [{"from": "human", "value": "<image>\nWhat is the man doing?"},
                     {"from": "gpt",   "value": "He is riding a bicycle."}]}

The literal marker `<image>` inside a human turn is expanded to
`num_visual_tokens` copies of the model's real image placeholder token before
tokenisation. If no marker is present it is prepended to the first human turn.

Two things here fix concrete failures in the first version:

*   Loss is computed on assistant turns only, located by tokenising growing
    conversation prefixes with `apply_chat_template`. This is version-proof --
    it never hardcodes Qwen's `<|im_start|>` layout -- and it supports
    multi-turn conversations, which the old question/answer schema could not.

*   `SHORT_ANSWER_HINT` is appended to VQA-style questions at data-prep time
    (the LLaVA-1.5 trick). Without it, training on VQAv2's one-word answers
    teaches the model that *every* reply is one word, and it permanently loses
    the ability to write a sentence. With it, answer length becomes something
    the prompt controls at inference time.
"""

import json
import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

IMAGE_MARKER = "<image>"

# Must be identical at train and inference time -- import it, don't retype it.
SYSTEM_PROMPT = "You are a helpful assistant that can see and understand images."

SHORT_ANSWER_HINT = "\nAnswer the question using a single word or phrase."

DESCRIBE_INSTRUCTIONS = [
    "Describe this image.",
    "What is happening in this image?",
    "Write a short caption for this image.",
    "Describe the image in detail.",
    "What do you see in this picture?",
    "Provide a description of this image.",
]


class VLMDataset(Dataset):
    def __init__(
        self,
        jsonl_path: str,
        image_root: str,
        tokenizer,
        image_processor,
        num_visual_tokens: int,
        image_token: str,
        max_length: int = 1024,
    ):
        self.examples = [
            json.loads(l) for l in Path(jsonl_path).read_text(encoding="utf-8").splitlines() if l.strip()
        ]
        self.image_root = Path(image_root)
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_length = max_length
        self.image_token = image_token
        self.image_placeholder = image_token * num_visual_tokens

    def __len__(self):
        return len(self.examples)

    def _to_messages(self, ex):
        turns = ex["conversations"]
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        seen_image = False
        for t in turns:
            role = "user" if t["from"] in ("human", "user") else "assistant"
            content = t["value"]
            if role == "user":
                if IMAGE_MARKER in content:
                    content = content.replace(IMAGE_MARKER, self.image_placeholder)
                    seen_image = True
                elif not seen_image:
                    content = f"{self.image_placeholder}\n{content}"
                    seen_image = True
            messages.append({"role": role, "content": content})
        return messages

    def _encode(self, text, **kw):
        return self.tokenizer(text, add_special_tokens=False, **kw)["input_ids"]

    def __getitem__(self, idx):
        ex = self.examples[idx]

        image = Image.open(self.image_root / ex["image"]).convert("RGB")
        pixel_values = self.image_processor(images=image, return_tensors="pt")["pixel_values"][0]

        messages = self._to_messages(ex)

        input_ids: list[int] = []
        labels: list[int] = []
        prev_len = 0

        for i, msg in enumerate(messages):
            if msg["role"] != "assistant":
                continue
            # Everything up to and including the "<|im_start|>assistant\n" header.
            prompt_text = self.tokenizer.apply_chat_template(
                messages[:i], add_generation_prompt=True, tokenize=False
            )
            full_text = self.tokenizer.apply_chat_template(
                messages[: i + 1], add_generation_prompt=False, tokenize=False
            )
            prompt_ids = self._encode(prompt_text)
            full_ids = self._encode(full_text)

            # Context since the previous assistant turn is supervised as -100,
            # the assistant's own tokens carry the loss.
            input_ids.extend(full_ids[prev_len:])
            labels.extend([-100] * (len(prompt_ids) - prev_len))
            labels.extend(full_ids[len(prompt_ids):])
            prev_len = len(full_ids)

        input_ids = input_ids[: self.max_length]
        labels = labels[: self.max_length]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "pixel_values": pixel_values,
            "n_supervised": int(sum(1 for l in labels if l != -100)),
        }


def make_collate_fn(pad_token_id: int):
    """Right-padded collation. Causal attention + right padding means the pad
    positions cannot influence any real token, and they are masked out of the
    loss and the attention mask regardless."""

    def collate_fn(batch):
        # Drop examples whose answer was entirely truncated away; a batch with
        # zero supervised tokens produces a NaN loss.
        batch = [b for b in batch if b["n_supervised"] > 0] or batch[:1]

        max_len = max(b["input_ids"].size(0) for b in batch)
        n = len(batch)

        input_ids = torch.full((n, max_len), pad_token_id, dtype=torch.long)
        labels = torch.full((n, max_len), -100, dtype=torch.long)
        attention_mask = torch.zeros((n, max_len), dtype=torch.long)

        for i, b in enumerate(batch):
            L = b["input_ids"].size(0)
            input_ids[i, :L] = b["input_ids"]
            labels[i, : b["labels"].size(0)] = b["labels"]
            attention_mask[i, :L] = 1

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
        }

    return collate_fn


def build_prompt(tokenizer, question: str, image_placeholder: str, short_answer: bool = False) -> str:
    """Single source of truth for inference-time prompt construction."""
    if short_answer:
        question = question.rstrip() + SHORT_ANSWER_HINT
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"{image_placeholder}\n{question}"},
    ]
    return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)


def random_describe_instruction(rng: random.Random | None = None) -> str:
    return (rng or random).choice(DESCRIBE_INSTRUCTIONS)
