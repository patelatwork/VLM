# TinyVLM

A small LLaVA-style VLM: SigLIP2 vision encoder → MLP projector → Qwen2.5-0.5B-Instruct with LoRA.

```
Model/model.py                architecture
Model/dataset.py              conversation schema, prompt construction, collation
Training/prepare_data.py      builds the JSONL corpora
Training/train.py             both stages (--stage 1 | 2)
Inference/inference.py        load + answer
Inference/diagnose_grounding.py   is the model actually looking at the image?
Inference/app_gradio.py       demo UI
push_to_hub.py                upload projector + adapter
vlm-train-v2.ipynb            Kaggle pipeline end to end
```

Quick start: open `vlm-train-v2.ipynb` on Kaggle (GPU T4 x2, Internet on, with the
`adityajn105/flickr8k` dataset attached).

**Before trusting any checkpoint, run the grounding check:**

```bash
python Inference/diagnose_grounding.py \
    --projector_ckpt ckpt/stage2/projector.pt \
    --lora_dir ckpt/stage2/lora_adapter \
    --images a.jpg b.jpg c.jpg
```

Same question, several different images. If the answers are identical, the LLM is
answering from the question prior and the projector carries no signal — that is a
failed alignment stage, not a decoding-parameter problem.

---

## Example: A Single Forward Pass Through a LLaVA-Style VLM

### Setup

- **Vision encoder:** SigLIP2-base, patch size 16, input image 224×224
- **LLM:** Qwen2.5-0.5B, hidden dim 896
- **Task:** `"What is the man doing in this image?"` → `"He is riding a bicycle."`

---

### Step 1 — Image → Patches → Embeddings

A 224×224 image with patch size 16 gives:

$$(224/16) \times (224/16) = 14 \times 14 = 196 \text{ patches}$$

SigLIP2's ViT encodes this into:

```
image_embeds.shape = [1, 196, 768]   # [batch, num_patches, vision_dim]
```

Each of the 196 patches is a 768-dim vector describing a 16×16 chunk of the image.

---

### Step 2 — Projector

Qwen2.5-0.5B expects token embeddings of dim 896, not 768. So we pass the 196 patch vectors through a small MLP:

```python
projector = nn.Sequential(
    nn.Linear(768, 896),
    nn.GELU(),
    nn.Linear(896, 896)
)

visual_tokens = projector(image_embeds)   # [1, 196, 896]
```

Now we have 196 "pseudo-word" vectors, each 896-dim — same shape as a real text token embedding would be.

---

### Step 3 — Text Tokenization

The prompt gets tokenized normally, but with a placeholder for where the image goes:

```
"<image> What is the man doing in this image?"
```

Suppose that tokenizes to 10 text tokens (including the `<image>` placeholder token). We look up their embeddings:

```
text_embeds.shape = [1, 10, 896]
```

---

### Step 4 — Splice Visual Tokens into the Sequence

Wherever the `<image>` placeholder token sits, we swap it out for all 196 visual tokens:

```
before: [tok0, tok1, <image>, tok2, ..., tok9]              length 10
after:  [tok0, tok1, v0, v1, ..., v195, tok2, ..., tok9]     length 205
```

```
combined_embeds.shape = [1, 205, 896]
```

This is now just a normal sequence of embeddings as far as the LLM is concerned — it has no idea 196 of them came from an image.

---

### Step 5 — Feed Through the LLM, Compute Loss Only on the Answer

```python
outputs = qwen_model(inputs_embeds=combined_embeds)
logits = outputs.logits   # [1, 205, vocab_size]
```

For training, the labels tensor is the same length (205), but everything except the answer tokens (`"He is riding a bicycle."`) is masked with `-100` so the loss function ignores them:

```
labels = [-100, -100, ..., -100, h_id, e_id, is_id, riding_id, ..., -100]
```

So the model is only ever penalized for getting the *answer* right — it's not asked to "predict" the image patches or the question itself.

---

### Step 6 — Backward Pass, Stage-Dependent

| Stage | What's Frozen | What's Trained | Goal |
|---|---|---|---|
| **Stage 1** — Projector pretraining | SigLIP2, Qwen | `projector` only | Align visual vectors with the LLM's embedding space (e.g., a "bicycle wheel" vector lands near the token embedding for "bicycle") |
| **Stage 2** — Instruction tuning | SigLIP2 (usually) | Qwen (full or via LoRA) + projector | Teach the LLM to reason over visual tokens and answer conversationally, not just caption |

---

### Why This Works at All

The key trick: the LLM was never explicitly designed to accept images. It just sees 205 vectors of dim 896 and predicts the next token, same as always. The "vision understanding" lives entirely in whether the projector has learned a mapping such that a vector representing "bicycle wheel" ends up sitting near where the token embedding for "bicycle" naturally sits in Qwen's space. Stage 1 training is literally forcing that alignment via the captioning loss.

---

## How the implementation differs from the walkthrough above

The walkthrough is the clean mental model. Four places where the shipped code
deliberately departs from it, each because the naive version failed in practice:

**196 visual tokens → 49.** A 2×2 pixel-shuffle folds each patch neighbourhood
into one token (the channel dim absorbs what the sequence dim gives up, so
nothing is discarded). With a 0.5B LLM and a single-session training budget,
sample efficiency beats spatial resolution — and it raises the ratio of
supervised tokens to total tokens by ~4×.

**Splicing → repeat-and-scatter.** Rather than inserting one `<image>` token and
splicing embeddings around it, the placeholder is repeated 49 times in
`input_ids` and overwritten via `masked_scatter`. Sequence length never changes,
so `labels` and `attention_mask` line up by construction instead of by index
arithmetic — which is where the original padding and masking bugs lived.

**A new `<image>` token → Qwen's existing `<|image_pad|>`.** Adding a token
forces `resize_token_embeddings`, which *shrinks* Qwen's embedding matrix
(151936 rows on disk, 151666 known to the tokenizer) and makes PEFT serialise
the whole table into the adapter — 1.1 GB instead of ~35 MB.

**The projector ends in a scale-matched LayerNorm.** Raw SigLIP hidden states
have ~104× the per-token norm of Qwen's text embeddings. Dropped in unscaled,
the visual tokens sit far outside the manifold the LLM was trained on, and the
projector spends its whole budget just learning to rescale. Initialising the
LayerNorm gain to the LLM's own embedding std puts the ratio at ~1.0× from step
zero.

## Answer length is controlled by the prompt

Training on VQAv2 alone teaches the model that every answer is one word — its
targets tokenize to `['red', '<|im_end|>']`, two supervised tokens. Stage 2 mixes
one-word VQA with multi-sentence descriptions, and VQA examples carry an explicit
hint, so length becomes a runtime choice:

```python
answer(model, img, "What color is the bus?")                    # "The bus is red."
answer(model, img, "What color is the bus?", short_answer=True) # "red"
```
