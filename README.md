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
