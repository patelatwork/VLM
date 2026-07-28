"""
model.py

TinyVLM architecture:
  SigLIP2 vision encoder (frozen) -> pixel-shuffle -> MLP projector -> Qwen2.5 LLM (LoRA)

Design notes / differences from the first version, all of which mattered:

1.  We reuse Qwen2.5's existing `<|image_pad|>` token instead of adding a new
    `<image>` token. Adding a token forced `resize_token_embeddings(151666)`,
    which actually *shrank* the embedding matrix (Qwen ships 151936 rows, the
    tokenizer only knows 151665) and forced PEFT to serialize the whole
    embedding table into the adapter -- a 1.1 GB adapter instead of ~9 MB.

2.  The image placeholder is repeated `num_visual_tokens` times in `input_ids`
    rather than appearing once and being spliced. That keeps `input_ids`,
    `labels` and `attention_mask` the same length as `inputs_embeds`, so
    padding and masking are handled by ordinary collation instead of bespoke
    index arithmetic.

3.  The projector ends in a LayerNorm whose gain is initialised to the LLM's
    own token-embedding std. Raw SigLIP hidden states are ~10-20x larger in
    norm than Qwen token embeddings; without this the visual tokens land far
    outside the embedding manifold and the projector burns its (short)
    training budget just learning to rescale.

4.  Pixel-shuffle folds each 2x2 patch neighbourhood into one token
    (196 -> 49 tokens, no information discarded). With a 0.5B LLM and a
    single-session training budget, sample efficiency beats spatial
    resolution -- and it makes the supervised-token / total-token ratio
    roughly 4x better.
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, AutoImageProcessor

VISION_MODEL_NAME = "google/siglip2-base-patch16-224"
LLM_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

# Already present in the Qwen2.5 vocabulary (reserved for Qwen2-VL), so using it
# needs no vocab resize and no embedding surgery.
IMAGE_TOKEN = "<|image_pad|>"


def _load_vision_tower(name: str, dtype):
    """
    Return just the ViT stack.

    Deliberately routed through AutoModel so the architecture comes from the
    checkpoint's own config. Hand-picking a class is a trap: despite the name,
    `google/siglip2-base-patch16-224` is stored as `siglip_vision_model`, and
    loading it with `Siglip2VisionModel` does NOT raise -- it reports
    "Reinit due to size mismatch" for `embeddings.patch_embedding` and
    `embeddings.position_embedding` and hands back an encoder whose input layer
    is randomly initialised. That produces a vision tower that emits noise, and
    nothing downstream would tell you.

    Keeping only the returned submodule makes the parent (and SigLIP's unused
    text tower) unreachable, so it is freed on the next collection anyway.
    """
    return AutoModel.from_pretrained(name, dtype=dtype).vision_model


def _assert_vision_tower_loaded(vision_encoder, name: str):
    """Cheap guard against a silently reinitialised patch embedding (see
    `_load_vision_tower`). A real conv patch embedding is 4-D [D, 3, P, P];
    the reinit path leaves a 2-D weight behind."""
    w = getattr(getattr(vision_encoder, "embeddings", None), "patch_embedding", None)
    if w is None:
        return
    w = w.weight
    if w.dim() != 4 or w.shape[1] != 3:
        raise RuntimeError(
            f"vision tower for {name!r} did not load its pretrained patch embedding "
            f"(got weight shape {tuple(w.shape)}, expected [D, 3, P, P]). The encoder "
            "would emit noise. Check the transformers version / checkpoint pairing."
        )


def pixel_shuffle(x: torch.Tensor, stride: int) -> torch.Tensor:
    """
    [B, N, D] -> [B, N/stride^2, D*stride^2], folding each stride x stride
    neighbourhood of patches into a single token. Unlike pooling this is
    lossless -- the channel dim absorbs what the sequence dim gives up.
    """
    if stride == 1:
        return x
    b, n, d = x.shape
    h = w = int(n**0.5)
    if h * w != n:
        raise ValueError(f"expected a square patch grid, got {n} patches")
    if h % stride or w % stride:
        raise ValueError(f"{h}x{w} patch grid is not divisible by stride {stride}")

    x = x.view(b, h, w // stride, d * stride)
    x = x.permute(0, 2, 1, 3).contiguous()
    x = x.view(b, w // stride, h // stride, d * stride * stride)
    x = x.permute(0, 2, 1, 3).contiguous()
    return x.view(b, (h // stride) * (w // stride), d * stride * stride)


class Projector(nn.Module):
    """Maps vision embedding dim -> LLM embedding dim, at the LLM's own scale."""

    def __init__(self, vision_dim: int, llm_dim: int, pool_stride: int = 2):
        super().__init__()
        self.pool_stride = pool_stride
        in_dim = vision_dim * pool_stride * pool_stride
        self.net = nn.Sequential(
            nn.Linear(in_dim, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim),
        )
        # Gain is re-initialised to the LLM embedding std by TinyVLM.__init__.
        self.out_norm = nn.LayerNorm(llm_dim)

    def forward(self, x):
        x = pixel_shuffle(x, self.pool_stride)
        return self.out_norm(self.net(x))


class TinyVLM(nn.Module):
    def __init__(
        self,
        vision_model_name: str = VISION_MODEL_NAME,
        llm_model_name: str = LLM_MODEL_NAME,
        freeze_vision: bool = True,
        freeze_llm: bool = True,
        pool_stride: int = 2,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()

        # --- Vision encoder ---
        # Load the vision tower alone. `AutoModel(...).vision_model` also
        # materialises SigLIP's unused text tower (~450 MB) and emits confusing
        # vocab warnings from its CLIP-inherited text config.
        self.vision_encoder = _load_vision_tower(vision_model_name, dtype)
        self.image_processor = AutoImageProcessor.from_pretrained(vision_model_name)
        vision_cfg = self.vision_encoder.config
        vision_dim = vision_cfg.hidden_size
        _assert_vision_tower_loaded(self.vision_encoder, vision_model_name)

        # --- LLM ---
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.llm = AutoModelForCausalLM.from_pretrained(llm_model_name, dtype=dtype)
        llm_dim = self.llm.config.hidden_size

        self.image_token_id = self.tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)
        if self.image_token_id is None or self.image_token_id == self.tokenizer.unk_token_id:
            raise RuntimeError(
                f"{IMAGE_TOKEN!r} is not in the {llm_model_name} vocabulary. "
                "Pick another reserved token rather than adding one -- see the "
                "module docstring for why adding one is a trap."
            )

        # --- Projector ---
        self.pool_stride = pool_stride
        grid = vision_cfg.image_size // vision_cfg.patch_size
        self.num_visual_tokens = (grid // pool_stride) ** 2

        self.projector = Projector(vision_dim, llm_dim, pool_stride).to(dtype)

        # Match the LLM's embedding scale so visual tokens start life inside the
        # distribution the LLM was trained on.
        with torch.no_grad():
            emb = self.llm.get_input_embeddings().weight
            # Rows beyond the tokenizer's range are unused padding in Qwen and
            # are near-zero; they would skew the statistic.
            target_std = emb[: len(self.tokenizer)].float().std().item()
            self.projector.out_norm.weight.fill_(target_std)
            self.projector.out_norm.bias.zero_()
        self.embed_std = target_std

        if freeze_vision:
            for p in self.vision_encoder.parameters():
                p.requires_grad = False
        if freeze_llm:
            for p in self.llm.parameters():
                p.requires_grad = False

    # ------------------------------------------------------------------ #

    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """[B, 3, H, W] -> visual tokens [B, num_visual_tokens, llm_dim]"""
        # When the encoder is frozen there is nothing to backprop into it, so
        # skipping its activation graph is a large memory and speed win.
        vision_frozen = not any(p.requires_grad for p in self.vision_encoder.parameters())
        with torch.set_grad_enabled(not vision_frozen and torch.is_grad_enabled()):
            vision_out = self.vision_encoder(pixel_values=pixel_values).last_hidden_state
        return self.projector(vision_out)

    def build_inputs_embeds(self, input_ids: torch.Tensor, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Replace every `<|image_pad|>` embedding with the corresponding projected
        visual token. Sequence length is unchanged, so labels and attention_mask
        built by the collator line up by construction.
        """
        inputs_embeds = self.llm.get_input_embeddings()(input_ids)

        if pixel_values is None:
            return inputs_embeds

        visual = self.encode_image(pixel_values)  # [B, N, D]
        mask = input_ids == self.image_token_id

        expected = visual.shape[0] * visual.shape[1]
        found = int(mask.sum())
        if found != expected:
            raise ValueError(
                f"expected {expected} image placeholder tokens "
                f"({visual.shape[0]} images x {visual.shape[1]} tokens) but found {found}. "
                "The dataset and the model disagree on num_visual_tokens -- check pool_stride."
            )

        return inputs_embeds.masked_scatter(
            mask.unsqueeze(-1), visual.to(inputs_embeds.dtype)
        )

    def forward(self, input_ids, attention_mask, pixel_values=None, labels=None):
        inputs_embeds = self.build_inputs_embeds(input_ids, pixel_values)
        return self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )

    @torch.no_grad()
    def generate(self, input_ids, attention_mask, pixel_values=None, max_new_tokens=128, **gen_kwargs):
        inputs_embeds = self.build_inputs_embeds(input_ids, pixel_values)
        gen_kwargs.setdefault("pad_token_id", self.tokenizer.pad_token_id)
        gen_kwargs.setdefault("eos_token_id", self.tokenizer.eos_token_id)
        # With inputs_embeds (and no input_ids) HF returns ONLY the new tokens.
        return self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            **gen_kwargs,
        )

    # ------------------------------------------------------------------ #

    def trainable_parameter_report(self) -> str:
        total = sum(p.numel() for p in self.parameters())
        train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        by_part = {}
        for name, mod in [
            ("vision", self.vision_encoder),
            ("projector", self.projector),
            ("llm", self.llm),
        ]:
            by_part[name] = sum(p.numel() for p in mod.parameters() if p.requires_grad)
        parts = ", ".join(f"{k}={v:,}" for k, v in by_part.items())
        return f"trainable {train:,} / {total:,} ({100 * train / total:.3f}%)  [{parts}]"
