"""
model.py

Defines the VLM architecture:
  SigLIP2 vision encoder (frozen or LoRA) -> MLP projector -> Qwen2.5 LLM (frozen or LoRA)

The design follows the LLaVA-style "unified token space" approach: image patch
embeddings are projected into the LLM's embedding dimension and spliced into
the text token sequence at a reserved <image> placeholder position.
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, AutoImageProcessor

VISION_MODEL_NAME = "google/siglip2-base-patch16-224"
LLM_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
IMAGE_TOKEN = "<image>"


class Projector(nn.Module):
    """Maps vision embedding dim -> LLM embedding dim."""

    def __init__(self, vision_dim: int, llm_dim: int, hidden_mult: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(vision_dim, llm_dim * hidden_mult),
            nn.GELU(),
            nn.Linear(llm_dim * hidden_mult, llm_dim),
        )

    def forward(self, x):
        return self.net(x)


class TinyVLM(nn.Module):
    def __init__(
        self,
        vision_model_name: str = VISION_MODEL_NAME,
        llm_model_name: str = LLM_MODEL_NAME,
        freeze_vision: bool = True,
        freeze_llm: bool = True,
    ):
        super().__init__()

        # --- Vision encoder ---
        self.vision_encoder = AutoModel.from_pretrained(vision_model_name).vision_model
        self.image_processor = AutoImageProcessor.from_pretrained(vision_model_name)
        vision_dim = self.vision_encoder.config.hidden_size

        # --- LLM ---
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
        if IMAGE_TOKEN not in self.tokenizer.get_vocab():
            self.tokenizer.add_special_tokens({"additional_special_tokens": [IMAGE_TOKEN]})
        self.llm = AutoModelForCausalLM.from_pretrained(llm_model_name)
        self.llm.resize_token_embeddings(len(self.tokenizer))
        llm_dim = self.llm.config.hidden_size

        self.image_token_id = self.tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)

        # --- Projector (always trainable) ---
        self.projector = Projector(vision_dim, llm_dim)

        if freeze_vision:
            for p in self.vision_encoder.parameters():
                p.requires_grad = False

        if freeze_llm:
            for p in self.llm.parameters():
                p.requires_grad = False

    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """pixel_values: [B, 3, H, W] -> visual tokens [B, num_patches, llm_dim]"""
        vision_out = self.vision_encoder(pixel_values=pixel_values).last_hidden_state
        return self.projector(vision_out)

    def build_inputs_embeds(self, input_ids: torch.Tensor, pixel_values: torch.Tensor):
        """
        Splice visual tokens into the text embedding sequence at the <image>
        placeholder position. Assumes exactly one <image> token per example
        and a single image per example (extend for multi-image later).
        """
        text_embeds = self.llm.get_input_embeddings()(input_ids)  # [B, T, D]
        visual_tokens = self.encode_image(pixel_values)  # [B, N, D]

        batch_embeds = []
        batch_attention_masks = []
        for b in range(input_ids.size(0)):
            ids = input_ids[b]
            img_pos = (ids == self.image_token_id).nonzero(as_tuple=True)[0]
            if len(img_pos) == 0:
                # no image placeholder found -- text-only fallback
                batch_embeds.append(text_embeds[b])
                batch_attention_masks.append(torch.ones(text_embeds.size(1), device=ids.device))
                continue
            pos = img_pos[0].item()
            merged = torch.cat(
                [text_embeds[b, :pos], visual_tokens[b], text_embeds[b, pos + 1:]],
                dim=0,
            )
            batch_embeds.append(merged)
            batch_attention_masks.append(torch.ones(merged.size(0), device=ids.device))

        # pad to max length in batch
        max_len = max(e.size(0) for e in batch_embeds)
        d = batch_embeds[0].size(-1)
        padded = torch.zeros(len(batch_embeds), max_len, d, device=input_ids.device, dtype=batch_embeds[0].dtype)
        attn = torch.zeros(len(batch_embeds), max_len, device=input_ids.device)
        for i, e in enumerate(batch_embeds):
            padded[i, : e.size(0)] = e
            attn[i, : e.size(0)] = 1
        return padded, attn

    def forward(self, input_ids, pixel_values, labels=None):
        inputs_embeds, attention_mask = self.build_inputs_embeds(input_ids, pixel_values)

        # labels must be padded/expanded the same way input was (done in dataset collate_fn)
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )
        return outputs

    @torch.no_grad()
    def generate(self, input_ids, pixel_values, max_new_tokens=64, **gen_kwargs):
        inputs_embeds, attention_mask = self.build_inputs_embeds(input_ids, pixel_values)
        return self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            **gen_kwargs,
        )
