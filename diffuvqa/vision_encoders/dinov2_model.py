# CHANGED: New file — replaces clip_model.py as the vision encoder.
# Uses facebook/dinov2-base from HuggingFace instead of OpenAI CLIP.
# Output shape: [B, num_patches+1, 768] — identical layout to CLIP ViT,
# so the transpose + MLP pipeline in feature_fusion.forward() is unchanged.
# Normalization (ImageNet mean/std) is applied in the dataloader (train.py),
# which matches DINOv2's training distribution — no change needed there either.

import torch
import torch.nn as nn
from transformers import Dinov2Model


class DINOv2Encoder(nn.Module):
    def __init__(self, model_name: str = "facebook/dinov2-base"):
        super().__init__()
        # CHANGED: loads DINOv2 weights from HuggingFace Hub
        self.model = Dinov2Model.from_pretrained(model_name)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 3, H, W]  (ImageNet-normalized by the dataloader)
        # last_hidden_state: [B, num_patches+1, hidden_size=768]
        # The +1 is the CLS token prepended at position 0, matching CLIP's convention.
        outputs = self.model(pixel_values=x)
        return outputs.last_hidden_state


def build_dinov2_model(model_name: str = "facebook/dinov2-base") -> DINOv2Encoder:
    # CHANGED: drop-in replacement for clip_model.build_model().
    # `model_name` is a HuggingFace model ID (set via args.image_encoder in config.json).
    return DINOv2Encoder(model_name)
