from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from PIL import Image
from torchvision.transforms import transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_image_transform(image_resolution: int):
    return transforms.Compose(
        [
            transforms.Resize((image_resolution, image_resolution)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def load_image_tensor(
    image_path: str,
    image_resolution: int,
    device: torch.device,
) -> torch.Tensor:
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    transform = build_image_transform(image_resolution)
    image = Image.open(path).convert("RGB")
    return transform(image).unsqueeze(0).to(device)


def _tokenize_sentence(tokenizer_obj, sentence: str, seq_len: int) -> list:
    tokenizer = tokenizer_obj.tokenizer
    encoded = tokenizer(
        [sentence],
        padding="max_length",
        max_length=seq_len,
        truncation=True,
        add_special_tokens=False,
    )
    return encoded["input_ids"][0]


def build_condition_tensors(
    tokenizer_obj,
    question: str,
    seq_len: int,
    device: torch.device,
    answer_seed_text: Optional[str] = None,
    image_name: str = "input",
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    question = (question or "").strip()
    answer_seed_text = (answer_seed_text or "").strip()

    q_ids = _tokenize_sentence(tokenizer_obj, question, seq_len)
    a_ids = _tokenize_sentence(tokenizer_obj, answer_seed_text, seq_len)

    input_ids = q_ids + a_ids
    text_mask = [0] * len(q_ids) + [1] * len(a_ids)

    cond = {
        "input_q_id": torch.tensor([q_ids], dtype=torch.long, device=device),
        "input_a_id": torch.tensor([a_ids], dtype=torch.long, device=device),
        "input_ids": torch.tensor([input_ids], dtype=torch.long, device=device),
        "input_mask": torch.tensor([text_mask], dtype=torch.long, device=device),
        "image_name": [image_name],
    }

    # The model pops input_q_id inside forward; keep a dedicated copy for x_start prep.
    cond_for_ddpm = {
        "input_q_id": cond["input_q_id"].clone(),
    }
    return cond, cond_for_ddpm
