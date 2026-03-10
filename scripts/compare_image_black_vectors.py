#!/usr/bin/env python3
"""Compare fused multimodal vectors for real images vs black replica images.

This script measures how much image content changes the question-image fused vector.
For each sample, it computes fused vectors twice:
1) real image + question
2) black image + question
and writes per-sample metrics to JSONL.
"""

import argparse
import copy
import json
import os
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F
from torchvision.transforms import transforms

from diffuvqa.utils import logger
from diffuvqa.vqa_datasets import _resolve_image_path, load_data_vqa
from shared.basic_utils import add_dict_to_argparser, create_model_and_diffusion, load_defaults_config, load_tokenizer


def _build_argparser() -> argparse.ArgumentParser:
    defaults = load_defaults_config()
    parser = argparse.ArgumentParser(description="Compare real-image and black-image fused vectors")
    add_dict_to_argparser(parser, defaults)
    parser.add_argument("--split", type=str, default="test", choices=["train", "valid", "test"])
    parser.add_argument("--max_samples", type=int, default=100)
    parser.add_argument("--device", type=str, default="auto", help="auto, cpu, or cuda")
    parser.add_argument("--out_jsonl", type=str, default="vector_compare_real_vs_black.jsonl")
    parser.add_argument("--out_pt", type=str, default="", help="Optional .pt output for full vectors")
    parser.add_argument("--vector_preview_dims", type=int, default=12)
    return parser


def _merge_training_args(args: argparse.Namespace) -> argparse.Namespace:
    if not args.model_path:
        return args

    config_path = os.path.join(os.path.dirname(args.model_path), "training_args.json")
    if not os.path.exists(config_path):
        logger.log(f"No training_args.json next to model path: {config_path}")
        return args

    with open(config_path, "r", encoding="utf-8") as f:
        training_args = json.load(f)

    # Keep explicit runtime overrides from CLI.
    preserved = {
        "model_path": args.model_path,
        "batch_size": args.batch_size,
        "split": args.split,
        "max_samples": args.max_samples,
        "device": args.device,
        "out_jsonl": args.out_jsonl,
        "out_pt": args.out_pt,
        "vector_preview_dims": args.vector_preview_dims,
        "data_dir": args.data_dir,
        "dataset": args.dataset,
        "image_dir": args.image_dir,
    }

    args.__dict__.update(training_args)
    args.__dict__.update(preserved)
    return args


def _select_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def _clone_cond_for_model(cond: Dict[str, Any]) -> Dict[str, Any]:
    cloned: Dict[str, Any] = {}
    for key, value in cond.items():
        if torch.is_tensor(value):
            cloned[key] = value.clone()
        elif isinstance(value, list):
            cloned[key] = value[:]
        elif isinstance(value, tuple):
            cloned[key] = list(value)
        else:
            cloned[key] = copy.deepcopy(value)
    return cloned


def _move_cond_to_device(cond: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    moved: Dict[str, Any] = {}
    for key, value in cond.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def _decode_question(tokenizer_wrapper: Any, token_ids: torch.Tensor) -> str:
    ids = token_ids.detach().cpu().tolist()
    if hasattr(tokenizer_wrapper, "tokenizer") and not isinstance(tokenizer_wrapper.tokenizer, dict):
        return tokenizer_wrapper.tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=True).strip()

    # Fallback for dict-based tokenizer.
    if hasattr(tokenizer_wrapper, "rev_tokenizer"):
        words = []
        for idx in ids:
            tok = tokenizer_wrapper.rev_tokenizer.get(idx, "[UNK]")
            if tok in ("[PAD]", "[END]"):
                break
            if tok != "[START]":
                words.append(tok)
        return " ".join(words).strip()

    return ""


def _vector_preview(vec: torch.Tensor, k: int) -> List[float]:
    k = max(1, min(k, vec.numel()))
    return [float(x) for x in vec[:k].detach().cpu().tolist()]


def main() -> None:
    args = _build_argparser().parse_args()
    logger.configure()
    args = _merge_training_args(args)
    device = _select_device(args.device)

    logger.log(f"Loading model from: {args.model_path}")
    logger.log(f"Using device: {device}")

    model, _ = create_model_and_diffusion(args=args)
    state_dict = torch.load(args.model_path, map_location=device)
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model = model.to(device).eval().requires_grad_(False)

    tokenizer = load_tokenizer(args)

    emb_weight = model.word_embedding.weight.clone().detach()
    emb_dim = emb_weight.size(1)
    if emb_dim == args.hidden_dim:
        model_emb = torch.nn.Embedding(
            num_embeddings=emb_weight.size(0),
            embedding_dim=emb_dim,
            _weight=emb_weight,
        ).eval().requires_grad_(False)
    else:
        model_emb = model.word_embedding.eval().requires_grad_(False)

    transform = transforms.Compose(
        [
            transforms.Resize((args.image_resolution, args.image_resolution)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    data_loader = load_data_vqa(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        args=args,
        model_emb=model_emb.cpu(),
        transform=transform,
        split=args.split,
        loaded_vocab=tokenizer,
        loop=False,
    )

    os.makedirs(os.path.dirname(args.out_jsonl) or ".", exist_ok=True)

    total_count = 0
    cos_values: List[float] = []
    l2_values: List[float] = []

    real_vectors: List[torch.Tensor] = []
    black_vectors: List[torch.Tensor] = []

    with torch.no_grad(), open(args.out_jsonl, "w", encoding="utf-8") as out_f:
        for images, cond in data_loader:
            if total_count >= args.max_samples:
                break

            images = images.to(device)
            cond_dev = _move_cond_to_device(cond, device)

            cond_real = _clone_cond_for_model(cond_dev)
            cond_black = _clone_cond_for_model(cond_dev)

            fused_real, _ = model.get_ddpm_input(images, cond_real)
            fused_black, _ = model.get_ddpm_input(torch.zeros_like(images), cond_black)

            # Pool token dimension for one vector per sample.
            vec_real = fused_real.mean(dim=1)
            vec_black = fused_black.mean(dim=1)

            cos_sim = F.cosine_similarity(vec_real, vec_black, dim=-1)
            l2_dist = torch.norm(vec_real - vec_black, p=2, dim=-1)
            delta_norm = torch.norm(vec_real - vec_black, p=2, dim=-1)
            real_norm = torch.norm(vec_real, p=2, dim=-1)
            black_norm = torch.norm(vec_black, p=2, dim=-1)

            batch_size = vec_real.size(0)
            image_names = cond.get("image_name", [""] * batch_size)

            for i in range(batch_size):
                if total_count >= args.max_samples:
                    break

                q_text = _decode_question(tokenizer, cond_dev["input_q_id"][i])
                image_name = image_names[i] if i < len(image_names) else ""
                image_path = _resolve_image_path(args, image_name)

                record = {
                    "index": total_count,
                    "question": q_text,
                    "image_name": image_name,
                    "image_path": image_path,
                    "mode_real": "question + real_image",
                    "mode_black": "question + black_image",
                    "cosine_similarity": float(cos_sim[i].item()),
                    "l2_distance": float(l2_dist[i].item()),
                    "delta_norm": float(delta_norm[i].item()),
                    "real_vector_norm": float(real_norm[i].item()),
                    "black_vector_norm": float(black_norm[i].item()),
                    "real_vector_preview": _vector_preview(vec_real[i], args.vector_preview_dims),
                    "black_vector_preview": _vector_preview(vec_black[i], args.vector_preview_dims),
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

                cos_values.append(record["cosine_similarity"])
                l2_values.append(record["l2_distance"])

                if args.out_pt:
                    real_vectors.append(vec_real[i].detach().cpu())
                    black_vectors.append(vec_black[i].detach().cpu())

                total_count += 1

    if args.out_pt and real_vectors:
        os.makedirs(os.path.dirname(args.out_pt) or ".", exist_ok=True)
        payload = {
            "real_vectors": torch.stack(real_vectors, dim=0),
            "black_vectors": torch.stack(black_vectors, dim=0),
            "max_samples": total_count,
            "model_path": args.model_path,
            "split": args.split,
        }
        torch.save(payload, args.out_pt)
        logger.log(f"Saved full vectors to: {args.out_pt}")

    if total_count == 0:
        logger.log("No samples were processed. Check dataset/model arguments.")
        return

    mean_cos = sum(cos_values) / len(cos_values)
    mean_l2 = sum(l2_values) / len(l2_values)

    logger.log("=" * 60)
    logger.log(f"Processed samples: {total_count}")
    logger.log(f"Output JSONL: {args.out_jsonl}")
    logger.log(f"Average cosine similarity (real vs black): {mean_cos:.6f}")
    logger.log(f"Average L2 distance (real vs black): {mean_l2:.6f}")
    logger.log("Lower cosine / higher L2 means image content has stronger effect.")
    logger.log("=" * 60)


if __name__ == "__main__":
    main()
