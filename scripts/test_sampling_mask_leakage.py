import argparse
import json
import os
from types import SimpleNamespace

import torch as th
from torchvision.transforms import transforms
from functools import partial

from diffuvqa.rounding import denoised_fn_round
from diffuvqa.vqa_datasets import load_data_vqa
from shared.basic_utils import load_tokenizer, create_model_and_diffusion


def _load_args_from_training_json(training_args_path: str) -> SimpleNamespace:
    with open(training_args_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return SimpleNamespace(**cfg)


def _build_runtime_args(base_args: SimpleNamespace, data_root: str, split: str, batch_size: int) -> SimpleNamespace:
    # SLAKE jsonl entries use image_name like "imgs/xmlab102/source.jpg".
    # Therefore image_dir must be <data_root>/imgs so join() resolves to <data_root>/imgs/imgs/...
    base_args.data_dir = data_root
    base_args.image_dir = os.path.join(data_root, "imgs")
    base_args.split = split
    base_args.batch_size = batch_size

    # Keep defaults if not present.
    if not hasattr(base_args, "seq_len"):
        base_args.seq_len = 128
    if not hasattr(base_args, "image_resolution"):
        base_args.image_resolution = 384
    if not hasattr(base_args, "hidden_dim"):
        base_args.hidden_dim = 768
    if not hasattr(base_args, "vocab_size"):
        base_args.vocab_size = 30522
    if not hasattr(base_args, "vocab"):
        base_args.vocab = "bert"
    if not hasattr(base_args, "model"):
        base_args.model = "transformer-bert"
    if not hasattr(base_args, "dataset"):
        base_args.dataset = "SLAKE"

    return base_args


def _normalize_text(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


def _build_sampling_mask(input_mask: th.Tensor, x_start: th.Tensor, fuse_len: int, answer_len: int, mode: str) -> th.Tensor:
    bsz = input_mask.size(0)
    if mode == "legacy":
        fuse_mask = th.zeros((bsz, fuse_len), dtype=input_mask.dtype, device=input_mask.device)
        full_mask = th.cat([fuse_mask, input_mask], dim=1)
    elif mode == "fixed":
        full_mask = th.cat(
            [
                th.zeros((bsz, fuse_len), dtype=input_mask.dtype, device=input_mask.device),
                th.ones((bsz, answer_len), dtype=input_mask.dtype, device=input_mask.device),
            ],
            dim=1,
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")

    total_len = x_start.size(1)
    cur_len = full_mask.size(1)
    if cur_len < total_len:
        pad_len = total_len - cur_len
        pad_tensor = th.zeros((bsz, pad_len), dtype=full_mask.dtype, device=full_mask.device)
        full_mask = th.cat([full_mask, pad_tensor], dim=1)
    elif cur_len > total_len:
        full_mask = full_mask[:, :total_len]

    return th.broadcast_to(full_mask.unsqueeze(dim=-1), x_start.shape)


def _run_tiny_e2e(
    run_args: SimpleNamespace,
    data_loader,
    checkpoint_path: str,
    device: th.device,
    max_samples: int,
    sample_steps: int,
    top_p: float,
    clamp_step: int,
    clip_denoised: bool,
    tiny_mode: str,
):
    run_args.model_path = checkpoint_path
    model, diffusion = create_model_and_diffusion(args=run_args)

    state_dict = th.load(checkpoint_path, map_location=device)
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    model.eval().requires_grad_(False).to(device)

    emb_weight = model.word_embedding.weight.clone().detach()
    emb_dim = emb_weight.size(1)
    if emb_dim == run_args.hidden_dim:
        model_emb = th.nn.Embedding(
            num_embeddings=emb_weight.size(0),
            embedding_dim=emb_dim,
            _weight=emb_weight.to(device),
        ).eval().requires_grad_(False)
    else:
        model_emb = model.word_embedding.eval().requires_grad_(False)
        model_emb.to(device)

    tokenizer = load_tokenizer(run_args)

    enabled_modes = ["legacy", "fixed"] if tiny_mode == "compare" else [tiny_mode]
    stats = {
        "legacy": {"n": 0, "em": 0, "empty": 0},
        "fixed": {"n": 0, "em": 0, "empty": 0},
    }
    examples = []

    if sample_steps == run_args.diffusion_steps:
        sample_fn = diffusion.p_sample_loop
        step_gap = 1
    else:
        sample_fn = diffusion.ddim_sample_loop
        step_gap = max(1, run_args.diffusion_steps // sample_steps)

    with th.no_grad():
        for image, cond in data_loader:
            done = True
            for mode in enabled_modes:
                if stats[mode]["n"] < max_samples:
                    done = False
                    break
            if done:
                break

            image = image.to(device)
            input_ids_x = cond["input_ids"].to(device)
            input_ids_a = cond["input_a_id"].to(device)
            input_mask = cond["input_mask"].to(device)

            cond_for_model = {}
            for key, value in cond.items():
                if isinstance(value, th.Tensor):
                    cond_for_model[key] = value.to(device)
                else:
                    cond_for_model[key] = value

            fuse_feats, _ = model.get_ddpm_input(image, cond_for_model)
            input_emb = model.get_embeds(input_ids_a)
            x_start = th.cat([fuse_feats, input_emb], dim=1)

            fuse_len = fuse_feats.size(1)
            answer_len = input_emb.size(1)
            sample_shape = tuple(x_start.shape)

            refs = []
            for seq in input_ids_x:
                refs.append(_normalize_text(tokenizer.decode_token(seq[run_args.seq_len:].to(th.device("cpu")))))

            legacy_preds = None

            for mode in enabled_modes:
                input_ids_mask = _build_sampling_mask(input_mask, x_start, fuse_len, answer_len, mode).to(device)
                noise = th.randn_like(x_start)
                x_noised = th.where(input_ids_mask == 0, x_start, noise)

                samples = sample_fn(
                    model,
                    sample_shape,
                    noise=x_noised,
                    clip_denoised=clip_denoised,
                    denoised_fn=partial(denoised_fn_round, run_args, model_emb),
                    model_kwargs={},
                    top_p=top_p,
                    clamp_step=clamp_step,
                    clamp_first=True,
                    mask=input_ids_mask,
                    x_start=x_start,
                    gap=step_gap,
                )

                sample = samples[-1]
                a_shape = sample.size(1) // 2
                sample = sample[:, a_shape:, :]
                logits = model.get_logits(sample)
                cands = th.topk(logits, k=1, dim=-1).indices

                preds = []
                for seq in cands:
                    preds.append(_normalize_text(tokenizer.decode_token(seq.to(th.device("cpu")))))

                for i, (pred, ref) in enumerate(zip(preds, refs)):
                    if stats[mode]["n"] >= max_samples:
                        break
                    stats[mode]["n"] += 1
                    if pred == "":
                        stats[mode]["empty"] += 1
                    if pred == ref and ref != "":
                        stats[mode]["em"] += 1

                if mode == "legacy":
                    legacy_preds = preds
                elif mode == "fixed" and legacy_preds is not None:
                    for i, (pred_fixed, ref) in enumerate(zip(preds, refs)):
                        if len(examples) >= min(8, max_samples):
                            break
                        examples.append(
                            {
                                "reference": ref,
                                "legacy_pred": legacy_preds[i],
                                "fixed_pred": pred_fixed,
                            }
                        )

            done = True
            for mode in enabled_modes:
                if stats[mode]["n"] < max_samples:
                    done = False
                    break
            if done:
                break

    def finalize(one):
        n = max(1, one["n"])
        return {
            "samples": one["n"],
            "exact_match": round(one["em"] / n, 6),
            "empty_rate": round(one["empty"] / n, 6),
        }

    report = {
        "mode": tiny_mode,
    }
    if "legacy" in enabled_modes:
        report["legacy"] = finalize(stats["legacy"])
    if "fixed" in enabled_modes:
        report["fixed"] = finalize(stats["fixed"])
    if tiny_mode == "compare":
        report["examples"] = examples
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect answer leakage caused by legacy sampling mask construction.")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint .pt (used to locate default training_args.json)")
    parser.add_argument("--training-args", default="", help="Optional explicit training_args.json path")
    parser.add_argument("--data-root", required=True, help="SLAKE root folder containing train.jsonl/test.jsonl/valid.jsonl and imgs/")
    parser.add_argument("--split", default="test", choices=["train", "valid", "test"])
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-batches", type=int, default=3)
    parser.add_argument("--tiny-e2e", action="store_true", help="Run tiny end-to-end sampling comparison (legacy vs fixed mask)")
    parser.add_argument("--tiny-max-samples", type=int, default=6)
    parser.add_argument("--tiny-steps", type=int, default=10)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--top-p", type=float, default=0.0)
    parser.add_argument("--clamp-step", type=int, default=0)
    parser.add_argument("--clip-denoised", action="store_true")
    parser.add_argument("--tiny-mode", default="compare", choices=["compare", "legacy", "fixed"], help="Run both legacy+fixed or only one mode")
    parser.add_argument("--tiny-force", action="store_true", help="Disable CPU safety caps for tiny-e2e")
    args = parser.parse_args()

    training_args_path = args.training_args or os.path.join(os.path.dirname(args.checkpoint), "training_args.json")
    if not os.path.exists(training_args_path):
        raise FileNotFoundError(f"training_args.json not found: {training_args_path}")

    base_args = _load_args_from_training_json(training_args_path)
    run_args = _build_runtime_args(base_args, args.data_root, args.split, args.batch_size)

    tokenizer = load_tokenizer(run_args)

    # Data loader requires model_emb; for this mask test, random embeddings are sufficient.
    model_emb = th.nn.Embedding(num_embeddings=run_args.vocab_size, embedding_dim=run_args.hidden_dim)

    transform = transforms.Compose([
        transforms.Resize((run_args.image_resolution, run_args.image_resolution)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    data_loader = load_data_vqa(
        batch_size=run_args.batch_size,
        seq_len=run_args.seq_len,
        args=run_args,
        model_emb=model_emb.cpu(),
        transform=transform,
        split=run_args.split,
        loaded_vocab=tokenizer,
        loop=False,
    )

    checked_batches = 0
    total_answer_non_pad = 0
    legacy_generated_answer_tokens = 0
    fixed_generated_answer_tokens = 0

    for _, cond in data_loader:
        input_mask = cond["input_mask"]
        input_a_id = cond["input_a_id"]

        answer_len = input_a_id.size(1)

        # Legacy behavior (buggy branch): after concat with fuse mask and truncation to [fuse+answer],
        # the answer segment effectively takes input_mask[:, :answer_len] (question mask => mostly zeros).
        legacy_answer_mask = input_mask[:, :answer_len]

        # Fixed behavior: always generate/noise full answer segment.
        fixed_answer_mask = th.ones_like(legacy_answer_mask)

        answer_non_pad = (input_a_id != 0)
        total_answer_non_pad += int(answer_non_pad.sum().item())

        legacy_generated_answer_tokens += int(((legacy_answer_mask == 1) & answer_non_pad).sum().item())
        fixed_generated_answer_tokens += int(((fixed_answer_mask == 1) & answer_non_pad).sum().item())

        checked_batches += 1
        if checked_batches >= args.num_batches:
            break

    if total_answer_non_pad == 0:
        raise RuntimeError("No non-pad answer tokens found in inspected batches; cannot evaluate leakage.")

    legacy_generation_ratio = legacy_generated_answer_tokens / total_answer_non_pad
    fixed_generation_ratio = fixed_generated_answer_tokens / total_answer_non_pad

    # If legacy_generation_ratio is very low, answer tokens are not being generated/noised -> leakage risk.
    leakage_detected = legacy_generation_ratio < 0.5

    report = {
        "checkpoint": os.path.abspath(args.checkpoint),
        "training_args": os.path.abspath(training_args_path),
        "data_root": os.path.abspath(args.data_root),
        "split": args.split,
        "checked_batches": checked_batches,
        "answer_non_pad_tokens": total_answer_non_pad,
        "legacy_generation_ratio": round(legacy_generation_ratio, 6),
        "fixed_generation_ratio": round(fixed_generation_ratio, 6),
        "leakage_detected": leakage_detected,
        "verdict": "FAIL (legacy mask leaks/conditions answer tokens)" if leakage_detected else "PASS",
    }

    if args.tiny_e2e:
        if args.device == "auto":
            device = th.device("cuda" if th.cuda.is_available() else "cpu")
        else:
            device = th.device(args.device)

        # CPU safety caps: avoid long hangs when compare mode is used with large settings.
        if device.type == "cpu" and not args.tiny_force:
            mode_mult = 2 if args.tiny_mode == "compare" else 1
            work_units = args.tiny_max_samples * args.tiny_steps * mode_mult
            max_work_units = 200
            if work_units > max_work_units:
                scale = max_work_units / float(work_units)
                new_steps = max(5, int(args.tiny_steps * scale))
                if new_steps == args.tiny_steps:
                    new_samples = max(2, int(args.tiny_max_samples * scale))
                    args.tiny_max_samples = new_samples
                else:
                    args.tiny_steps = new_steps

        tiny_report = _run_tiny_e2e(
            run_args=run_args,
            data_loader=data_loader,
            checkpoint_path=args.checkpoint,
            device=device,
            max_samples=args.tiny_max_samples,
            sample_steps=args.tiny_steps,
            top_p=args.top_p,
            clamp_step=args.clamp_step,
            clip_denoised=args.clip_denoised,
            tiny_mode=args.tiny_mode,
        )
        report["tiny_e2e"] = {
            "device": str(device),
            "sample_steps": args.tiny_steps,
            "max_samples": args.tiny_max_samples,
            "top_p": args.top_p,
            "clamp_step": args.clamp_step,
            "mode": args.tiny_mode,
            "results": tiny_report,
        }

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
