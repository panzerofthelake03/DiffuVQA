import argparse
import json
import os
from pathlib import Path

import torch as th
from torchvision.transforms import transforms

from diffuvqa.rounding import denoised_fn_round
from diffuvqa.vqa_datasets import load_data_vqa
from shared.basic_utils import create_model_and_diffusion, load_defaults_config, load_tokenizer
from functools import partial


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return betas


def parse_args():
    p = argparse.ArgumentParser(description="Smoke test for sample_vqa_GPU masking behavior")
    p.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    p.add_argument("--split", default="test")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--step", type=int, default=30, help="Sampling steps")
    p.add_argument("--seed", type=int, default=105)
    p.add_argument("--device", default="cuda" if th.cuda.is_available() else "cpu")
    p.add_argument("--atol", type=float, default=1e-6)
    p.add_argument(
        "--mask-mode",
        choices=["legacy", "patched"],
        default="legacy",
        help="legacy reproduces sample_vqa_GPU old mask; patched uses fuse(0)+answer(1)",
    )
    return p.parse_args()


def build_runtime_args(checkpoint_path: Path):
    defaults = load_defaults_config()
    cfg_path = checkpoint_path.parent / "training_args.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing training args: {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as f:
        train_cfg = json.load(f)

    defaults.update(train_cfg)
    defaults["model_path"] = str(checkpoint_path)
    defaults["batch_size"] = int(defaults.get("batch_size", 2))

    class Args:
        pass

    args = Args()
    for k, v in defaults.items():
        setattr(args, k, v)
    return args


def first_batch(data_loader, device):
    for image, cond in data_loader:
        cond["input_q_id"] = cond["input_q_id"].to(device)
        cond["input_ids"] = cond["input_ids"].to(device)
        return image.to(device), cond
    raise RuntimeError("Data loader produced no batches")


def main():
    cli = parse_args()
    device = th.device(cli.device)
    checkpoint = Path(cli.checkpoint).resolve()

    args = build_runtime_args(checkpoint)
    args.model_path = str(checkpoint)
    args.batch_size = cli.batch_size
    args.step = cli.step
    args.seed2 = cli.seed
    args.split = cli.split
    if not hasattr(args, "use_noising_f"):
        args.use_noising_f = False

    th.manual_seed(cli.seed)
    if th.cuda.is_available():
        th.cuda.manual_seed_all(cli.seed)

    model, diffusion = create_model_and_diffusion(args=args)
    state_dict = th.load(args.model_path, map_location=device)
    state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device).eval().requires_grad_(False)

    tokenizer = load_tokenizer(args)

    emb_weight = model.word_embedding.weight.clone().detach()
    emb_dim = emb_weight.size(1)
    if emb_dim == int(args.hidden_dim):
        model_emb = th.nn.Embedding(
            num_embeddings=emb_weight.size(0),
            embedding_dim=emb_dim,
            _weight=emb_weight.to(device),
        ).eval().requires_grad_(False)
    else:
        model_emb = model.word_embedding.eval().requires_grad_(False)
        model_emb = model_emb.to(device)

    transform = transforms.Compose([
        transforms.Resize((int(args.image_resolution), int(args.image_resolution))),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    data_test = load_data_vqa(
        batch_size=args.batch_size,
        seq_len=int(args.seq_len),
        args=args,
        model_emb=model_emb.cpu(),
        transform=transform,
        split=args.split,
        loaded_vocab=tokenizer,
        loop=False,
    )

    image, cond = first_batch(data_test, device)
    input_ids_x = cond.pop("input_ids").to(device)
    input_ids_a = cond.pop("input_a_id").to(device)
    input_emb = model.get_embeds(input_ids_a)

    input_ids_mask = cond.pop("input_mask").to(device)
    input_ids_mask_ori = input_ids_mask.to("cpu")

    fuse_feats, _ = model.get_ddpm_input(image, cond)
    x_start = th.cat([fuse_feats, input_emb], dim=1)

    fuse_len = fuse_feats.size(1)
    bsz = input_ids_mask.size(0)
    answer_len = input_emb.size(1)

    if cli.mask_mode == "legacy":
        # Reproduce sample_vqa_GPU.py legacy mask logic exactly.
        fuse_mask = th.zeros((bsz, fuse_len), dtype=input_ids_mask.dtype, device=input_ids_mask.device)
        full_mask = th.cat([fuse_mask, input_ids_mask], dim=1)
    else:
        # Validate patched behavior: x_start = [fuse | answer].
        full_mask = th.cat([
            th.zeros((bsz, fuse_len), dtype=input_ids_mask.dtype, device=input_ids_mask.device),
            th.ones((bsz, answer_len), dtype=input_ids_mask.dtype, device=input_ids_mask.device),
        ], dim=1)

    total_len = x_start.size(1)
    raw_mask_len = full_mask.size(1)
    if raw_mask_len < total_len:
        pad_len = total_len - raw_mask_len
        full_mask = th.cat(
            [full_mask, th.zeros((bsz, pad_len), dtype=full_mask.dtype, device=full_mask.device)],
            dim=1,
        )
    elif raw_mask_len > total_len:
        full_mask = full_mask[:, :total_len]

    input_ids_mask_broadcast = th.broadcast_to(full_mask.unsqueeze(dim=-1), x_start.shape).to(device)

    noise = th.randn_like(x_start)
    x_noised = th.where(input_ids_mask_broadcast == 0, x_start, noise)

    if int(args.step) == int(args.diffusion_steps):
        use_ddim = False
        step_gap = 1
    else:
        use_ddim = True
        step_gap = max(int(args.diffusion_steps) // int(args.step), 1)

    sample_fn = diffusion.ddim_sample_loop if use_ddim else diffusion.p_sample_loop
    sample_shape = (x_start.shape[0], int(args.seq_len), int(args.hidden_dim))

    samples = sample_fn(
        model,
        sample_shape,
        noise=x_noised,
        clip_denoised=bool(getattr(args, "clip_denoised", False)),
        denoised_fn=partial(denoised_fn_round, args, model_emb),
        model_kwargs={},
        top_p=float(getattr(args, "top_p", 0.0)),
        clamp_step=int(getattr(args, "clamp_step", 0)),
        clamp_first=True,
        mask=input_ids_mask_broadcast,
        x_start=x_start,
        gap=step_gap,
    )

    final_sample = samples[-1]

    zeros = int((full_mask == 0).sum().item())
    ones = int((full_mask == 1).sum().item())
    unique_vals = sorted([int(v) for v in th.unique(full_mask).detach().cpu().tolist()])

    max_diff_xnoised = float((x_noised - x_start).abs().max().item())
    max_diff_final = float((final_sample - x_start).abs().max().item())
    noising_applied = bool(max_diff_xnoised > cli.atol)
    diffusion_changed = bool(max_diff_final > cli.atol)

    a_shape = final_sample.size(1) // 2
    decoded_hidden = final_sample[:, a_shape:, :]
    logits = model.get_logits(decoded_hidden)
    cands = th.topk(logits, k=1, dim=-1).indices

    generated = tokenizer.decode_token(cands[0].to("cpu"))
    source = tokenizer.decode_token(input_ids_x[0, : int(args.seq_len)].to("cpu"))
    reference = tokenizer.decode_token(input_ids_x[0, int(args.seq_len):].to("cpu"))

    expected_pass = ((not noising_applied) and (not diffusion_changed) and ones == 0)
    if cli.mask_mode == "patched":
        expected_pass = (noising_applied and diffusion_changed and ones > 0)

    report = {
        "checkpoint": str(checkpoint),
        "mask_mode": cli.mask_mode,
        "device": str(device),
        "diffusion_steps": int(args.diffusion_steps),
        "sampling_step": int(args.step),
        "step_gap": int(step_gap),
        "use_ddim": bool(use_ddim),
        "shape_x_start": list(x_start.shape),
        "shape_mask_raw": [int(bsz), int(raw_mask_len)],
        "shape_mask_used": list(full_mask.shape),
        "mask_unique_values": unique_vals,
        "mask_zero_count": zeros,
        "mask_one_count": ones,
        "noising_applied": noising_applied,
        "diffusion_changed_output": diffusion_changed,
        "max_abs_diff_xnoised_vs_xstart": max_diff_xnoised,
        "max_abs_diff_final_vs_xstart": max_diff_final,
        "question": source,
        "reference_answer": reference,
        "generated_answer": generated,
        "mask_behavior": "all_zero_after_truncate" if ones == 0 else "contains_generation_region",
        "smoke_test_pass": bool(expected_pass),
    }

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
