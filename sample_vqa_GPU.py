import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import torch
import argparse
import os, json

import numpy as np
import torch as th
import torch.distributed as dist
from torchvision.transforms import transforms
from transformers import set_seed
from diffuvqa.rounding import denoised_fn_round, get_efficient_knn
from diffuvqa.vqa_datasets import load_data_vqa
from shared.excel_export_module import record_sampling_data
from datetime import datetime

import time
import io
import sys
import contextlib
from diffuvqa.utils import dist_util, logger
from functools import partial
from shared.basic_utils import (
    load_defaults_config,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
    load_tokenizer
)

torch.multiprocessing.set_sharing_strategy('file_system')


def create_argparser():
    defaults = dict(model_path='', step=2500, out_dir='', top_p=0)
    decode_defaults = dict(split='test', clamp_step=0, seed2=105, clip_denoised=False, num_samples=1,
                           decode_top_k=5, min_answer_tokens=2, short_answer_penalty=1.0)
    defaults.update(load_defaults_config())
    defaults.update(decode_defaults)
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


@th.no_grad()
def main():

    args = create_argparser().parse_args()

    logger.configure()

    if not hasattr(args, 'use_noising_f'):
        args.use_noising_f = False

    logger.log("### Loading model from %s" % args.model_path)
    config_path = os.path.join(os.path.split(args.model_path)[0], "training_args.json")
    with open(config_path, 'rb', ) as f:
        training_args = json.load(f)

    original_model_path = args.model_path
    original_batch_size = args.batch_size
    original_Seed = args.seed
    original_diffusion_step = args.step

    training_args['batch_size'] = args.batch_size

    cli_vocab = args.vocab if hasattr(args, 'vocab') and args.vocab else None
    cli_model = args.model if hasattr(args, 'model') and args.model else None

    args.__dict__.update(training_args)

    _FAMILY_KEYS = ('vocab', 'model', 'use_plm_init')
    for key in _FAMILY_KEYS:
        cli_val = {'vocab': cli_vocab, 'model': cli_model}.get(key)
        ckpt_val = training_args.get(key)
        if cli_val and ckpt_val and cli_val != ckpt_val:
            raise ValueError(
                f"[sampling] CLI --{key}={cli_val!r} conflicts with checkpoint "
                f"training_args {key}={ckpt_val!r}. Use the checkpoint's value or omit the flag."
            )

    if original_model_path != "":
        args.model_path = original_model_path

    if original_Seed is not None:
        args.seed = original_Seed

    if original_batch_size is not None:
        args.batch_size = original_batch_size

    if original_diffusion_step is not None:
        args.step = original_diffusion_step

    logger.log("### Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(args=args)

    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    logger.log(f"### Using device: {device}")
    
    state_dict = torch.load(args.model_path, map_location=device, weights_only=False)
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    logger.log(f'### The parameter count is {pytorch_total_params}')

    model.eval().requires_grad_(False).to(device)

    tokenizer = load_tokenizer(args)
    # Create a model embedding object for nearest-neighbor / rounding.
    # If the pretrained word embedding dim matches args.hidden_dim we can clone it,
    # otherwise keep the original embedding module (it will be moved to CUDA).
    emb_weight = model.word_embedding.weight.clone().detach()
    emb_dim = emb_weight.size(1)
    if emb_dim == args.hidden_dim:
        model_emb = th.nn.Embedding(num_embeddings=emb_weight.size(0), embedding_dim=emb_dim, _weight=emb_weight.cuda()).eval().requires_grad_(False)
    else:
        model_emb = model.word_embedding.eval().requires_grad_(False)
        try:
            model_emb.to(device)
        except Exception:
            pass

    set_seed(args.seed2)

    # Build a boolean mask [vocab_size] that is True for every ## continuation token.
    # Used in (1) denoised_fn_round to prevent trajectory lock-in on subword tokens,
    # and (2) final logit masking to eliminate ## tokens from discrete output.
    _raw_tok = tokenizer.tokenizer if hasattr(tokenizer, 'tokenizer') else tokenizer
    _vocab = _raw_tok.get_vocab()
    subword_mask = torch.zeros(len(_vocab), dtype=torch.bool)
    for token, idx in _vocab.items():
        if token.startswith('##'):
            subword_mask[idx] = True
    subword_mask = subword_mask.to(device)
    logger.log(f"### Subword mask: {subword_mask.sum().item()} ## tokens excluded from decoding")

    logger.log(f"### Sampling...on {args.split}")

    transform = transforms.Compose([
        transforms.Resize((args.image_resolution, args.image_resolution)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    logger.log(f"batch_size={args.batch_size}")
    data_test = load_data_vqa(batch_size=args.batch_size, seq_len=args.seq_len, args=args, model_emb=model_emb.cpu(),
                               transform=transform, split=args.split, loaded_vocab=tokenizer, loop=False)

    start_t = time.time()

    model_base_name = os.path.basename(os.path.split(args.model_path)[0]) + f'.{os.path.split(args.model_path)[1]}'
    if args.out_dir:
        samples_dir = args.out_dir
        os.makedirs(samples_dir, exist_ok=True)
    else:
        samples_dir = os.path.abspath(os.path.join(os.getcwd(), "samples"))
        os.makedirs(samples_dir, exist_ok=True)

    checkpoint_name = os.path.basename(args.model_path)
    out_filename = f"{checkpoint_name}.seed{args.seed}_step{args.clamp_step}_samplestep{args.step}_bsize{args.batch_size}.jsonl"
    if len(out_filename) > 200:
        out_filename = out_filename[:200]

    out_path = os.path.join(samples_dir, out_filename)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    logger.log(f"out_path: {out_path}")

    all_text_data = []
    all_image_data = []

    try:
        for image, cond in data_test:
            cond['input_q_id'] = cond['input_q_id'].to(device)
            cond['input_ids'] = cond['input_ids'].to(device)
            all_text_data.append(cond)
            all_image_data.append(image.to(device))
    except StopIteration:
        pass

    model_emb.to(device)

    text_iterator = iter(all_text_data)
    image_iterator = iter(all_image_data)

    from tqdm import tqdm
    total_batches = len(all_text_data)
    pbar = tqdm(total=total_batches, desc="Sampling", unit="batch",
               bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

    for image, cond in zip(image_iterator, text_iterator):

        if not cond:
            pbar.update(1)
            continue

        input_ids_x = cond.pop('input_ids').to(device)
        input_ids_a = cond.pop('input_a_id').to(device)

        input_ids_mask = cond.pop('input_mask').to(device)
        input_ids_mask_ori = input_ids_mask.to(th.device("cpu"))
        image_name = cond.pop('image_name')

        fuse_feats, _ = model.get_ddpm_input(image, cond)
        fuse_len = fuse_feats.size(1)
        bsz = fuse_feats.size(0)

        answer_len = input_ids_a.size(1)
        ans_noise = th.randn(bsz, answer_len, args.hidden_dim, device=device)
        x_start = torch.cat([fuse_feats, ans_noise], dim=1)

        # 0 = frozen (image fusion tokens), 1 = diffused (answer tokens)
        fuse_mask = th.zeros((bsz, fuse_len), dtype=th.int64, device=device)
        ans_mask  = th.ones((bsz, answer_len), dtype=th.int64, device=device)
        full_mask  = th.cat([fuse_mask, ans_mask], dim=1)
        input_ids_mask = th.broadcast_to(full_mask.unsqueeze(dim=-1), x_start.shape).to(device)

        model_kwargs = {}

        if args.step == args.diffusion_steps:
            args.use_ddim = False
            step_gap = 1
        else:
            args.use_ddim = True
            step_gap = args.diffusion_steps // args.step

        sample_fn = (diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop)
        sample_shape = x_start.shape

        num_samples = getattr(args, 'num_samples', 1)
        all_sample_candidates = []
        for _ in range(num_samples):
            _noise = th.randn_like(x_start)
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                _samples = sample_fn(
                    model,
                    sample_shape,
                    noise=_noise,
                    clip_denoised=args.clip_denoised,
                    denoised_fn=partial(denoised_fn_round, args, model_emb, subword_mask=subword_mask),
                    model_kwargs=model_kwargs,
                    top_p=args.top_p,
                    clamp_step=args.clamp_step,
                    clamp_first=True,
                    mask=input_ids_mask,
                    x_start=x_start,
                    gap=step_gap,
                )
            all_sample_candidates.append(_samples[-1])

        if num_samples == 1:
            sample = all_sample_candidates[0]
        else:
            # MBR: pick the candidate closest to the mean (most central sample)
            stacked = th.stack(all_sample_candidates, dim=0)
            mean_rep = stacked.mean(dim=0, keepdim=True)
            dists = ((stacked - mean_rep) ** 2).sum(dim=-1).mean(dim=-1)
            best_idx = dists.argmin(dim=0)
            sample = th.stack([all_sample_candidates[best_idx[b]][b] for b in range(bsz)], dim=0)

        sample = sample[:, fuse_len:fuse_len + answer_len, :]
        logits = model.get_logits(sample)

        # Mask ## continuation tokens from logits so they can never appear in output.
        logits = logits.masked_fill(subword_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        decode_top_k = max(1, getattr(args, 'decode_top_k', 5))
        cands = th.topk(logits, k=decode_top_k, dim=-1)

        probs = th.softmax(logits, dim=-1)
        chosen_probs = probs.gather(-1, cands.indices)
        seq_confidence = chosen_probs[:, :, 0].mean(dim=1)
        seq_logprob = th.log(chosen_probs[:, :, 0].clamp(min=1e-12)).sum(dim=1)

        try:
            sample_flat = sample.contiguous().view(-1, sample.size(-1))
            val, idx_nn = get_efficient_knn(model_emb.weight.to(sample.device), sample_flat)
            val = val.view(sample.size(0), sample.size(1))
            # val = -squared_L2; take sqrt to get true L2 distance
            avg_nn_dist = (-val).clamp(min=0.0).sqrt().mean(dim=1)
        except Exception:
            avg_nn_dist = th.zeros(sample.size(0), device=sample.device)

        word_lst_recover = []
        word_lst_ref = []
        word_lst_source = []

        sep_id = tokenizer.tokenizer.sep_token_id if hasattr(tokenizer, 'tokenizer') else tokenizer.sep_token_id
        pad_id = tokenizer.tokenizer.pad_token_id if hasattr(tokenizer, 'tokenizer') else tokenizer.pad_token_id

        min_answer_tokens = max(1, getattr(args, 'min_answer_tokens', 2))
        short_answer_penalty = getattr(args, 'short_answer_penalty', 1.0)
        conf_threshold = 0.1
        stop_ids = set(filter(None, [sep_id, pad_id]))

        for b_idx in range(cands.indices.size(0)):
            top_k_ids = cands.indices[b_idx].cpu()
            top_k_probs = chosen_probs[b_idx].cpu()

            best_seq = None
            best_score = float('-inf')

            for k_i in range(decode_top_k):
                seq = top_k_ids[:, k_i]
                prob_seq = top_k_probs[:, k_i]
                token_list = seq.tolist()

                cut = len(token_list)
                for i, t in enumerate(token_list):
                    if t in stop_ids and i >= min_answer_tokens:
                        cut = i
                        break

                seq = seq[:cut] if cut > 0 else seq[:1]
                prob_seq_cut = prob_seq[:len(seq)]

                if cut == len(token_list):
                    prob_list = prob_seq_cut.tolist()
                    last_confident = len(prob_list)
                    for j in range(len(prob_list) - 1, -1, -1):
                        if prob_list[j] >= conf_threshold:
                            last_confident = j + 1
                            break
                    seq = seq[:last_confident] if last_confident > 0 else seq[:1]
                    prob_seq_cut = prob_seq[:len(seq)]

                eff_len = len(seq)
                log_prob_sum = th.log(prob_seq_cut.clamp(min=1e-12)).sum().item()
                avg_log_prob = log_prob_sum / max(eff_len, 1)

                penalty = short_answer_penalty if eff_len < min_answer_tokens else 0.0
                score = avg_log_prob - penalty

                if score > best_score:
                    best_score = score
                    best_seq = seq

            tokens = tokenizer.decode_token(best_seq)
            word_lst_recover.append(tokens)

        for seq, input_mask in zip(input_ids_x, input_ids_mask_ori):
            seq = seq.to(th.device("cpu"))
            word_lst_source.append(tokenizer.decode_token(seq[:args.seq_len]))
            word_lst_ref.append(tokenizer.decode_token(seq[args.seq_len:]))

        with open(out_path, 'a', encoding='utf-8') as fout:
            for i, (recov, ref, src, image_name_i) in enumerate(zip(word_lst_recover, word_lst_ref, word_lst_source, image_name)):
                conf_val = float(seq_confidence[i].cpu().item()) if 'seq_confidence' in locals() else None
                avg_dist = float(avg_nn_dist[i].cpu().item()) if 'avg_nn_dist' in locals() else None
                rationale = f"Average token prob={conf_val:.3f}, avg_nn_l2={avg_dist:.3f}" if conf_val is not None else "n/a"
                out_obj = {"image_name": image_name_i, "question": src, "reference_answer": ref, "generate_answer": recov, "confidence": conf_val, "rationale": rationale}
                print(json.dumps(out_obj, ensure_ascii=False), file=fout)
        
        pbar.update(1)

    pbar.close()

    total_samples = len(all_text_data)
    sampling_duration = time.time() - start_t

    sampling_parameters = {
        "model_path": args.model_path,
        "batch_size": args.batch_size,
        "top_p": args.top_p,
        "seed": args.seed,
        "sampling_steps": args.step,
        "total_samples": total_samples,
        "sampling_duration_in_seconds": sampling_duration,
    }

    record_sampling_data(sampling_parameters, output_dir="reports")

    logger.log(f'### Total takes {sampling_duration:.2f}s')
    logger.log(f'### Written the decoded output to {out_path}')


if __name__ == "__main__":
    main()
