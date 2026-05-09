"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import torch
import argparse
import os, json
from tracemalloc import start

import numpy as np
import torch as th
import torch.distributed as dist
from torchvision.transforms import transforms
from transformers import set_seed
from diffuvqa.rounding import denoised_fn_round, get_efficient_knn
from diffuvqa.vqa_datasets import load_data_vqa
from shared.excel_export_module import record_sampling_data
from datetime import datetime

# from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

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

def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


def create_argparser():
    defaults = dict(model_path='', step=2500, out_dir='', top_p=0)
    decode_defaults = dict(split='test', clamp_step=0, seed2=105, clip_denoised=False, num_samples=1)
    defaults.update(load_defaults_config())
    defaults.update(decode_defaults)
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


@th.no_grad()
def main():

    args = create_argparser().parse_args()

    logger.configure()
    logger.log(f"DEBUG: Parsed model_path = {args.model_path}")

    # Backwards-compatibility: some older training runs don't record newer flags.
    if not hasattr(args, 'use_noising_f'):
        args.use_noising_f = False

    logger.log("### Loading model from %s" % args.model_path)
    # load configurations.
    config_path = os.path.join(os.path.split(args.model_path)[0], "training_args.json")
    logger.log(f"config_path: {config_path}")
    with open(config_path, 'rb', ) as f:
        training_args = json.load(f)

    original_model_path = args.model_path
    original_batch_size = args.batch_size
    original_Seed = args.seed
    original_diffusion_step = args.step

    training_args['batch_size'] = args.batch_size

    # Capture CLI family flags before training_args overwrites them
    cli_vocab = args.vocab if hasattr(args, 'vocab') and args.vocab else None
    cli_model = args.model if hasattr(args, 'model') and args.model else None

    args.__dict__.update(training_args)

    # Fail-fast: reject CLI flags that contradict the checkpoint's training family
    _FAMILY_KEYS = ('vocab', 'model', 'use_plm_init')
    for key in _FAMILY_KEYS:
        cli_val = {'vocab': cli_vocab, 'model': cli_model}.get(key)
        ckpt_val = training_args.get(key)
        if cli_val and ckpt_val and cli_val != ckpt_val:
            raise ValueError(
                f"[sampling] CLI --{key}={cli_val!r} conflicts with checkpoint "
                f"training_args {key}={ckpt_val!r}. Use the checkpoint's value or omit the flag."
            )

    if(original_model_path != ""):
        args.model_path = original_model_path
    logger.log(f"### Updated args: {args}")

    if(original_Seed is not None):
        args.seed = original_Seed

    if(original_batch_size is not None):
        args.batch_size = original_batch_size

    logger.log(f">>> diffusion_steps before: {original_diffusion_step}")
    if(original_diffusion_step is not None):
        args.step = original_diffusion_step

    num_steps = args.diffusion_steps

    betas =  betas_for_alpha_bar(num_steps, lambda t: 1 - np.sqrt(t + 0.0001),)
    alphas = 1 - betas  # α = 1 - β
    alphas = torch.from_numpy(alphas)
    # 计算所有步骤的alpha累乘结果
    alphas_prod = torch.cumprod(alphas, 0)  # 如果 alphas 是 [a1, a2, a3]，那么 alphas_prod 将会是 [a1, a1*a2, a1*a2*a3]
    # 计算前一步的alpha累乘结果，用于后续计算，初始值设为1
    alphas_prod_p = torch.cat([torch.tensor([1]).float(), alphas_prod[:-1]],
                              0)  # 如果 alphas_prod 是 [a1, a1*a2, a1*a2*a3]，那么 alphas_prod_p 将会是 [1, a1, a1*a2]
    # 计算alphas_prod的平方根，用于后续计算
    alphas_bar_sqrt = torch.sqrt(alphas_prod)
    # 计算1减去alphas_prod的自然对数，用于后续计算
    one_minus_alphas_bar_log = torch.log(1 - alphas_prod)
    # 计算1减去alphas_prod的平方根，用于后续计算
    one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)

    # 断言所有计算出的张量形状相同，确保数据一致性
    assert alphas.shape == alphas_prod.shape == alphas_prod_p.shape == \
           alphas_bar_sqrt.shape == one_minus_alphas_bar_log.shape == \
           one_minus_alphas_bar_sqrt.shape

    logger.log("### Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(args=args)
    
    # Determine device
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
        # Use the model's embedding module directly to avoid mismatched shape assertions.
        model_emb = model.word_embedding.eval().requires_grad_(False)
        try:
            model_emb.to(device)
        except Exception:
            pass

    set_seed(args.seed2)

    logger.log(f"### Sampling...on {args.split}")

    transform = transforms.Compose([
        transforms.Resize((args.image_resolution, args.image_resolution)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    ## load data
    # small info
    logger.log(f"batch_size={args.batch_size}")
    data_test = load_data_vqa(batch_size=args.batch_size, seq_len=args.seq_len, args=args, model_emb=model_emb.cpu(),
                               transform=transform, split=args.split, loaded_vocab=tokenizer, loop=False)

    start_t = time.time()

    # Build a short, stable samples directory to avoid very long nested paths on Windows.
    # Use a single `samples` folder under the provided out_dir and a concise filename.
    model_base_name = os.path.basename(os.path.split(args.model_path)[0]) + f'.{os.path.split(args.model_path)[1]}'
    # Use a short repo-root samples directory to avoid Windows MAX_PATH issues.
    if(args.out_dir):
        samples_dir = args.out_dir
        os.makedirs(samples_dir, exist_ok=True)
    else:
        samples_dir = os.path.abspath(os.path.join(os.getcwd(), "samples"))
        os.makedirs(samples_dir, exist_ok=True)

    # Create a compact filename: <model_folder>.<checkpoint>.seed<seed>_step<clamp>.jsonl
    print("<<<<model_base_name:", model_base_name)
    print("<<<<args.model_path:", args.model_path)
    checkpoint_name = os.path.basename(args.model_path)  # Get the actual model file name
    print("<<<<<checkpoint_name:", checkpoint_name)
    out_filename = f"{checkpoint_name}.seed{args.seed}_step{args.clamp_step}_samplestep{args.step}_bsize{args.batch_size}.jsonl"
    # Truncate filename if it's excessively long to avoid Windows path length issues.
    if len(out_filename) > 200:
        out_filename = out_filename[:200]

    out_path = os.path.join(samples_dir, out_filename)
    # Ensure parent directory exists (defensive) before opening the file.
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    print("out_path:", out_path)
    print("batch_size:", args.batch_size)
    all_text_data = []
    all_image_data = []

    try:
        for image, cond in data_test:
            cond['input_q_id'] = cond['input_q_id'].to(device)
            cond['input_ids'] = cond['input_ids'].to(device)
            all_text_data.append(cond)
            all_image_data.append(image.to(device))
    except StopIteration:
        print('### End of reading iteration...')

    model_emb.to(device)

    text_iterator = iter(all_text_data)
    image_iterator = iter(all_image_data)

    # Add progress bar for sampling
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

        # masks and metadata
        input_ids_mask = cond.pop('input_mask').to(device)
        input_ids_mask_ori = input_ids_mask.to(th.device("cpu"))
        image_name = cond.pop('image_name')

        # x_start: image fusion features + pure noise for answer positions.
        # We intentionally do NOT use answer embeddings here so the model
        # must generate the answer from scratch (no data leakage).
        fuse_feats, _ = model.get_ddpm_input(image, cond)
        fuse_len = fuse_feats.size(1)
        bsz = fuse_feats.size(0)

        answer_len = input_ids_a.size(1)
        ans_noise = th.randn(bsz, answer_len, args.hidden_dim, device=device)
        x_start = torch.cat([fuse_feats, ans_noise], dim=1)

        # Mask: 0 = frozen (image fusion tokens), 1 = diffused (answer tokens)
        fuse_mask = th.zeros((bsz, fuse_len), dtype=th.int64, device=device)
        ans_mask  = th.ones((bsz, answer_len), dtype=th.int64, device=device)
        full_mask  = th.cat([fuse_mask, ans_mask], dim=1)   # [B, fuse_len + answer_len]

        input_ids_mask = th.broadcast_to(full_mask.unsqueeze(dim=-1), x_start.shape).to(device)

        # Start from pure noise everywhere; frozen positions will be restored each step
        x_noised = th.randn_like(x_start)

        f = x_start  # auxiliary conditioning tensor (same shape as x_start)

        model_kwargs = {}

        if args.step == args.diffusion_steps:
            args.use_ddim = False
            step_gap = 1
        else:
            args.use_ddim = True
            step_gap = args.diffusion_steps // args.step

        sample_fn = (diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop)

        # sample_shape is used only to extract batch size in p_sample_loop.
        # The actual sequence shape is determined by x_noised (fuse_len + seq_len).
        sample_shape = x_start.shape

        # The diffusion sampling procedure prints progress using carriage returns
        # which causes the terminal to repeatedly overwrite the same lines on
        # some shells (PowerShell). Suppress stdout/stderr for the duration of
        # the sampling to keep the terminal clean.
        # Seçenek 4: MBR — num_samples > 1 ise birden fazla sample üret, en tutarlısını seç
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
                    denoised_fn=partial(denoised_fn_round, args, model_emb),
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
            # MBR: batch içindeki her pozisyon için candidate'ler arasında
            # ortalama L2 mesafesi en düşük olanı seç (en merkezi sample)
            stacked = th.stack(all_sample_candidates, dim=0)  # [N, B, seq, dim]
            mean_rep = stacked.mean(dim=0, keepdim=True)       # [1, B, seq, dim]
            dists = ((stacked - mean_rep) ** 2).sum(dim=-1).mean(dim=-1)  # [N, B]
            best_idx = dists.argmin(dim=0)  # [B]
            sample = th.stack([all_sample_candidates[best_idx[b]][b] for b in range(bsz)], dim=0)

        sample = sample[:, fuse_len:fuse_len + answer_len, :]
        logits = model.get_logits(sample)
        cands = th.topk(logits, k=1, dim=-1)

        probs = th.softmax(logits, dim=-1)
        chosen_probs = probs.gather(-1, cands.indices).squeeze(-1)
        seq_confidence = chosen_probs.mean(dim=1)
        seq_logprob = th.log(chosen_probs.clamp(min=1e-12)).sum(dim=1)

        try:
            sample_flat = sample.contiguous().view(-1, sample.size(-1))
            val, idx_nn = get_efficient_knn(model_emb.weight.to(sample.device), sample_flat)
            val = val.view(sample.size(0), sample.size(1))
            avg_nn_dist = (-val).mean(dim=1)
        except Exception:
            avg_nn_dist = th.zeros(sample.size(0), device=sample.device)

        word_lst_recover = []
        word_lst_ref = []
        word_lst_source = []

        sep_id = tokenizer.tokenizer.sep_token_id if hasattr(tokenizer, 'tokenizer') else tokenizer.sep_token_id
        pad_id = tokenizer.tokenizer.pad_token_id if hasattr(tokenizer, 'tokenizer') else tokenizer.pad_token_id

        for seq, prob_seq in zip(cands.indices, chosen_probs):
            seq = seq.to(th.device("cpu"))
            prob_seq = prob_seq.to(th.device("cpu"))

            # Seçenek 1: [SEP] veya [PAD] tokenına kadar kes
            token_list = seq.tolist()
            stop_ids = set(filter(None, [sep_id, pad_id]))
            cut = next((i for i, t in enumerate(token_list) if t in stop_ids), len(token_list))
            seq = seq[:cut] if cut > 0 else seq

            # Seçenek 3: confidence < threshold olan tokenları kes (trailing noise)
            conf_threshold = 0.3
            if cut == len(token_list):
                # SEP/PAD bulunamadıysa trailing low-confidence tokenları sil
                prob_list = prob_seq[:len(seq)].tolist()
                last_confident = len(prob_list)
                for j in range(len(prob_list) - 1, -1, -1):
                    if prob_list[j] >= conf_threshold:
                        last_confident = j + 1
                        break
                seq = seq[:last_confident] if last_confident > 0 else seq[:1]

            tokens = tokenizer.decode_token(seq)
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
        
        # Update progress bar after each batch
        pbar.update(1)
        # break
        #
        # for (recov, ref, src) in zip(word_lst_recover, word_lst_ref, word_lst_source):
        #     print(json.dumps(
        #         {"question": src, "reference_answer": ref, "generate_answer": recov}),
        #           file=fout)
        # fout.close()

    # Close progress bar
    pbar.close()

    # After sampling is completed, calculate total sample size and record parameters

    total_samples = len(all_text_data)  # Calculate the total number of samples
    sampling_duration = time.time() - start_t  # Calculate the total sampling duration

    # Define the sampling parameters to record
    sampling_parameters = {
        "model_path": args.model_path,
        "batch_size": args.batch_size,
        "top_p": args.top_p,
        "seed": args.seed,
        "sampling_steps": args.step,
        "total_samples": total_samples,
        "sampling_duration_in_seconds": sampling_duration  # Record the duration
    }

    # Record the sampling data in the Excel file
    record_sampling_data(sampling_parameters, output_dir="reports")

    print(f"### Recorded sampling data: {sampling_parameters}")

    print('### Total takes {:.2f}s .....'.format(time.time() - start_t))
    print(f'### Written the decoded output to {out_path}')


if __name__ == "__main__":
    main()
