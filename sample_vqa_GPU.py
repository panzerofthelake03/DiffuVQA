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

# from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

import time
import io
import sys
import contextlib
from diffuvqa.utils import dist_util, logger
from functools import partial
from basic_utils import (
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
    decode_defaults = dict(split='test', clamp_step=0, seed2=105, clip_denoised=False)
    defaults.update(load_defaults_config())
    defaults.update(decode_defaults)
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


@th.no_grad()
def main():

    args = create_argparser().parse_args()

    # Backwards-compatibility: some older training runs don't record newer flags.
    if not hasattr(args, 'use_noising_f'):
        args.use_noising_f = False

    logger.configure()

    # load configurations.
    config_path = os.path.join(os.path.split(args.model_path)[0], "training_args.json")
    print(config_path)
    with open(config_path, 'rb', ) as f:
        training_args = json.load(f)
    training_args['batch_size'] = args.batch_size
    args.__dict__.update(training_args)

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
    state_dict = torch.load(args.model_path, map_location="cuda")
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    logger.log(f'### The parameter count is {pytorch_total_params}')

    model.eval().requires_grad_(False).to(th.device("cuda"))

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
            model_emb.to(th.device("cuda"))
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
    samples_dir = os.path.abspath(os.path.join(os.getcwd(), "samples"))
    os.makedirs(samples_dir, exist_ok=True)

    # Create a compact filename: <model_folder>.<checkpoint>.seed<seed>_step<clamp>.jsonl
    checkpoint_name = os.path.split(args.model_path)[1]
    out_filename = f"{os.path.basename(os.path.split(args.model_path)[0])}.{checkpoint_name}.seed{args.seed}_step{args.clamp_step}.jsonl"
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
            cond['input_q_id'] = cond['input_q_id'].to(th.device("cuda"))
            cond['input_ids'] = cond['input_ids'].to(th.device("cuda"))
            all_text_data.append(cond)
            all_image_data.append(image.to(th.device("cuda")))
    except StopIteration:
        print('### End of reading iteration...')

    model_emb.to(th.device("cuda"))

    text_iterator = iter(all_text_data)
    image_iterator = iter(all_image_data)

    for image, cond in zip(image_iterator, text_iterator):

        if not cond:
            continue

        input_ids_x = cond.pop('input_ids').to(th.device("cuda"))
        input_ids_a = cond.pop('input_a_id').to(th.device("cuda"))
        input_emb = model.get_embeds(input_ids_a)

        # masks and metadata
        input_ids_mask = cond.pop('input_mask').to(th.device("cuda"))
        input_ids_mask_ori = input_ids_mask.to(th.device("cpu"))
        image_name = cond.pop('image_name')

        # x_start mean prep
        fuse_feats, _ = model.get_ddpm_input(image, cond)
        f = torch.cat([fuse_feats, fuse_feats], dim=1)
    # debug: fuse_feats.shape suppressed to avoid noisy stdout
        x_start = torch.cat([fuse_feats, input_emb], dim=1)

        # Build a full mask that covers the image-fuse tokens (zeros) + text tokens (input_ids_mask)
        fuse_len = fuse_feats.size(1)
        bsz = input_ids_mask.size(0)
        fuse_mask = th.zeros((bsz, fuse_len), dtype=input_ids_mask.dtype, device=input_ids_mask.device)
        full_mask = th.cat([fuse_mask, input_ids_mask], dim=1)

        # Ensure full_mask length matches x_start sequence length; pad or truncate as needed
        total_len = x_start.size(1)
        cur_len = full_mask.size(1)
        if cur_len < total_len:
            pad_len = total_len - cur_len
            pad_tensor = th.zeros((bsz, pad_len), dtype=full_mask.dtype, device=full_mask.device)
            full_mask = th.cat([full_mask, pad_tensor], dim=1)
        elif cur_len > total_len:
            full_mask = full_mask[:, :total_len]

        input_ids_mask = th.broadcast_to(full_mask.unsqueeze(dim=-1), x_start.shape).to(th.device("cuda"))

        noise = th.randn_like(x_start)
        if args.use_noising_f:
            print("noising f")
            noise = alphas_bar_sqrt[num_steps - 1] * f + one_minus_alphas_bar_sqrt[num_steps - 1] * noise

        x_noised = th.where(input_ids_mask == 0, x_start, noise)

        model_kwargs = {}

        if args.step == args.diffusion_steps:
            args.use_ddim = False
            step_gap = 1
        else:
            args.use_ddim = True
            step_gap = args.diffusion_steps // args.step

        sample_fn = (diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop)

        sample_shape = (x_start.shape[0], args.seq_len, args.hidden_dim)

        # The diffusion sampling procedure prints progress using carriage returns
        # which causes the terminal to repeatedly overwrite the same lines on
        # some shells (PowerShell). Suppress stdout/stderr for the duration of
        # the sampling to keep the terminal clean.
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            samples = sample_fn(
                model,
                sample_shape,
                noise=x_noised,
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

        sample = samples[-1]
        a_shape = sample.size(1) // 2
        sample = sample[:, a_shape:, :]
    # sample shape suppressed
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

    # debug tensor printing removed to avoid huge outputs that clutter/overwrite
    # the terminal during long runs.
        for seq, input_mask in zip(cands.indices, input_ids_mask_ori):
            seq = seq.to(th.device("cpu"))
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
        # break
        #
        # for (recov, ref, src) in zip(word_lst_recover, word_lst_ref, word_lst_source):
        #     print(json.dumps(
        #         {"question": src, "reference_answer": ref, "generate_answer": recov}),
        #           file=fout)
        # fout.close()

    print('### Total takes {:.2f}s .....'.format(time.time() - start_t))
    print(f'### Written the decoded output to {out_path}')


if __name__ == "__main__":
    main()
