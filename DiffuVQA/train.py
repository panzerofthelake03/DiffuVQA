"""
Train a diffusion model on images.
"""
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import argparse
import torch.nn as nn
import json, torch, os
import numpy as np
from diffuvqa.utils import dist_util, logger
from diffuvqa.vqa_datasets import load_data_vqa
from diffuvqa.step_sample import create_named_schedule_sampler
from basic_utils import (
    load_defaults_config,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
    load_model_emb,
    load_tokenizer
)
from train_util import TrainLoop
from transformers import set_seed
import wandb

import sys
import os
from torchvision import transforms

current_dir = os.path.dirname(os.path.abspath(__file__))

sys.path.append(current_dir)

### custom your wandb setting here ###
# os.environ["WANDB_API_KEY"] = ""
os.environ["WANDB_MODE"] = "offline"


def create_argparser():
    defaults = dict()
    defaults.update(load_defaults_config())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults) # update latest args according to argparse
    return parser

def main():
    args = create_argparser().parse_args()
    set_seed(args.seed) 
    # dist_util.setup_dist()
    logger.configure()
    logger.log("### Creating data loader...")

    tokenizer = load_tokenizer(args)
    model_weight, tokenizer = load_model_emb(args, tokenizer)
    transform = transforms.Compose([
        transforms.Resize((args.image_resolution, args.image_resolution)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    data = load_data_vqa(batch_size=args.batch_size, seq_len=args.seq_len, args=args, model_emb=model_weight,
                         transform=transform, split="train", loaded_vocab=tokenizer)
    if args.valid:
        data_valid = load_data_vqa(batch_size=args.batch_size, seq_len=args.seq_len, args=args, model_emb=model_weight,
                               transform=transform, split='valid', loaded_vocab=tokenizer)
    else:
        data_valid = None


    print('#'*30, 'size of vocab', args.vocab_size)

    logger.log("### Creating model and diffusion...")
    print("use{}".format(args.model))
    # print('#'*30, 'CUDA_VISIBLE_DEVICES', os.environ['CUDA_VISIBLE_DEVICES'])
    model, diffusion = create_model_and_diffusion(args=args)
    # print('#'*30, 'cuda', dist_util.dev())

    if torch.cuda.device_count() > 1:
        print(f"Let's use {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)

    model.to(dist_util.dev()) #  DEBUG **
    # model.cuda() #  DEBUG **

    pytorch_total_params = sum(p.numel() for p in model.parameters())

    logger.log(f'### The parameter count is {pytorch_total_params}')
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log(f'### Saving the hyperparameters to {args.checkpoint_path}/training_args.json')
    with open(f'{args.checkpoint_path}/training_args.json', 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    if ('LOCAL_RANK' not in os.environ) or (int(os.environ['LOCAL_RANK']) == 0):
        wandb.init(
            project=os.getenv("WANDB_PROJECT", "DiffuVQA"),
            name=args.checkpoint_path,
        )
        wandb.config.update(args.__dict__, allow_val_change=True)
 
    logger.log("### Training...")

    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=0,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        learning_steps=args.learning_steps,
        checkpoint_path=args.checkpoint_path,
        gradient_clipping=args.gradient_clipping,
        eval_data=data_valid,
        eval_interval=args.eval_interval
    ).run_loop(args)

if __name__ == "__main__":
    main()
