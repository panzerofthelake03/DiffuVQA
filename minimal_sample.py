#!/usr/bin/env python3
"""
Sequential sampling script - processes samples one by one
"""
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
import json
import gc
import argparse
from tqdm import tqdm

# Copy the original sampling logic but with memory management
def main():
    # Very aggressive memory settings
    torch.cuda.empty_cache()
    
    # Run sampling with extremely small parameters
    cmd_parts = [
        r".\venv\Scripts\python.exe",
        r".\sample_vqa_GPU.py",
        "--model_path", "diffuvqa/config/ema_0.9999_000500.pt",
        "--batch_size", "1",
        "--split", "test", 
        "--dataset", "slake",
        "--data_dir", "datasets",
        "--image_dir", "datasets/slake/imgs",
        "--seed2", "105",
        "--diffusion_steps", "20",  # Very small number of steps
        "--timestep_respacing", "20"
    ]
    
    print("Starting with minimal diffusion steps...")
    cmd = " ".join(cmd_parts)
    print(f"Command: {cmd}")
    
    # Set memory optimization environment variables
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    
    # Execute
    os.system(cmd)

if __name__ == "__main__":
    main()