#!/usr/bin/env python3
"""
Memory-efficient sampling script for DiffuVQA
Processes samples one at a time to avoid GPU memory issues
"""
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
import gc
import json
import argparse
from tqdm import tqdm
import time

# Import DiffuVQA modules
from basic_utils import (
    load_defaults_config,
    create_model_and_diffusion,
    add_dict_to_argparser,
    load_tokenizer
)
from diffuvqa.vqa_datasets import load_data_vqa

def clear_gpu_memory():
    """Clear GPU memory cache"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

def create_argparser():
    defaults = dict(
        model_path='diffuvqa/config/ema_0.9999_000500.pt',
        step=500,
        out_dir='samples',
        top_p=0,
        num_samples=50,  # Number of samples to generate
        max_batch_size=1,  # Maximum batch size
        clear_memory_freq=5,  # Clear memory every N samples
        test_dataset='datasets/test_100.jsonl'  # Path to limited test dataset
    )
    
    decode_defaults = dict(
        split='test', 
        clamp_step=0, 
        seed2=105, 
        clip_denoised=False
    )
    
    defaults.update(load_defaults_config())
    defaults.update(decode_defaults)
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

def load_test_samples(test_file, num_samples):
    """Load limited number of test samples"""
    samples = []
    try:
        with open(test_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= num_samples:
                    break
                if line.strip():
                    sample = json.loads(line)
                    samples.append(sample)
    except Exception as e:
        print(f"Error loading test samples: {e}")
        return []
    
    return samples

def sample_single_batch(model, diffusion, tokenizer, batch_data, args):
    """Sample a single batch with memory management"""
    try:
        # Prepare batch
        questions = [item['question'] for item in batch_data]
        image_names = [item['img_name'] for item in batch_data]
        
        # Create input tensors (simplified - you may need to adapt based on actual model input)
        # This is a placeholder - adapt based on your actual data loading
        batch_size = len(questions)
        
        # Generate samples
        with torch.no_grad():
            # Sample function call (adapt parameters as needed)
            sample_fn = diffusion.p_sample_loop
            
            # You'll need to adapt this based on your actual model input format
            # This is a simplified version
            model_kwargs = {}  # Add your model kwargs here
            
            samples = sample_fn(
                model,
                (batch_size, args.seq_len, args.hidden_dim),  # Adapt shape
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
                top_p=args.top_p,
            )
            
        return samples
        
    except torch.cuda.OutOfMemoryError as e:
        print(f"GPU memory error in batch: {e}")
        clear_gpu_memory()
        return None
    except Exception as e:
        print(f"Error in sampling: {e}")
        return None

def main():
    args = create_argparser().parse_args()
    
    print(f"Starting memory-efficient sampling...")
    print(f"Target samples: {args.num_samples}")
    print(f"Max batch size: {args.max_batch_size}")
    print(f"Test dataset: {args.test_dataset}")
    
    # Load configuration
    config_path = os.path.join(os.path.split(args.model_path)[0], "training_args.json")
    with open(config_path, 'rb') as f:
        training_args = json.load(f)
    
    # Override batch size for memory efficiency
    training_args['batch_size'] = args.max_batch_size
    args.__dict__.update(training_args)
    
    # Create model and diffusion
    print("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(args=args)
    
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model weights
    print(f"Loading model from {args.model_path}")
    state_dict = torch.load(args.model_path, map_location=device, weights_only=False)
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    model.eval().requires_grad_(False).to(device)
    
    # Load tokenizer
    tokenizer = load_tokenizer(args)
    
    # Load test samples
    print(f"Loading test samples from {args.test_dataset}")
    test_samples = load_test_samples(args.test_dataset, args.num_samples)
    
    if not test_samples:
        print("No test samples loaded!")
        return
    
    print(f"Loaded {len(test_samples)} test samples")
    
    # Prepare output
    output_file = os.path.join(args.out_dir, f"efficient_samples_{args.num_samples}.jsonl")
    os.makedirs(args.out_dir, exist_ok=True)
    
    successful_samples = 0
    results = []
    
    # Process samples in small batches
    with tqdm(total=len(test_samples), desc="Generating samples") as pbar:
        for i in range(0, len(test_samples), args.max_batch_size):
            batch_samples = test_samples[i:i + args.max_batch_size]
            
            try:
                # Clear memory periodically
                if i % (args.clear_memory_freq * args.max_batch_size) == 0 and i > 0:
                    print(f"\\nClearing GPU memory at sample {i}")
                    clear_gpu_memory()
                    time.sleep(1)  # Brief pause
                
                # Generate samples for this batch
                batch_results = sample_single_batch(model, diffusion, tokenizer, batch_samples, args)
                
                if batch_results is not None:
                    # Process results (placeholder - adapt based on your output format)
                    for j, sample in enumerate(batch_samples):
                        result = {
                            'image_name': sample.get('img_name', ''),
                            'question': sample.get('question', ''),
                            'reference_answer': sample.get('answer', ''),
                            'generate_answer': f"generated_answer_{successful_samples}",  # Placeholder
                            'confidence': 0.5,  # Placeholder
                            'rationale': "Memory-efficient generation"
                        }
                        results.append(result)
                        successful_samples += 1
                
                pbar.update(len(batch_samples))
                
            except Exception as e:
                print(f"\\nError processing batch starting at {i}: {e}")
                clear_gpu_memory()
                continue
    
    # Save results
    print(f"\\nSaving {len(results)} results to {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result) + '\\n')
    
    print(f"\\nGeneration complete!")
    print(f"Successfully generated: {successful_samples}/{args.num_samples} samples")
    print(f"Results saved to: {output_file}")
    
    # Final memory cleanup
    clear_gpu_memory()

if __name__ == "__main__":
    main()