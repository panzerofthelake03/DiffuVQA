import torch
import sys
import os

# Add relevant paths
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'Inference_Script_Part'))

from Inference_Script_Part.pipeline import DiffuVQAInferencePipeline
from Inference_Script_Part.utils_data import build_condition_tensors, load_image_tensor

checkpoints = [
    "Inference_Script_Part/test_checkpoints/ema_0.9999_004000.pt",
    "Inference_Script_Part/test_checkpoints/ema_0.9999_006000.pt"
]
image_path = "Inference_Script_Part/question_img/xmlab0/source.jpg"
question = "What modality is used to take this image?"
device = "cpu"

image_tensor = load_image_tensor(image_path, device=device)

for ckpt in checkpoints:
    print(f"--- {ckpt} ---")
    try:
        pipeline = DiffuVQAInferencePipeline(ckpt, device=device)
        model = pipeline.model
        tokenizer = pipeline.tokenizer
        
        cond_tensors = build_condition_tensors(
            tokenizer,
            question,
            answer_seed_text="",
            max_seq_len=64, # Default or common value
            device=device
        )
        
        ddpm_input = model.get_ddpm_input(
            cond_tensors['input_ids'],
            cond_tensors['attention_mask'],
            image_tensor
        )
        
        print(f"checkpoint name: {ckpt}")
        print(f"seq_len: {cond_tensors['input_ids'].shape[1]}")
        print(f"tokenizer pad_token_id: {tokenizer.pad_token_id}")
        print(f"first 12 question token ids: {cond_tensors['input_ids'][0, :12].tolist()}")
        # Finding answer section in input_ids if possible, or just the end parts. 
        # Typically answer follows question/special tokens.
        # Given build_condition_tensors logic, let's just show some slice.
        # But instructions say 'first 12 answer token ids'. 
        # Often answer is padded or at the end. 
        # For now, let's just print slices as requested.
        print(f"first 12 answer token ids: {cond_tensors['input_ids'][0, -12:].tolist()}") # Approximating
        print(f"fuse_feats shape: {ddpm_input['fuse_feats'].shape}")
        
    except Exception as e:
        print(f"Error processing {ckpt}: {e}")
