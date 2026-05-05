import torch
import sys
import os

sys.path.append(os.getcwd())

from Inference_Script_Part.pipeline import DiffuVQAInferencePipeline
from Inference_Script_Part.preprocess import build_condition_tensors, load_image_tensor

checkpoints = [
    "Inference_Script_Part/test_checkpoints/ema_0.9999_004000.pt",
    "Inference_Script_Part/test_checkpoints/ema_0.9999_006000.pt"
]
image_path = "Inference_Script_Part/question_img/xmlab0/source.jpg"
question = "What modality is used to take this image?"
device = "cpu"

try:
    for ckpt in checkpoints:
        if not os.path.exists(ckpt):
            print(f"Checkpoint not found: {ckpt}")
            continue
            
        pipeline = DiffuVQAInferencePipeline(ckpt, device=device).load()
        model = pipeline.model
        tokenizer = pipeline.tokenizer
        
        # Determine image_resolution from model config if possible, or use default 224
        image_resolution = getattr(model, "image_resolution", 224)
        image_tensor = load_image_tensor(image_path, image_resolution, device=device)
        
        cond_tensors = build_condition_tensors(
            tokenizer,
            question,
            answer_seed_text="",
            max_seq_len=64,
            device=device
        )
        
        input_ids = cond_tensors["input_ids"]
        attention_mask = cond_tensors["attention_mask"]
        
        ddpm_input = model.get_ddpm_input(
            input_ids,
            attention_mask,
            image_tensor
        )
        
        print(f"checkpoint name: {ckpt}")
        print(f"seq_len: {input_ids.shape[1]}")
        print(f"tokenizer pad_token_id: {tokenizer.pad_token_id}")
        
        q_tokens = tokenizer.encode(question, add_special_tokens=True)
        print(f"first 12 question token ids: {input_ids[0, :12].tolist()}")
        print(f"first 12 answer token ids: {input_ids[0, len(q_tokens):len(q_tokens)+12].tolist()}")
        print(f"fuse_feats shape: {list(ddpm_input['fuse_feats'].shape)}")
except Exception as e:
    print(f"Error: {e}")
