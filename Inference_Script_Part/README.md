# Inference Script Part (Phase 2)

This folder implements all Phase 2 roadmap items for DiffuVQA:

- Task 2.1 Model loading and initialization
- Task 2.2 Input preprocessing pipeline
- Task 2.3 Core inference logic
- Task 2.4 Result post-processing and logging

## What is implemented

1. Reusable checkpoint loader
- Loads defaults from `diffuvqa/config.json`
- Auto-merges `training_args.json` from the checkpoint directory
- Supports multiple checkpoint formats (`state_dict`, `model_state_dict`, raw tensor dict)
- Handles DataParallel keys (`module.` prefix)
- Supports strict and non-strict state loading

2. Preprocessing pipeline
- Image resize + normalization that matches training
- Question tokenization with training-compatible behavior
- Question/answer tensors and masks (`input_q_id`, `input_a_id`, `input_ids`, `input_mask`)

3. Core single-sample inference
- Single image + single prompt execution
- DDPM or DDIM decode path (step-aware)
- Diffusion rounding with `denoised_fn_round`
- Runtime-safe response object even on failures

4. Post-processing and logging
- Decoded, cleaned text output
- Structured response dataclass (`InferenceResult`) for API reuse
- Optional JSONL logging with latency, confidence, decode metadata

## Files

- `config_utils.py`: runtime config merge for any trained checkpoint
- `preprocess.py`: image/text preprocessing
- `pipeline.py`: reusable inference pipeline class
- `run_inference.py`: CLI entry point
- `__init__.py`: package exports

## Usage

From repository root:

```powershell
python Inference_Script_Part/run_inference.py `
  --checkpoint path/to/ema_0.9999_5000.pt `
  --image path/to/image.png `
  --question "What abnormality is visible?" `
  --output-jsonl logs/inference.jsonl
```

Optional flags:

- `--device cuda|cpu`
- `--steps 50` (uses DDIM when lower than training diffusion steps)
- `--top-p 0.9`
- `--clamp-step 0`
- `--clip-denoised`
- `--strict-load`

## API-style usage

```python
from Inference_Script_Part import DiffuVQAInferencePipeline

pipeline = DiffuVQAInferencePipeline(
    checkpoint_path="path/to/model.pt",
    device="cuda",
).load()

result = pipeline.predict(
    image_path="path/to/image.png",
    question="What is shown in this image?",
)

print(result.answer)
```

`result` is a structured object suitable for FastAPI/Streamlit integration in later phases.
