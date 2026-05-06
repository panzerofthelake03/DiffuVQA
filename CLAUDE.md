# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DiffuVQA is a medical Visual Question Answering (Med-VQA) system that reframes VQA as a **conditional text generation task using diffusion models**, rather than classification. It conditions the reverse diffusion process on both medical images and questions to generate open-ended answers.

## Setup & Installation

```bash
pip install -r requirements.txt
```

Datasets (SLAKE, Kvasir-VQA, Med-VQA-2019) are downloaded from HuggingFace and placed under `datasets/`.

## Directory Structure

```
DiffuVQA/
├── train.py                  # Training entry point
├── sample_vqa_GPU.py         # Inference entry point
├── diffuvqa/                 # Core model package
│   ├── vqa_model.py          #   TransformerNetModel, CVAE, Pooler
│   ├── gaussian_diffusion.py #   Diffusion forward/reverse process
│   ├── vqa_datasets.py       #   Dataset loader (ImageTextDataset)
│   ├── rounding.py           #   Nearest-neighbour token decoding
│   ├── step_sample.py        #   Timestep samplers (uniform, lossaware)
│   ├── attention/            #   Multi-head & cross-attention
│   ├── language_encoders/    #   BERT-family text encoders
│   ├── vision_encoders/      #   CLIP & Swin image encoders
│   └── utils/                #   losses, nn helpers, dist, logger
├── shared/                   # Shared utilities (used by all branches)
│   ├── basic_utils.py        #   Model factory, tokenizer manager
│   ├── train_util.py         #   TrainLoop (EMA, checkpointing)
│   └── excel_export_module.py#   Metrics → Excel export
├── eval/                     # Evaluation scripts
│   ├── eval_DiffuVQA.py      #   BLEU/ROUGE/METEOR/BERTScore/CIDEr
│   ├── enhanced_eval_metrics.py # Semantic sim, NER, TF-IDF
│   ├── compare_samples.py    #   Compare output files
│   └── prepare_eval.py       #   Pre-process outputs for eval
├── scripts/                  # Runner & utility scripts
│   ├── train.sh / run_train.py
│   ├── run_decode.py / run_decode.sh
│   └── convert_*, clean_*, diagnose_*, scan_*
├── datasets/                 # train.jsonl, valid.jsonl, test.jsonl
├── checkpoints/              # Saved .pt models + training_args.json
├── outputs/                  # Generated sample .jsonl files
├── notebooks/                # Colab notebooks
└── docs/                     # Project docs, argument explanations, images
```

## Common Commands

**Training (single GPU):**
```bash
python train.py --dataset slake --data_dir datasets/slake --lr 0.00001 --batch_size 64 --learning_steps 150000 --seed 105 --noise_schedule sqrt --hidden_dim 768 --vocab pubmedbert --seq_len 64
```

**Training (multi-GPU via script):**
```bash
bash scripts/train.sh
```

**Training via wrapper script:**
```bash
cd scripts && python run_train.py --diff_steps 2000 --lr 0.00001 --learning_steps 150000 --dataset slake --data_dir /path/to/datasets
```

**Inference/Sampling:**
```bash
python sample_vqa_GPU.py --model_path checkpoints/ema_0.9999_150000.pt --split test --batch_size 64 --step 2000 --seed 105 --out_dir outputs
```

**Evaluation (BLEU, ROUGE, METEOR, BERTScore, CIDEr):**
```bash
python eval/eval_DiffuVQA.py --folder outputs --mbr
```

**Google Colab:** Use `notebooks/run_diffuvqa_colab.ipynb` for cloud-based training/inference.

## Architecture

### Model Pipeline

1. **Image Encoder** (`diffuvqa/vision_encoders/clip_model.py`) — CLIP (ViT-B/32 or RN50) extracts visual features from medical images at resolution 384.
2. **Question Encoder** (`diffuvqa/language_encoders/bert_model.py`) — BERT/PubMedBERT/BioBERT/RoBERTa encodes questions; controlled via `--vocab` parameter.
3. **Fusion** (`diffuvqa/attention/attention_model.py`) — Cross-attention between image and question features produces a conditioning vector.
4. **Diffusion** (`diffuvqa/gaussian_diffusion.py`) — TransformerNetModel (`diffuvqa/vqa_model.py`) denoises text embeddings conditioned on the fused representation. Uses Conditional Information Gaussian Noising (CIGN) to inject conditioning into the forward process.
5. **Decoding** (`diffuvqa/rounding.py`) — Rounds denoised embeddings to vocabulary tokens via nearest-neighbor lookup.

### Key Data Flow

- Input: JSONL files (`datasets/train.jsonl`, `valid.jsonl`, `test.jsonl`) with `image`, `question`, `answer` fields.
- Loaded by `diffuvqa/vqa_datasets.py` → `ImageTextDataset` / `load_data_vqa()`.
- Answers are tokenized and embedded; noise is added during training and removed during inference.

### Shared Utilities (`shared/`)

- `shared/basic_utils.py` — `myTokenizer` handles BERT/PubMedBERT/BioBERT/RoBERTa vocab selection; `create_model_and_diffusion()` is the main factory used by both `train.py` and `sample_vqa_GPU.py`.
- `shared/train_util.py` — `TrainLoop` class manages the full training loop: EMA weight tracking (`ema_rate=0.9999`), checkpoint saving, validation, and logging.
- `shared/excel_export_module.py` — `DiffuVQAExcelExporter` logs training/sampling metrics to Excel.

### Diffusion Configuration

Default config in `diffuvqa/config.json`:
- `diff_steps: 2500`, `noise_schedule: sqrt`
- `schedule_sampler: lossaware` (importance sampling of timesteps via `diffuvqa/step_sample.py`)
- `hidden_dim: 768`, `seq_len: 32`
- `lr: 1e-5`, `batch_size: 20`, `learning_steps: 500000`

### Supported Vocab Options

`--vocab` accepts: `bert`, `pubmedbert`, `biobert`, `roberta` — controls both the tokenizer and the language encoder weights loaded in `shared/basic_utils.py`.

## Evaluation Metrics

`eval/eval_DiffuVQA.py` computes: BLEU-1/2/3/4, ROUGE-L, METEOR, BERTScore, CIDEr.  
`eval/enhanced_eval_metrics.py` adds: semantic similarity (BioSentenceTransformer), NER overlap, TF-IDF similarity.

Pass `--mbr` to `eval/eval_DiffuVQA.py` to use Minimum Bayes Risk decoding (picks the best among multiple generated samples).
