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