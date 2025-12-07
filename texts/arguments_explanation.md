# Arguments Explanation for `sample_vqa_GPU.py`
Example:
.\venv\Scripts\python.exe sample_vqa_GPU.py --model_path "diffuvqa/config/ema_0.9999_001000.pt" --batch_size 1 --top_p -1 --out_dir "samples" --seed 123 --step 5

This document provides a detailed explanation of the arguments used in the `sample_vqa_GPU.py` script.

## Arguments

### 1. `--model_path`
- **Description:** Specifies the path to the pre-trained model checkpoint file.
- **Details:**
  - This file contains the weights of the diffusion model used for generating answers to visual questions.
  - Example: `diffuvqa/config/ema_0.9999_000500.pt`
  - The file name typically includes:
    - `ema`: Indicates Exponential Moving Average of weights.
    - `0.9999`: The EMA decay rate.
    - `000500`: The number of training steps.

### 2. `--batch_size`
- **Description:** Specifies the number of samples to process in a single batch.
- **Details:**
  - A batch size of `1` processes one image-question pair at a time.
  - Larger batch sizes improve throughput but require more GPU memory.
  - Example: `--batch_size 1`

### 3. `--top_p`
- **Description:** Specifies the nucleus sampling parameter for controlling the diversity of generated answers.
- **Details:**
  - Nucleus Sampling considers only the top `p` cumulative probability mass of tokens during generation.
  - Setting `--top_p -1` disables nucleus sampling, using the default sampling strategy.
  - Typical values: `0.8` to `1.0` for diverse but coherent outputs.

### 4. `--out_dir`
- **Description:** Specifies the directory where the generated samples will be saved.
- **Details:**
  - The script creates a file in this directory to store the generated answers.
  - Example: `--out_dir "samples"`
  - The output file name is dynamically constructed based on the model name, seed, and step (e.g., `ema_0.9999_000500.pt.seed123_step50.jsonl`).

### 5. `--seed`
- **Description:** Sets the random seed for reproducibility.
- **Details:**
  - Ensures that the same inputs produce the same outputs across runs.
  - Affects random processes such as noise generation in the diffusion model.
  - Example: `--seed 123`

### 6. `--step`
- **Description:** Specifies the number of diffusion steps to use during sampling.
- **Details:**
  - Diffusion models generate samples by iteratively refining noise over a series of steps.
  - A smaller number of steps (e.g., `50`) results in faster sampling but may reduce output quality.
  - A larger number of steps (e.g., `1000`) produces higher-quality outputs but takes longer to compute.
  - Example: `--step 50`

## Example Command
```bash
.\venv\Scripts\python.exe sample_vqa_GPU.py \
    --model_path "diffuvqa/config/ema_0.9999_000500.pt" \
    --batch_size 1 \
    --top_p -1 \
    --out_dir "samples" \
    --seed 123 \
    --step 50
```

## Output
- The generated samples will be saved in the specified `--out_dir` directory.
- Example output file: `samples/ema_0.9999_000500.pt.seed123_step50.jsonl`

This file contains the generated answers, along with metadata such as confidence scores and rationales.

---

# Arguments Explanation for `train.py` and `eval_DiffuVQA.py`

## `train.py`

The `train.py` script is used to train the DiffuVQA model on a specified dataset. Below is a detailed explanation of the arguments:

### Arguments

#### 1. `--lr`
- **Description:** Specifies the learning rate for training.
- **Details:**
  - A smaller learning rate (e.g., `0.0001`) ensures stable training but may take longer to converge.
  - Example: `--lr 0.0001`

#### 2. `--batch_size`
- **Description:** Specifies the number of samples to process in a single batch.
- **Details:**
  - Larger batch sizes improve throughput but require more GPU memory.
  - Example: `--batch_size 4`

#### 3. `--learning_steps`
- **Description:** Specifies the total number of training steps.
- **Details:**
  - Determines how long the model will be trained.
  - Example: `--learning_steps 200`

#### 4. `--save_interval`
- **Description:** Specifies how often (in steps) to save the model checkpoint.
- **Details:**
  - Example: `--save_interval 100`

#### 5. `--log_interval`
- **Description:** Specifies how often (in steps) to log training progress.
- **Details:**
  - Example: `--log_interval 25`

#### 6. `--data_dir`
- **Description:** Specifies the directory containing the dataset.
- **Details:**
  - Example: `--data_dir datasets`

#### 7. `--image_dir`
- **Description:** Specifies the directory containing the images for the dataset.
- **Details:**
  - Example: `--image_dir datasets/slake/imgs`

#### 8. `--dataset`
- **Description:** Specifies the name of the dataset to use for training.
- **Details:**
  - Example: `--dataset slake`

### Example Command
```bash
.\venv\Scripts\python.exe .   
```
python .\train.py --lr 0.0001 --batch_size 4 --learning_steps 500 --save_interval 100 --data_dir datasets --image_dir datasets/slake/imgs/imgs
---

## `eval_DiffuVQA.py`

The `eval_DiffuVQA.py` script is used to evaluate the DiffuVQA model on generated samples. Below is a detailed explanation of the arguments:

### Arguments

#### 1. `--folder`
- **Description:** Specifies the folder containing the decoded text files to evaluate.
- **Details:**
  - Example: `--folder samples`

#### 2. `--filename`
- **Description:** Specifies the path to a single decoded text file to evaluate.
- **Details:**
  - If provided, only this file will be evaluated.
  - Example: `--file ema_0.9999_001000.pt.seed123_step0_samplestep10_bsize1.jsonl`

#### 3. `--mbr`
- **Description:** Enables Minimum Bayes Risk (MBR) decoding.
- **Details:**
  - MBR decoding selects the most probable output based on a risk function.
  - Example: `--mbr`

#### 4. `--sos`
- **Description:** Specifies the start-of-sequence token.
- **Details:**
  - Default: `[CLS]`
  - Example: `--sos [CLS]`

#### 5. `--eos`
- **Description:** Specifies the end-of-sequence token.
- **Details:**
  - Default: `[SEP]`
  - Example: `--eos [SEP]`

#### 6. `--sep`
- **Description:** Specifies the separator token.
- **Details:**
  - Default: `[SEP]`
  - Example: `--sep [SEP]`

#### 7. `--pad`
- **Description:** Specifies the padding token.
- **Details:**
  - Default: `[PAD]`
  - Example: `--pad [PAD]`

### Example Command
```bash
.\venv\Scripts\python.exe eval_DiffuVQA.py \
    --folder samples \
    --file ema_0.9999_001000.pt.seed123_step0_samplestep10_bsize1.jsonl
```