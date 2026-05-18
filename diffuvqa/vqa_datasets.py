from diffuvqa.utils import logger
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

import torch
import json
import psutil
import datasets
from datasets import Dataset as Dataset2
import sys
import os
from PIL import Image


def _resolve_image_path(args, image_name):
    """Build absolute image path for a dataset entry."""
    if not image_name:
        logger.log("Warning: empty image_name, using placeholder path")
        return ''
    if args.dataset.lower() == 'davekevin':
        return os.path.join(args.data_dir, 'daveKevin_images', image_name)
    return os.path.join(args.image_dir, image_name)


def _preview_first_samples(args, data_lst, split, max_samples=3):
    """Render and save a quick preview of the first few question/image pairs."""
    questions = data_lst.get('question', [])
    image_names = data_lst.get('image_name', [])
    answers = data_lst.get('answer', [])

    sample_count = min(max_samples, len(questions), len(image_names))
    if sample_count <= 0:
        logger.log('### Preview skipped: no samples available.')
        return

    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        logger.log(f"Warning: matplotlib unavailable, cannot preview samples: {e}")
        return

    fig, axes = plt.subplots(1, sample_count, figsize=(5 * sample_count, 5))
    if sample_count == 1:
        axes = [axes]

    for i in range(sample_count):
        img_path = _resolve_image_path(args, image_names[i])
        logger.log(f"Attempted image path: {img_path}")
        parent_dir = os.path.dirname(img_path) if img_path else ''
        logger.log(f"Parent folder of attempted image path: {parent_dir}")
        if parent_dir and os.path.isdir(parent_dir):
            try:
                parent_items = os.listdir(parent_dir)
                logger.log(f"Inside parent folder ({parent_dir}): {parent_items[:30]}")
            except Exception as e:
                logger.log(f"Could not list parent folder ({parent_dir}): {e}")
        else:
            logger.log(f"Parent folder does not exist: {parent_dir}")

        if img_path and os.path.isdir(img_path):
            try:
                image_path_items = os.listdir(img_path)
                logger.log(f"Inside image_path directory ({img_path}): {image_path_items[:30]}")
            except Exception as e:
                logger.log(f"Could not list image_path directory ({img_path}): {e}")
        
        try:
            if img_path and os.path.exists(img_path):
                logger.log(f"### Loading image for preview: {img_path}")
                img = Image.open(img_path).convert('RGB')
            else:
                logger.log("Warning: Image not found, using placeholder.")
                img = Image.new('RGB', (224, 224), (0, 0, 0))
        except Exception as e:
            logger.log(f"Error loading image: {e}")
            img = Image.new('RGB', (224, 224), (0, 0, 0))

        q_text = str(questions[i]) if i < len(questions) else ''
        a_text = str(answers[i]) if i < len(answers) else ''

        axes[i].imshow(img)
        axes[i].set_title(f"Q: {q_text[:60]}..." if len(q_text) > 60 else f"Q: {q_text}", fontsize=9, wrap=True)
        axes[i].text(
            0.5,
            -0.15,
            f"A: {a_text[:60]}..." if len(a_text) > 60 else f"A: {a_text}",
            transform=axes[i].transAxes,
            fontsize=8,
            ha='center',
            wrap=True,
        )
        axes[i].axis('off')

    plt.tight_layout()
    preview_name = f"{args.dataset}_{split}_first3_preview.png"
    preview_path = os.path.join(args.data_dir, preview_name)
    plt.savefig(preview_path, dpi=100, bbox_inches='tight')
    logger.log(f"### Saved sample preview to {preview_path}")

    # Show preview window so users can see the first samples immediately.
    try:
        plt.show()
    except Exception:
        pass
    finally:
        plt.close(fig)


def load_data_vqa(
    batch_size, 
    seq_len, 
    deterministic=False, 
    args=None, 
    model_emb=None,
    transform=None,
    split=None,
    loaded_vocab=None,
    loop=True,
):

    logger.log('#' * 30 + '\nLoading text data...')

    data, data_lst = get_corpus(args, seq_len, split=split, loaded_vocab=loaded_vocab)

    dataset = ImageTextDataset(args, transform=transform, text_datasets=data, data_lst=data_lst, data_args=args, model_emb=model_emb, split=split)

    dw = 0 if os.name == 'nt' else 2
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        drop_last=True,
        shuffle=True,
        num_workers=dw,
    )

    if loop and split == 'valid':
        return infinite_loader(data_loader)
    return data_loader

def infinite_loader(data_loader):
    while True:
        yield from data_loader

def helper_tokenize(sentence_lst, vocab_dict, seq_len, split):
    logger.log(f"RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")
    n = len(sentence_lst['question'])
    logger.log(f"Dataset size: {n}")

    input_id_q = []
    input_id_a = []
    chunk_size = 512
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        q_batch = sentence_lst['question'][start:end]
        a_batch = sentence_lst['answer'][start:end]

        tq = vocab_dict.tokenizer(q_batch, padding='max_length', max_length=seq_len, truncation=True, add_special_tokens=False)
        ta = vocab_dict.tokenizer(a_batch, padding='max_length', max_length=seq_len, truncation=True, add_special_tokens=False)

        input_id_q.extend(tq['input_ids'])
        input_id_a.extend(ta['input_ids'])

        logger.log(f"Tokenized {end} / {n} -- RAM: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")

    # Merge question and answer ids and create the mask (0 for question, 1 for answer)
    input_ids = [q + a for q, a in zip(input_id_q, input_id_a)]
    input_mask = [[0] * len(q) + [1] * len(a) for q, a in zip(input_id_q, input_id_a)]


    tokenized_group = {
        'input_id_q': input_id_q,
        'input_id_a': input_id_a,
        'input_ids': input_ids,
        'input_mask': input_mask,
    }

    tokenized_datasets = Dataset2.from_dict(tokenized_group)
    raw_datasets = datasets.DatasetDict()
    raw_datasets[split] = tokenized_datasets
    logger.log(f"Finished tokenization. RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")
    return raw_datasets


def get_corpus(args, seq_len, split, loaded_vocab=None):
    
    logger.log('#' * 30 + f"\nLoading dataset {args.dataset} from {args.data_dir}...")

    if args.dataset == 'ImageCLEFmed-MEDVQA-GI-2023-Development-Dataset' or args.dataset == 'Kvasir_VQA' or args.dataset == "pvqa" or args.dataset=="VQAMED2019":
        data_lst = {'question':[], 'answer': [], 'image_name': []}
    else:
        data_lst = {'question':[], 'answer': [], 'image_name': [], 'qid': [], 'img_id': []}

    if args.dataset.lower() == 'davekevin':
        logger.log('### Loading DaveKevin/MedVQA dataset from HuggingFace...')
        from datasets import load_dataset
        import os

        hf_token = os.environ.get('HF_TOKEN', None)
        try:
            hf_dataset = load_dataset(
                "DaveKevin/MedVQA", 
                split="train",
                token=hf_token,
                cache_dir=os.path.join(args.data_dir, '.hf_cache')
            )
        except Exception as e:
            logger.log(f'### Error loading from HuggingFace: {e}')
            logger.log('### Tip: Set HF_TOKEN environment variable or login with: huggingface-cli login')
            raise
        
        total_size = len(hf_dataset)
        train_size = int(0.8 * total_size)
        valid_size = int(0.1 * total_size)
        
        if split == 'train':
            logger.log(f'### Using samples 0-{train_size} for TRAIN set...')
            dataset_split = hf_dataset.select(range(0, train_size))
        elif split == 'valid':
            logger.log(f'### Using samples {train_size}-{train_size + valid_size} for VALID set...')
            dataset_split = hf_dataset.select(range(train_size, train_size + valid_size))
        elif split == 'test':
            logger.log(f'### Using samples {train_size + valid_size}-{total_size} for TEST set...')
            dataset_split = hf_dataset.select(range(train_size + valid_size, total_size))
        else:
            assert False, "invalid split for dataset"
        
        image_save_dir = os.path.join(args.data_dir, 'daveKevin_images', split)
        os.makedirs(image_save_dir, exist_ok=True)

        logger.log(f'### Processing {len(dataset_split)} samples from DaveKevin/MedVQA...')
        for idx, sample in enumerate(dataset_split):
            image_filename = f'medvqa_{split}_{idx:06d}.png'
            image_path = os.path.join(image_save_dir, image_filename)

            if not os.path.exists(image_path):
                try:
                    img = sample['image'].convert('RGB')
                    img.save(image_path)
                except Exception as e:
                    logger.log(f"Warning: Failed to save image {idx}: {e}")
                    Image.new('RGB', (224, 224), (0, 0, 0)).save(image_path)

            data_lst['question'].append(sample['question'].strip())
            data_lst['answer'].append(sample['answer'].strip())
            data_lst['image_name'].append(os.path.join(split, image_filename))
            
            if idx % 500 == 0:
                logger.log(f'### Processed {idx}/{len(dataset_split)} samples...')
        
        logger.log(f'### Finished processing DaveKevin/MedVQA {split} set')
        logger.log('### Data samples...')
        logger.log(f"questions: {data_lst['question'][:2]} answers: {data_lst['answer'][:2]} images: {data_lst['image_name'][:2]}")
        
        _preview_first_samples(args, data_lst, split, max_samples=3)
        
        vocab_dict = loaded_vocab
        train_dataset = helper_tokenize(data_lst, vocab_dict, seq_len, split)
        return train_dataset, data_lst

    if split == 'train':
        logger.log('### Loading from the TRAIN set...')
        path = f'{args.data_dir}/train.jsonl'
    elif split == 'valid':
        logger.log('### Loading from the VALID set...')
        path = f'{args.data_dir}/valid.jsonl'
    elif split == 'test':
        logger.log('### Loading from the TEST set...')
        path = f'{args.data_dir}/test.jsonl'
    else:
        assert False, "invalid split for dataset"

    # Open files using utf-8 and replace invalid bytes to avoid UnicodeDecodeError
    with open(path, 'rb') as f_reader:
        line_num = 0
        for raw_line in f_reader:
            line_num += 1
            # decode using utf-8 with replacement for invalid bytes
            try:
                line = raw_line.decode('utf-8')
            except Exception:
                line = raw_line.decode('utf-8', errors='replace')
            if line.startswith('\ufeff'):
                line = line.lstrip('\ufeff')
            if not line.strip():
                continue
            try:
                content = json.loads(line)
            except json.JSONDecodeError:
                content = json.loads(line.encode('utf-8', errors='replace').decode('utf-8'))
            if args.dataset == 'VQAMED2019' or args.dataset == 'Med_RAD' or args.dataset == "pvqa" or args.dataset=="VQAMED2019":
                for key in ['question', 'answer', 'image_name']:
                    if key in content:
                        if isinstance(content[key], str):
                            data_lst[key].append(content[key].strip())
                        else:
                            data_lst[key].append(content[key])
            else:
                expected_keys = ['question', 'answer', 'qid', 'img_id']

                if not isinstance(content, dict):
                    logger.log(f"Warning: record is not a dict (file {path}, line {line_num}), skipping.")
                    continue

                for key in expected_keys:
                    if key in content:
                        val = content[key]
                        data_lst[key].append(val.strip() if isinstance(val, str) else val)
                    else:
                        default = -1 if key in ('qid', 'img_id') else ''
                        data_lst[key].append(default)
                        logger.log(f"Warning: missing key '{key}' in record (file {path}, line {line_num}).")

                if 'img_name' in content:
                    img_val = content['img_name']
                    data_lst['image_name'].append(img_val.strip() if isinstance(img_val, str) else img_val)
                elif 'image_name' in content:
                    img_val = content['image_name']
                    data_lst['image_name'].append(img_val.strip() if isinstance(img_val, str) else img_val)
                else:
                    data_lst['image_name'].append('')

    logger.log('### Data samples...')
    logger.log(f"questions: {data_lst['question'][:2]} answers: {data_lst['answer'][:2]} images: {data_lst['image_name'][:2]}")

    _preview_first_samples(args, data_lst, split, max_samples=3)

    expected_len = len(data_lst['question'])
    for k, v in data_lst.items():
        if len(v) < expected_len:
            pad_value = -1 if (all(isinstance(x, int) for x in v if x is not None) and len(v) > 0) else ''
            data_lst[k].extend([pad_value] * (expected_len - len(v)))

    vocab_dict = loaded_vocab

    train_dataset = helper_tokenize(data_lst, vocab_dict, seq_len, split)
    return train_dataset, data_lst


class TextDataset(Dataset):
    def __init__(self, text_datasets, data_args, data_lst, model_emb=None, split=None):
        super().__init__()
        self.text_datasets = text_datasets
        self.length = len(self.text_datasets[split])
        self.data_args = data_args
        self.data_lst = data_lst
        self.model_emb = model_emb
        self.split = split

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        with torch.no_grad():
            input_ids = self.text_datasets[self.split][idx]['input_ids']
            hidden_state = self.model_emb(torch.tensor(input_ids))
            arr = np.array(hidden_state, dtype=np.float32)
            out_kwargs = {
                'input_q_id': np.array(self.text_datasets[self.split][idx]['input_id_q']),
                'input_a_id': np.array(self.text_datasets[self.split][idx]['input_id_a']),
                'input_ids':  np.array(self.text_datasets[self.split][idx]['input_ids']),
                'input_mask': np.array(self.text_datasets[self.split][idx]['input_mask']),
                'image_name': self.data_lst['image_name'][idx],
            }
            return arr, out_kwargs

class ImageDataset(Dataset):
    def __init__(self, image_root, data_lst, args, transform=None):
        super().__init__()
        self.image_root = image_root
        self.data_lst = data_lst
        self.args = args
        self.transform = transform

    def __len__(self):
        return len(self.data_lst['image_name'])

    def load_image_path(self):
        image_path = []
        for image_name in self.data_lst['image_name']:
            if self.args.dataset == 'daveKevin':
                image_path.append(f'{self.args.data_dir}/daveKevin_images/{image_name}')
            else:
                image_path.append(f'{self.image_root}/{image_name}')
        return image_path

    def __getitem__(self, idx):
        image_path = self.load_image_path()[idx]
        size = self.args.image_size if hasattr(self.args, 'image_size') else 224
        try:
            if not image_path or not os.path.exists(image_path):
                image = Image.new('RGB', (size, size), (0, 0, 0))
            else:
                image = Image.open(image_path).convert('RGB')
        except Exception:
            image = Image.new('RGB', (size, size), (0, 0, 0))

        if self.transform is not None:
            try:
                image = self.transform(image)
            except Exception:
                pass

        return image

class ImageTextDataset(Dataset):
    def __init__(self, args, transform=None, text_datasets=None, data_lst=None, data_args=None, model_emb=None, split=None):
        super().__init__()
        self.image_dataset = ImageDataset(args.image_dir, data_lst, args, transform)
        self.text_dataset = TextDataset(text_datasets, data_args, data_lst, model_emb, split)
        self.length = len(self.text_dataset)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        image = self.image_dataset[idx]
        _, cond = self.text_dataset[idx]
        return image, cond

def _collate_batch_helper(examples, pad_token_id, max_length, return_mask=False):

    result = torch.full([len(examples), max_length], pad_token_id, dtype=torch.int64).tolist()
    mask_ = torch.full([len(examples), max_length], pad_token_id, dtype=torch.int64).tolist()
    for i, example in enumerate(examples):
        curr_len = min(len(example), max_length)
        result[i][:curr_len] = example[:curr_len]
        mask_[i][:curr_len] = [1] * curr_len
    if return_mask:
        return result, mask_
    return result




