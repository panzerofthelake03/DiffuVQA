import argparse
import torch
import json, os
import time

from diffuvqa import gaussian_diffusion as gd
from diffuvqa.gaussian_diffusion import SpacedDiffusion, space_timesteps
from diffuvqa.vqa_model import TransformerNetModel
from transformers import AutoTokenizer, PreTrainedTokenizerFast


MODEL_RUNTIME_PRESETS = {
    'transformer-bert': {
        'model': 'transformer-bert',
        'vocab': 'bert',
        'config_name': 'bert-base-uncased',
        'use_plm_init': 'bert',
    },
    'transformer-bio-bert': {
        'model': 'transformer-bio-bert',
        'vocab': 'bio-bert',
        'config_name': 'dmis-lab/biobert-v1.1',
        'use_plm_init': 'bert',
    },
    'transformer-roberta': {
        'model': 'transformer-roberta',
        'vocab': 'roberta',
        'config_name': 'roberta-large',
        'use_plm_init': 'roberta',
    },
}


def _normalize_runtime_choice(value):
    if value is None:
        return ''
    return str(value).strip().lower().replace('_', '-').replace(' ', '-')


def _preset_from_model(value):
    normalized = _normalize_runtime_choice(value)
    if normalized in MODEL_RUNTIME_PRESETS:
        return normalized
    return ''


def _preset_from_vocab(value):
    normalized = _normalize_runtime_choice(value)
    if normalized == 'bert':
        return 'transformer-bert'
    if normalized in {'bio-bert', 'biobert'}:
        return 'transformer-bio-bert'
    if normalized == 'roberta':
        return 'transformer-roberta'
    return ''


def _preset_from_config_name(value):
    normalized = _normalize_runtime_choice(value)
    if normalized == 'bert-base-uncased':
        return 'transformer-bert'
    if normalized == 'dmis-lab/biobert-v1.1':
        return 'transformer-bio-bert'
    if normalized == 'roberta-large':
        return 'transformer-roberta'
    return ''


def _preset_from_use_plm_init(value):
    normalized = _normalize_runtime_choice(value)
    if normalized in {'bio-bert', 'biobert'}:
        return 'transformer-bio-bert'
    if normalized == 'roberta':
        return 'transformer-roberta'
    if normalized == 'bert':
        return ''
    return ''


def _collect_runtime_family_signals(args):
    signals = []

    model_preset = _preset_from_model(getattr(args, 'model', None))
    if model_preset:
        signals.append(('model', model_preset, getattr(args, 'model', None)))

    vocab_preset = _preset_from_vocab(getattr(args, 'vocab', None))
    if vocab_preset:
        signals.append(('vocab', vocab_preset, getattr(args, 'vocab', None)))

    config_preset = _preset_from_config_name(getattr(args, 'config_name', None))
    if config_preset:
        signals.append(('config_name', config_preset, getattr(args, 'config_name', None)))

    init_preset = _preset_from_use_plm_init(getattr(args, 'use_plm_init', None))
    if init_preset:
        signals.append(('use_plm_init', init_preset, getattr(args, 'use_plm_init', None)))

    return signals


def resolve_runtime_model_args(args, strict=False):
    signals = _collect_runtime_family_signals(args)
    distinct_presets = sorted({preset for _, preset, _ in signals})

    if strict and len(distinct_presets) > 1:
        signal_summary = ", ".join(
            f"{source}={raw!r} -> {preset}" for source, preset, raw in signals
        )
        raise ValueError(
            "Conflicting runtime model settings detected: "
            f"{signal_summary}. Align model/vocab/config_name/use_plm_init first."
        )

    chosen_preset = ''
    for source_name in ('vocab', 'config_name', 'model', 'use_plm_init'):
        for source, preset, _ in signals:
            if source == source_name:
                chosen_preset = preset
                break
        if chosen_preset:
            break

    if not chosen_preset:
        chosen_preset = _preset_from_model(getattr(args, 'model', None))

    if not chosen_preset:
        return args

    preset_values = MODEL_RUNTIME_PRESETS[chosen_preset]
    for key, value in preset_values.items():
        setattr(args, key, value)

    if hasattr(args, 'language_encoder_name'):
        args.language_encoder_name = preset_values['config_name']

    return args


def validate_runtime_model_args(args):
    return resolve_runtime_model_args(args, strict=True)

class myTokenizer():
    """
    Load tokenizer from bert config or defined BPE vocab dict
    """
    ################################################
    ### You can custome your own tokenizer here. ###
    ################################################
    def __init__(self, args):
        resolve_runtime_model_args(args)
        if args.vocab == 'bert':
            tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            self.tokenizer = tokenizer
            self.sep_token_id = tokenizer.sep_token_id
            self.pad_token_id = tokenizer.pad_token_id
            # save
            tokenizer.save_pretrained(args.checkpoint_path)
            print('save tokenizer to', args.checkpoint_path)
            
        elif args.vocab == 'bio-bert':
            tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-v1.1")
            self.tokenizer = tokenizer
            self.sep_token_id = tokenizer.sep_token_id
            self.pad_token_id = tokenizer.pad_token_id
            tokenizer.save_pretrained(args.checkpoint_path)
            print('save tokenizer to', args.checkpoint_path)
        
        elif args.vocab == 'roberta':
            # Load RoBERTa tokenizer
            tokenizer = AutoTokenizer.from_pretrained("roberta-large")
            self.tokenizer = tokenizer
            self.sep_token_id = tokenizer.sep_token_id
            self.pad_token_id = tokenizer.pad_token_id
            # save
            tokenizer.save_pretrained(args.checkpoint_path)
            print('save tokenizer to', args.checkpoint_path)
        else: 
            # load vocab from the path
            print('#'*30, 'load vocab from', args.vocab)
            vocab_dict = {'[START]': 0, '[END]': 1, '[UNK]':2, '[PAD]':3}
            with open(args.vocab, 'r', encoding='utf-8') as f:
                for row in f:
                    vocab_dict[row.strip().split(' ')[0]] = len(vocab_dict)
            self.tokenizer = vocab_dict
            self.rev_tokenizer = {v: k for k, v in vocab_dict.items()}
            self.sep_token_id = vocab_dict['[END]']
            self.pad_token_id = vocab_dict['[PAD]']
            # save
            if int(os.environ['LOCAL_RANK']) == 0:
                path_save_vocab = f'{args.checkpoint_path}/vocab.json'
                with open(path_save_vocab, 'w') as f:
                    json.dump(vocab_dict, f)
                
        self.vocab_size = len(self.tokenizer)
        args.vocab_size = self.vocab_size # update vocab size in args
    
    def encode_token(self, sentences):
        if isinstance(self.tokenizer, dict):
            input_ids = [[0] + [self.tokenizer.get(x, self.tokenizer['[UNK]']) for x in seq.split()] + [1] for seq in sentences]
        elif isinstance(self.tokenizer, PreTrainedTokenizerFast):
            input_ids = self.tokenizer(sentences, add_special_tokens=True)['input_ids']
        else:
            assert False, "invalid type of vocab_dict"
        return input_ids
        
    def decode_token(self, seq):
        if isinstance(self.tokenizer, dict):
            seq = seq.squeeze(-1).tolist()
            while len(seq)>0 and seq[-1] == self.pad_token_id:
                seq.pop()
            tokens = " ".join([self.rev_tokenizer[x] for x in seq]).replace('__ ', '').replace('@@ ', '')
        elif isinstance(self.tokenizer, PreTrainedTokenizerFast):
            seq = seq.squeeze(-1).tolist()
            while len(seq)>0 and seq[-1] == self.pad_token_id:
                seq.pop()
            tokens = self.tokenizer.decode(seq, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        else:
            assert False, "invalid type of vocab_dict"
        return tokens

def load_model_emb(args, tokenizer):
    ### random emb or pre-defined embedding like glove embedding. You can customize your own init here.
    model = torch.nn.Embedding(tokenizer.vocab_size, args.hidden_dim)
    path_save = '{}/random_emb.torch'.format(args.checkpoint_path)
    path_save_ind = path_save + ".done"

    if os.path.exists(path_save):
        print('reload the random embeddings', model)
        try:
            model.load_state_dict(torch.load(path_save))
        except RuntimeError as e:
            # Handle vocab size mismatch (e.g., switching from BERT to RoBERTa)
            if "size mismatch" in str(e):
                print(f"Warning: Embedding size mismatch. Reinitializing embeddings.")
                print(f"  Old checkpoint vocab size likely doesn't match current tokenizer")
                print(f"  Current tokenizer vocab size: {tokenizer.vocab_size}")
                torch.nn.init.normal_(model.weight)
                torch.save(model.state_dict(), path_save)
                print(f"  Saved new embeddings to {path_save}")
            else:
                raise e
    else:
        print('initializing the random embeddings', model)
        torch.nn.init.normal_(model.weight)
        torch.save(model.state_dict(), path_save)

        with open(path_save, 'rb+') as f:  
            os.fsync(f.fileno())
        with open(path_save_ind, "x") as _:
            pass

    return model, tokenizer


def load_tokenizer(args):
    tokenizer = myTokenizer(args)
    return tokenizer

def load_defaults_config():
    """
    Load defaults for training args.
    """
    with open('diffuvqa/config.json', 'r') as f:
        return json.load(f)


def create_model_and_diffusion(args):
    resolve_runtime_model_args(args)
    # model = TransformerNetModel(
    #     input_dims=hidden_dim,
    #     output_dims=(hidden_dim if not learn_sigma else hidden_dim*2),
    #     hidden_t_dim=hidden_t_dim,
    #     dropout=dropout,
    #     config_name=config_name,
    #     vocab_size=vocab_size,
    #     init_pretrained=use_plm_init
    # )
    if args.model == 'unet':
        model = UnetForDDPM(args=args)
    
    elif args.model == 'transformer':
        model = TransformerNet(args=args)

    elif args.model in MODEL_RUNTIME_PRESETS:
        output_dims = args.hidden_dim * 2 if getattr(args, 'learn_sigma', False) else args.hidden_dim
        model = TransformerNetModel(
            input_dims=args.hidden_dim,
            output_dims=output_dims,
            hidden_t_dim=args.hidden_t_dim,
            dropout=args.dropout,
            config_name=args.config_name,
            vocab_size=args.vocab_size,
            init_pretrained=args.use_plm_init,
            args=args
        )

    betas = gd.get_named_beta_schedule(args.noise_schedule, args.diffusion_steps)

    if not args.timestep_respacing:
        args.timestep_respacing = [args.diffusion_steps]

    diffusion = SpacedDiffusion(
        use_timesteps=space_timesteps(args.diffusion_steps, args.timestep_respacing),
        betas=betas,
        rescale_timesteps=args.rescale_timesteps,
        predict_xstart=args.predict_xstart,
        learn_sigmas = args.learn_sigma,
        sigma_small = args.sigma_small,
        use_kl = args.use_kl,
        rescale_learned_sigmas=args.rescale_learned_sigmas
    )

    return model, diffusion


def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)


def args_to_dict(args, keys):
    return {k: getattr(args, k) for k in keys}


def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")
