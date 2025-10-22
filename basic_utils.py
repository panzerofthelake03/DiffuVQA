import argparse
import torch
import json, os
import time

from diffuvqa import gaussian_diffusion as gd
from diffuvqa.gaussian_diffusion import SpacedDiffusion, space_timesteps
from diffuvqa.vqa_model import TransformerNetModel
from transformers import AutoTokenizer, PreTrainedTokenizerFast

class myTokenizer():
    """
    Load tokenizer from bert config or defined BPE vocab dict
    """
    ################################################
    ### You can custome your own tokenizer here. ###
    ################################################
    def __init__(self, args):
        if args.vocab == 'bert':
            tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
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
        model.load_state_dict(torch.load(path_save))
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

    elif args.model == 'transformer-bert':
        model =  TransformerNetModel(
        input_dims=768,
        output_dims=768,
        hidden_t_dim=128,
        dropout=0.1,
        config_name="bert-base-uncased",
        vocab_size=30522,
        init_pretrained="bert",
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
