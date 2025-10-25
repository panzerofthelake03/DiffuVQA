import argparse
import torch
import json
import numpy as np
from transformers import AutoTokenizer


def find_embedding_tensor(state_dict):
    # common keys to check
    keys = list(state_dict.keys())
    candidates = []
    for k in keys:
        kl = k.lower()
        if 'word_embeddings.weight' in kl or 'wte.weight' in kl or 'embed_tokens.weight' in kl or 'lm_head.weight' in kl or 'embeddings.word_embeddings.weight' in kl:
            candidates.append(k)
    # choose largest candidate by first dim if multiple
    best = None
    best_sz = 0
    for k in candidates:
        v = state_dict[k]
        if hasattr(v, 'shape'):
            if v.shape[0] > best_sz:
                best_sz = v.shape[0]
                best = k
    return best


def load_checkpoint(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    if isinstance(ckpt, dict) and ('state_dict' in ckpt or 'model' in ckpt):
        if 'state_dict' in ckpt:
            sd = ckpt['state_dict']
        elif 'model' in ckpt:
            sd = ckpt['model']
        else:
            sd = ckpt
    else:
        sd = ckpt
    if not isinstance(sd, dict):
        raise RuntimeError('Checkpoint format unexpected (not a state-dict).')
    return sd


def topk_similar_rows(mat, indices, k=5):
    # mat: numpy (V, D)
    # indices: list of row indices to query
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms==0] = 1.0
    matn = mat / norms
    outs = {}
    for i in indices:
        q = matn[i:i+1]  # 1,D
        sims = (matn @ q.T).squeeze()  # V
        idxs = np.argsort(-sims)[:k]
        outs[i] = [(int(ii), float(sims[ii])) for ii in idxs]
    return outs


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True, help='path to checkpoint (.pt)')
    p.add_argument('--tokenizer', default='bert-base-uncased')
    p.add_argument('--show_ids', nargs='*', type=int, default=[0,1,2,10,50,100])
    args = p.parse_args()

    print('Loading checkpoint:', args.ckpt)
    sd = load_checkpoint(args.ckpt)
    emb_key = find_embedding_tensor(sd)
    if emb_key is None:
        print('No embedding tensor key automatically found in checkpoint. Available keys (sample):')
        for k in list(sd.keys())[:40]:
            print('  ', k)
        raise SystemExit(1)

    print('Embedding key found:', emb_key)
    emb = sd[emb_key]
    if isinstance(emb, torch.nn.Parameter):
        emb = emb.data
    emb = emb.cpu().numpy()
    print('Embedding shape:', emb.shape)

    print('Loading tokenizer:', args.tokenizer)
    tok = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    try:
        vocab_size = tok.vocab_size
    except Exception:
        vocab_size = len(tok)
    print('Tokenizer vocab size:', vocab_size)

    if emb.shape[0] != vocab_size:
        print('WARNING: embedding rows != tokenizer vocab size ->', emb.shape[0], 'vs', vocab_size)
    else:
        print('Embedding rows match tokenizer vocab size.')

    # topk nearest neighbours for provided ids
    ids = [i for i in args.show_ids if i < emb.shape[0]]
    print('Computing top-k neighbours for ids:', ids)
    neighbours = topk_similar_rows(emb, ids, k=5)
    for qid, nb in neighbours.items():
        toks = [tok.convert_ids_to_tokens(int(i)) for i,_ in nb]
        sims = [s for _,s in nb]
        print(f'Query id {qid} -> tokens: {toks}, sims: {sims}')
    # show special token ids
    st_map = {}
    for st in ['cls_token', 'pad_token', 'sep_token', 'unk_token', 'bos_token', 'eos_token']:
        try:
            v = getattr(tok, st)
            if v:
                st_map[st] = tok.convert_tokens_to_ids(v)
        except Exception:
            pass
    print('Tokenizer special tokens mapping (if present):', st_map)
    print('Done.')
