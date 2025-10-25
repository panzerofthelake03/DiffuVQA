import argparse
import json
import re


def collapse_runs(tokens, max_run=3):
    out = []
    run_tok = None
    run_count = 0
    for t in tokens:
        if t == run_tok:
            run_count += 1
            if run_count <= max_run:
                out.append(t)
        else:
            run_tok = t
            run_count = 1
            out.append(t)
    return out


def clean_text(s, max_tokens=25):
    if not isinstance(s, str):
        return s
    # remove obvious special tokens
    s = re.sub(r'\[CLS\]|\[SEP\]|\[PAD\]|\[UNK\]', ' ', s)
    # remove long repeated 'uously' suffixes and any obvious repeated suffix
    s = re.sub(r'(uously){3,}', r'\1', s)
    # normalize whitespace
    s = ' '.join(s.split())
    # collapse repeated word runs (keep up to 3 repeats)
    toks = s.split()
    toks = collapse_runs(toks, max_run=3)
    # trim to max tokens
    toks = toks[:max_tokens]
    return ' '.join(toks).strip()


def process_file(in_path, out_path, max_tokens):
    written = 0
    with open(in_path, 'r', encoding='utf-8') as fin, open(out_path, 'w', encoding='utf-8') as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if 'generate_answer' in obj:
                obj['generate_answer'] = clean_text(obj.get('generate_answer', ''), max_tokens=max_tokens)
            if 'reference_answer' in obj:
                obj['reference_answer'] = clean_text(obj.get('reference_answer', ''), max_tokens=max_tokens)
            fout.write(json.dumps(obj, ensure_ascii=False) + '\n')
            written += 1
    return written


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--input', '-i', required=True)
    p.add_argument('--out', '-o', required=True)
    p.add_argument('--max_tokens', type=int, default=25)
    args = p.parse_args()
    n = process_file(args.input, args.out, args.max_tokens)
    print(f'Wrote {n} cleaned samples to {args.out}')
import json,glob,re,sys

IN='samples/*.jsonl'
OUT='samples/cleaned_samples.jsonl'

pattern_repeat = re.compile(r"\b(\S+)(?:\s+\1\b){2,}")
pattern_uously = re.compile(r"(uously){4,}")

cnt=0
with open(OUT,'w',encoding='utf-8') as fo:
    for p in glob.glob(IN):
        with open(p,'r',encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    obj=json.loads(line)
                except Exception:
                    continue
                gen = obj.get('generate_answer','')
                # remove special tokens
                gen = gen.replace('[CLS]','').replace('[SEP]','')
                gen = gen.replace('[PAD]','').replace('[UNK]','')
                # remove odd long 'uously' runs
                gen = pattern_uously.sub('uously', gen)
                # collapse 3+ repeated tokens to single occurrence
                # do several passes to be safe
                for _ in range(2):
                    gen = pattern_repeat.sub(r"\1", gen)
                # trim to first 25 tokens
                toks = gen.split()
                if len(toks) > 25:
                    toks = toks[:25]
                    gen = ' '.join(toks)
                # normalize whitespace
                gen = ' '.join(gen.split()).strip()
                if len(gen)==0:
                    gen = ''
                obj['generate_answer_clean'] = gen
                fo.write(json.dumps(obj, ensure_ascii=False) + '\n')
                cnt+=1
print('written',cnt,'cleaned samples to',OUT)
