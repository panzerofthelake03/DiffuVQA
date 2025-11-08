import json,glob,collections,sys

paths = glob.glob('samples/*.jsonl')
if not paths:
    print('No sample files found in ./samples')
    sys.exit(1)

all_gens=[]
all_refs=[]
image_empty=0
cls_count=0
uously_count=0
tokens=collections.Counter()

for p in paths:
    with open(p,'r',encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row=json.loads(line)
            except Exception:
                continue
            gen=(row.get('generate_answer') or '').strip()
            ref=(row.get('reference_answer') or row.get('answer') or '').strip()
            image_name = row.get('image_name','')
            all_gens.append(gen)
            all_refs.append(ref)
            if not image_name:
                image_empty += 1
            cls_count += gen.count('[CLS]') + gen.count('[SEP]')
            if 'uously' in gen:
                uously_count += 1
            for t in gen.split():
                tokens[t]+=1

n=len(all_gens)
print('files:', paths)
print('samples:', n)
print('empty image_name:', image_empty, f'({image_empty/n:.2%})')
print('avg gen len (tokens):', sum(len(g.split()) for g in all_gens)/n)
print('avg ref len (tokens):', sum(len(r.split()) for r in all_refs)/n)
print('total [CLS]/[SEP] occurrences in gens:', cls_count)
print("#gens containing 'uously':", uously_count)
print('top 20 tokens in generated outputs:')
for t,c in tokens.most_common(20):
    print(f'  {t}: {c}')

print('\nExample generated outputs (first 8):')
for i,g in enumerate(all_gens[:8]):
    print('---',i,'---')
    print(g[:1000])

