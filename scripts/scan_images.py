import os, json
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
datasets_dir = os.path.join(repo_root, 'datasets')
candidate_roots = [datasets_dir,
                   os.path.join(datasets_dir, 'slake', 'imgs', 'imgs'),
                   os.path.join(datasets_dir, 'slake', 'imgs'),
                   os.path.join(datasets_dir, 'slake')]

jsonl_files = ['train.jsonl', 'valid.jsonl', 'test.jsonl']

summary = {}
for jn in jsonl_files:
    path = os.path.join(datasets_dir, jn)
    summary[jn] = {'total': 0, 'no_image_name': 0, 'dir_like': 0, 'found': 0, 'not_found': 0, 'samples_not_found': []}
    if not os.path.exists(path):
        print(f'MISSING JSONL: {path}')
        continue
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        for i, line in enumerate(f):
            line=line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            summary[jn]['total'] += 1
            img = obj.get('image_name') or obj.get('image') or ''
            if img is None or (isinstance(img, str) and img.strip()==""):
                summary[jn]['no_image_name'] += 1
                summary[jn]['not_found'] += 1
                if len(summary[jn]['samples_not_found'])<20:
                    summary[jn]['samples_not_found'].append((i, img))
                continue
            # normalize
            img = img.strip()
            # if looks like directory
            if img.endswith('/') or img.endswith('imgs') or img.endswith('imgs/'):
                summary[jn]['dir_like'] += 1
            found = False
            for root in candidate_roots:
                candidate = os.path.join(root, img)
                if os.path.exists(candidate) and os.path.isfile(candidate):
                    found = True
                    summary[jn]['found'] += 1
                    break
                # sometimes image_name stores only basename
                candidate2 = os.path.join(root, os.path.basename(img))
                if os.path.exists(candidate2) and os.path.isfile(candidate2):
                    found = True
                    summary[jn]['found'] += 1
                    break
            if not found:
                summary[jn]['not_found'] += 1
                if len(summary[jn]['samples_not_found'])<20:
                    summary[jn]['samples_not_found'].append((i, img))

print('\nDataset image resolution summary:')
for jn, stat in summary.items():
    print(f"\n{jn}: total={stat['total']}, found={stat['found']}, not_found={stat['not_found']}, no_image_name={stat['no_image_name']}, dir_like={stat['dir_like']}")
    if stat['samples_not_found']:
        print('  sample not found entries (index, image_name):')
        for s in stat['samples_not_found']:
            print('   ', s)

print('\nCandidate roots checked:')
for r in candidate_roots:
    print(' ', r)
