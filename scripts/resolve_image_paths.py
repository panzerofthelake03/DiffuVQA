import os, json
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
datasets_dir = repo_root / 'datasets'
slake_imgs_root = datasets_dir / 'slake' / 'imgs' / 'imgs'

jsonl_files = ['train.jsonl', 'valid.jsonl', 'test.jsonl']

results = {}
for jf in jsonl_files:
    src = datasets_dir / jf
    if not src.exists():
        print(f"Missing {src}, skipping")
        continue
    out = datasets_dir / (jf.replace('.jsonl', '.resolved.jsonl'))
    backed = datasets_dir / (jf + '.bak')
    # backup original if not already
    if not backed.exists():
        src.replace(backed)
        # use backed as source now
        src = backed
    else:
        # original already backed, use the backup as source
        src = backed
    total = 0
    resolved = 0
    unresolved = 0
    with open(src, 'r', encoding='utf-8', errors='replace') as fin, open(out, 'w', encoding='utf-8') as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                fout.write(line + '\n')
                continue
            total += 1
            img_name = obj.get('img_name') or obj.get('image_name') or ''
            img_id = obj.get('img_id') or obj.get('img') or obj.get('img_id')
            # If img_name already points to an existing file relative to datasets_dir, keep it
            keep = False
            if isinstance(img_name, str) and img_name.strip():
                cand = datasets_dir / img_name
                if cand.exists() and cand.is_file():
                    keep = True
            if not keep:
                # attempt to resolve using img_id -> xmlab{img_id} folder
                resolved_path = None
                if img_id is not None:
                    folder = slake_imgs_root / f'xmlab{img_id}'
                    if folder.exists() and folder.is_dir():
                        # pick the first image file in the folder
                        for p in folder.iterdir():
                            if p.is_file() and p.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                                # make path relative to datasets_dir
                                resolved_path = os.path.join('slake', 'imgs', 'imgs', f'xmlab{img_id}', p.name)
                                break
                # fallback: try to use basename lookups across slake_imgs_root
                if resolved_path is None and isinstance(img_name, str) and img_name.strip():
                    base = os.path.basename(img_name)
                    # scan slake_imgs_root directories for this basename
                    for root, dirs, files in os.walk(slake_imgs_root):
                        if base in files:
                            rel = os.path.relpath(os.path.join(root, base), datasets_dir)
                            resolved_path = rel.replace('\\', '/')
                            break
                if resolved_path:
                    # write into obj under same key name as present
                    if 'img_name' in obj:
                        obj['img_name'] = resolved_path
                    elif 'image_name' in obj:
                        obj['image_name'] = resolved_path
                    resolved += 1
                else:
                    unresolved += 1
            else:
                resolved += 1
            fout.write(json.dumps(obj, ensure_ascii=False) + '\n')
    results[jf] = {'total': total, 'resolved': resolved, 'unresolved': unresolved, 'out_file': str(out)}

print('\nResolution summary:')
for jf, r in results.items():
    print(f"{jf}: total={r['total']}, resolved={r['resolved']}, unresolved={r['unresolved']}, out={r['out_file']}")

print('\nBackups: original files moved to .bak in datasets/; resolved files are *.resolved.jsonl')
