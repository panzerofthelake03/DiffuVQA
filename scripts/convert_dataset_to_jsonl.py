"""Convert dataset files to newline-delimited JSON (.jsonl) files expected by the repo.

This script accepts a dataset directory (e.g. datasets/slake) and will look for
common filenames (train.json, validation.json, test.json, train.jsonl, etc.) and
create corresponding train.jsonl, valid.jsonl and test.jsonl files in the same
folder.

It handles two common formats:
- a JSON array of objects (e.g. [ {...}, {...} ])
- an existing JSONL file (passes through)

Usage (PowerShell):
    python .\scripts\convert_dataset_to_jsonl.py --dir "C:\path\to\datasets\slake"

The script will not overwrite existing *.jsonl files unless you pass --overwrite.
"""

import argparse
import os
import json

COMMON_MAP = [
    ("train.json", "train.jsonl"),
    ("validation.json", "valid.jsonl"),
    ("valid.json", "valid.jsonl"),
    ("test.json", "test.jsonl"),
    ("train.jsonl", "train.jsonl"),
    ("valid.jsonl", "valid.jsonl"),
    ("test.jsonl", "test.jsonl"),
]


def convert_file(src_path, dst_path, overwrite=False):
    if os.path.exists(dst_path) and not overwrite:
        print(f"Destination already exists, skipping: {dst_path}")
        return

    with open(src_path, 'r', encoding='utf-8') as f:
        first_char = f.read(1)
        f.seek(0)
        if not first_char:
            print(f"Source is empty, skipping: {src_path}")
            return

        # Heuristic: if starts with '[' -> JSON array
        if first_char == '[':
            data = json.load(f)
            if not isinstance(data, list):
                raise RuntimeError(f"Expected list in JSON file: {src_path}")
            with open(dst_path, 'w', encoding='utf-8') as out:
                for obj in data:
                    out.write(json.dumps(obj, ensure_ascii=False) + '\n')
            print(f"Converted array JSON -> JSONL: {dst_path}")
        else:
            # assume JSONL or single JSON object per file
            # We attempt to validate first line as JSON; if OK, copy file
            try:
                # validate by reading first line
                _ = json.loads(f.readline())
                # copy the whole file
                f.seek(0)
                with open(dst_path, 'w', encoding='utf-8') as out:
                    for line in f:
                        out.write(line)
                print(f"Copied/validated JSONL file: {dst_path}")
            except Exception:
                # fallback: try to load whole file and iterate
                f.seek(0)
                try:
                    data = json.load(f)
                    if isinstance(data, list):
                        with open(dst_path, 'w', encoding='utf-8') as out:
                            for obj in data:
                                out.write(json.dumps(obj, ensure_ascii=False) + '\n')
                        print(f"Converted array JSON -> JSONL: {dst_path}")
                    else:
                        # single object -> write single-line JSONL
                        with open(dst_path, 'w', encoding='utf-8') as out:
                            out.write(json.dumps(data, ensure_ascii=False) + '\n')
                        print(f"Converted single JSON object -> JSONL: {dst_path}")
                except Exception as e:
                    print(f"Failed to parse {src_path}: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', required=True, help='Path to dataset directory containing train/validation/test files')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing jsonl files')
    args = parser.parse_args()

    d = args.dir
    if not os.path.isdir(d):
        print(f"Not a directory: {d}")
        return

    # Try to find files and convert
    for src_name, dst_name in COMMON_MAP:
        src_path = os.path.join(d, src_name)
        dst_path = os.path.join(d, dst_name)
        if os.path.exists(src_path):
            convert_file(src_path, dst_path, overwrite=args.overwrite)

    print("Done. Check the directory for train.jsonl, valid.jsonl, test.jsonl files.")


if __name__ == '__main__':
    main()
