#!/usr/bin/env python3
"""
quick_eval.py – Post-process generated answers and compute VQA metrics.

Post-processing applied (in order):
  1. Strip special tokens ([SEP], [CLS], [PAD])
  2. Strip leading commas / punctuation and trailing punctuation
  3. Deduplicate consecutive repeated tokens  ("Yes Yes" → "Yes")
  4. Trim to reference length when ref is short (≤ 3 tokens)

Metrics:
  exact_match_%   – exact match after lowercasing
  token_f1_%      – token-level F1 (VQA-style)
  bleu1_%         – BLEU-1
  partial_match_% – ≥1 ref token appears in gen tokens (case-insensitive)
  yesno_acc_%     – exact-match accuracy on yes/no-answer questions only
  hash_rate_%     – fraction of raw generated answers starting with ##
  avg_conf        – mean value of the 'confidence' field
  avg_nn_l2       – mean avg_nn_l2 parsed from the 'rationale' field
  avg_ans_len     – mean word count of the post-processed generated answer

Usage:
  python scripts/quick_eval.py FILE1.jsonl FILE2.jsonl ...
  python scripts/quick_eval.py --glob "C:/Users/BARIS/Downloads/ema_0.9999_029000*.jsonl"

When multiple files are supplied the results are printed as a comparison table.
"""

import argparse
import glob as glob_mod
import json
import re
import sys
from collections import Counter
from pathlib import Path

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------

SPECIAL_TOKENS = {"[SEP]", "[CLS]", "[PAD]", "[MASK]", "[UNK]"}


def _remove_special_tokens(text: str) -> str:
    for tok in SPECIAL_TOKENS:
        text = text.replace(tok, " ")
    return text


def _dedup_consecutive(tokens: list[str]) -> list[str]:
    """Remove consecutive duplicate tokens (case-insensitive comparison)."""
    out: list[str] = []
    for tok in tokens:
        if not out or tok.lower() != out[-1].lower():
            out.append(tok)
    return out


def postprocess(gen: str, ref: str = "", trim_to_ref: bool = True) -> str:
    gen = _remove_special_tokens(gen)
    # Strip leading commas / dashes / spaces, trailing punctuation / spaces
    gen = re.sub(r"^[\s,\-]+", "", gen)
    gen = re.sub(r"[\s,\-]+$", "", gen)
    gen = gen.strip()

    tokens = gen.split()
    tokens = _dedup_consecutive(tokens)

    if trim_to_ref and ref:
        ref_len = len(ref.split())
        if ref_len <= 3 and len(tokens) > ref_len:
            tokens = tokens[:ref_len]

    return " ".join(tokens)


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    return text.lower().strip()


def exact_match(gen: str, ref: str) -> float:
    return float(_normalize(gen) == _normalize(ref))


def token_f1(gen: str, ref: str) -> float:
    gen_toks = _normalize(gen).split()
    ref_toks = _normalize(ref).split()
    if not gen_toks or not ref_toks:
        return 0.0
    gen_counts = Counter(gen_toks)
    ref_counts = Counter(ref_toks)
    common = sum((gen_counts & ref_counts).values())
    if common == 0:
        return 0.0
    precision = common / len(gen_toks)
    recall = common / len(ref_toks)
    return 2 * precision * recall / (precision + recall)


def bleu1(gen: str, ref: str) -> float:
    gen_toks = gen.split()
    ref_toks = ref.split()
    if not gen_toks or not ref_toks:
        return 0.0
    try:
        return sentence_bleu(
            [ref_toks], gen_toks,
            weights=(1.0, 0, 0, 0),
            smoothing_function=SmoothingFunction().method1,
        )
    except Exception:
        return 0.0


def partial_match(gen: str, ref: str) -> float:
    gen_toks = set(_normalize(gen).split())
    ref_toks = set(_normalize(ref).split())
    if not ref_toks:
        return 0.0
    return float(bool(gen_toks & ref_toks))


def is_yesno(ref: str) -> bool:
    return _normalize(ref) in {"yes", "no"}


def has_hash_artifact(raw_gen: str) -> bool:
    """True if the raw (pre-processed) generated answer contains a ## token."""
    return "##" in raw_gen


def _parse_nn_l2(rationale: str) -> float | None:
    m = re.search(r"avg_nn_l2=([\d.]+)", rationale or "")
    return float(m.group(1)) if m else None


# ---------------------------------------------------------------------------
# Per-file evaluation
# ---------------------------------------------------------------------------

def evaluate_file(path: str, trim_to_ref: bool = True) -> dict:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue

    if not rows:
        return {}

    em_scores, f1_scores, b1_scores, pm_scores = [], [], [], []
    yn_correct, yn_total = 0, 0
    hash_count = 0
    confs, nn_l2s, ans_lens = [], [], []

    for row in rows:
        ref = (
            row.get("reference_answer") or row.get("reference") or
            row.get("answer") or row.get("ground_truth") or ""
        ).strip()
        raw_gen = (
            row.get("generate_answer") or row.get("generated_answer") or
            row.get("prediction") or row.get("recover") or ""
        ).strip()

        gen = postprocess(raw_gen, ref=ref, trim_to_ref=trim_to_ref)

        em = exact_match(gen, ref)
        f1 = token_f1(gen, ref)
        b1 = bleu1(gen, ref)
        pm = partial_match(gen, ref)

        em_scores.append(em)
        f1_scores.append(f1)
        b1_scores.append(b1)
        pm_scores.append(pm)

        if is_yesno(ref):
            yn_total += 1
            if _normalize(gen) == _normalize(ref):
                yn_correct += 1

        if has_hash_artifact(raw_gen):
            hash_count += 1

        conf = row.get("confidence")
        if conf is not None:
            confs.append(float(conf))

        nn_l2 = _parse_nn_l2(row.get("rationale", ""))
        if nn_l2 is not None:
            nn_l2s.append(nn_l2)

        ans_lens.append(len(gen.split()) if gen else 0)

    n = len(rows)

    def mean(lst):
        return sum(lst) / len(lst) if lst else 0.0

    return {
        "n": n,
        "exact_match_%": 100 * mean(em_scores),
        "token_f1_%": 100 * mean(f1_scores),
        "bleu1_%": 100 * mean(b1_scores),
        "partial_match_%": 100 * mean(pm_scores),
        "yesno_acc_%": 100 * yn_correct / yn_total if yn_total else 0.0,
        "hash_rate_%": 100 * hash_count / n,
        "avg_conf": mean(confs),
        "avg_nn_l2": mean(nn_l2s),
        "avg_ans_len": mean(ans_lens),
    }


# ---------------------------------------------------------------------------
# Table printing
# ---------------------------------------------------------------------------

METRICS = [
    "exact_match_%",
    "token_f1_%",
    "bleu1_%",
    "partial_match_%",
    "yesno_acc_%",
    "hash_rate_%",
    "avg_conf",
    "avg_nn_l2",
    "avg_ans_len",
]


def _short_label(path: str) -> str:
    name = Path(path).stem
    # e.g. ema_0.9999_082000.pt.seed102_step0_samplestep35_bsize64
    ckpt = re.search(r"(\d{5,6})\.pt", name)
    sstep = re.search(r"samplestep(\d+)", name)
    if ckpt and sstep:
        k = int(ckpt.group(1)) // 1000
        return f"{k:03d}k step{sstep.group(1)}"
    return name[:30]


def print_table(results: dict[str, dict]) -> None:
    labels = list(results.keys())
    col_w = max(14, max(len(l) for l in labels) + 2)
    metric_w = 16

    header = f"{'Metric':<{metric_w}}" + "".join(f"{l:>{col_w}}" for l in labels)
    print()
    print("Metrics Comparison — All Checkpoints (post-processed)")
    print("-" * len(header))
    print(header)
    print("-" * len(header))

    for m in METRICS:
        row = f"{m:<{metric_w}}"
        for label in labels:
            val = results[label].get(m, float("nan"))
            if m in ("avg_nn_l2",):
                row += f"{val:>{col_w}.2f}"
            elif m in ("avg_conf", "avg_ans_len"):
                row += f"{val:>{col_w}.2f}"
            else:
                row += f"{val:>{col_w}.2f}"
        print(row)

    print("-" * len(header))
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="*", help="JSONL sample file(s)")
    parser.add_argument("--glob", type=str, default=None, help="Glob pattern for JSONL files")
    parser.add_argument(
        "--no-trim", action="store_true",
        help="Disable trimming generated answer to reference length for short answers",
    )
    args = parser.parse_args()

    paths: list[str] = list(args.files)
    if args.glob:
        paths.extend(sorted(glob_mod.glob(args.glob)))
    paths = sorted(set(paths))

    if not paths:
        print("No files provided. Pass JSONL files as arguments or use --glob.")
        sys.exit(1)

    results: dict[str, dict] = {}
    for p in paths:
        label = _short_label(p)
        print(f"Evaluating {Path(p).name} …", flush=True)
        res = evaluate_file(p, trim_to_ref=not args.no_trim)
        if res:
            results[label] = res
        else:
            print(f"  WARNING: no rows loaded from {p}")

    if not results:
        print("No results to display.")
        sys.exit(1)

    if len(results) == 1:
        label, res = next(iter(results.items()))
        print(f"\n=== {label} ===")
        for m in METRICS:
            print(f"  {m:<20} {res.get(m, float('nan')):.4f}")
        print()
    else:
        print_table(results)


if __name__ == "__main__":
    main()
