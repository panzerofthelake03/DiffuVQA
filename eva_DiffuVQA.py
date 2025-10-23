import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score

# additional imports for CIDEr/BERTScore
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from bert_score import score as bert_score


# Lightweight ROUGE-L (LCS-based) implementation
def rouge_l_score(hypothesis, reference):
    h_tokens = hypothesis.split()
    r_tokens = reference.split()
    m = len(h_tokens)
    n = len(r_tokens)
    if m == 0 or n == 0:
        return 0.0
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m - 1, -1, -1):
        for j in range(n - 1, -1, -1):
            if h_tokens[i] == r_tokens[j]:
                dp[i][j] = 1 + dp[i + 1][j + 1]
            else:
                dp[i][j] = max(dp[i + 1][j], dp[i][j + 1])
    lcs = dp[0][0]
    prec = lcs / m
    rec = lcs / n
    if prec + rec == 0:
        return 0.0
    f = (2 * prec * rec) / (prec + rec)
    return f


def bleu1(hypothesis, reference):
    if len(hypothesis.strip()) == 0 or len(reference.strip()) == 0:
        return 0.0
    weights = (1.0, 0, 0, 0)
    try:
        return sentence_bleu([reference.split()], hypothesis.split(), weights=weights,
                             smoothing_function=SmoothingFunction().method1)
    except Exception:
        return 0.0


def exact_match(hypothesis, reference):
    return hypothesis.strip() == reference.strip()


def load_jsonl(path):
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except Exception:
                # tolerate malformed lines
                continue


def cider_score(candidates, references):
    # Simple CIDEr-like score: cosine similarity between TF-IDF vectors (per pair)
    if len(candidates) == 0:
        return 0.0
    vectorizer = TfidfVectorizer()
    all_sentences = list(candidates) + list(references)
    try:
        tfidf_matrix = vectorizer.fit_transform(all_sentences)
    except Exception:
        return 0.0
    candidates_tfidf = tfidf_matrix[:len(candidates)]
    references_tfidf = tfidf_matrix[len(candidates):]
    scores = []
    for i in range(len(candidates)):
        try:
            sim = cosine_similarity(candidates_tfidf[i], references_tfidf[i])
            scores.append(float(sim[0, 0]))
        except Exception:
            scores.append(0.0)
    return float(np.mean(scores))


def evaluate(files, bert_model='microsoft/deberta-xlarge-mnli'):
    stats = defaultdict(list)
    for path in files:
        for row in load_jsonl(path):
            q = (row.get('question', '') or row.get('source', '')).strip()
            # reference can be under several names
            ref = (row.get('reference_answer') or row.get('reference') or row.get('answer') or row.get('ground_truth') or '').strip()
            # generated text under common fields
            gen = (row.get('generate_answer') or row.get('generated_answer') or row.get('prediction') or row.get('generated') or row.get('recover') or '').strip()

            stats['questions'].append(q)
            stats['refs'].append(ref)
            stats['gens'].append(gen)

    refs = stats['refs']
    gens = stats['gens']

    n = len(refs)
    if n == 0:
        return {}

    exact = [exact_match(g, r) for g, r in zip(gens, refs)]
    bleu1_scores = [bleu1(g, r) for g, r in zip(gens, refs)]
    rougeL_scores = [rouge_l_score(g, r) for g, r in zip(gens, refs)]
    # METEOR expects pre-tokenized inputs (iterable of tokens). Tokenize safely.
    meteor_scores = []
    for g, r in zip(gens, refs):
        if len(g.strip()) == 0 or len(r.strip()) == 0:
            meteor_scores.append(0.0)
            continue
        try:
            # nltk.translate.meteor_score.meteor_score requires tokenized inputs
            meteor_scores.append(meteor_score([word_tokenize(r)], word_tokenize(g)))
        except Exception:
            meteor_scores.append(0.0)

    # CIDEr-like
    cider = cider_score(gens, refs)

    # BERTScore (may be slower) - returns P,R,F1 arrays
    try:
        P, R, F1 = bert_score(gens, refs, model_type=bert_model, lang='en', verbose=False)
        bert_f1 = float(np.mean(F1.tolist()))
    except Exception:
        bert_f1 = 0.0

    out = {
        'samples': n,
        'exact_match': float(np.mean(exact)),
        'bleu1': float(np.mean(bleu1_scores)),
        'rougeL': float(np.mean(rougeL_scores)),
        'meteor': float(np.mean(meteor_scores)),
        'cider_like': float(cider),
        'bert_score_f1': float(bert_f1),
    }
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--files', type=str, default=None, help='glob or comma-separated list of JSONL files (outputs)')
    parser.add_argument('--folder', type=str, default=None, help='folder containing JSONL sample files')
    parser.add_argument('--out', type=str, default='eval_results.json', help='output JSON file')
    args = parser.parse_args()

    files = []
    if args.files:
        for part in args.files.split(','):
            files.extend(glob.glob(part))
    if args.folder:
        files.extend(glob.glob(os.path.join(args.folder, '*.jsonl')))
    files = sorted(list(set(files)))

    if not files:
        print('No files found. Provide --files or --folder with JSONL files.')
        return

    results = evaluate(files)
    print('Evaluation results:', results)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()
