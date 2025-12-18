import os, sys, glob, json
import numpy as np
import argparse
import torch
from datetime import datetime
from collections import defaultdict

from torchmetrics.text.rouge import ROUGEScore
from excel_export_module import record_evaluation_data

rougeScore = ROUGEScore()
# Temporarily disable BERT Score due to PyTorch version compatibility issues
HAS_BERT_SCORE = False
try:
    from bert_score import score
    HAS_BERT_SCORE = True
except Exception as e:
    print(f"Warning: BERT Score disabled due to: {e}")
    HAS_BERT_SCORE = False

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from nltk.tokenize import word_tokenize

from nltk.translate.meteor_score import meteor_score
try:
    from pycocoevalcap.cider.cider import Cider
    HAS_PYCOCOEVALCAP = True
except Exception:
    Cider = None
    HAS_PYCOCOEVALCAP = False
import nltk
from nltk.metrics import edit_distance

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from basic_utils import (
    load_defaults_config,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
    load_tokenizer
)

# Try to import enhanced metrics
HAS_ENHANCED_METRICS = False
try:
    from enhanced_eval_metrics import EnhancedMetrics
    HAS_ENHANCED_METRICS = True
    print("Enhanced metrics loaded successfully!")
except Exception as e:
    print(f"Warning: Enhanced metrics not available: {e}")
    HAS_ENHANCED_METRICS = False

# Try to import Excel export module
HAS_EXCEL_EXPORT = False
try:
    from excel_export_module import DiffuVQAExcelExporter
    HAS_EXCEL_EXPORT = True
    print("Excel export module loaded successfully!")
except Exception as e:
    print(f"Warning: Excel export not available: {e}")
    HAS_EXCEL_EXPORT = False
def create_argparser():
    defaults = dict()
    defaults.update(load_defaults_config())
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str, default='config/ema_0.9999_300000.pt.samples', help='path to the folder of decoded texts')
    parser.add_argument('--mbr', action='store_true', help='mbr decoding or not')
    parser.add_argument('--sos', type=str, default='[CLS]', help='start token of the sentence')
    parser.add_argument('--eos', type=str, default='[SEP]', help='end token of the sentence')
    parser.add_argument('--sep', type=str, default='[SEP]', help='sep token of the sentence')
    parser.add_argument('--pad', type=str, default='[PAD]', help='pad token of the sentence')
    return parser



def calculate_f1(labels, preds, threshold=0.8):
    tp, fp, fn = 0, 0, 0
    for label, pred in zip(labels, preds):
        similarity_score = 1 - edit_distance(label, pred) / max(len(label), len(pred))
        if similarity_score >= threshold:
            tp += 1
        else:
            fp += 1
    fn = len(labels) - tp
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1_score


def get_bleu(recover, reference,n=1):
    weights = tuple((1.0 / n for _ in range(n)))
    return sentence_bleu([reference.split()], recover.split(),weights=weights, smoothing_function=SmoothingFunction().method4)

def calculate_meteor(recover,reference):
    score = meteor_score([reference], recover)
    return score


def compute_tf_idf(vectorizer, sentences):
    tfidf_matrix = vectorizer.fit_transform(sentences)
    return tfidf_matrix.toarray()

def compute_cosine_similarity(a, b):
    return cosine_similarity(a, b)

def cider_score(candidates, references):
    vectorizer = TfidfVectorizer()
    all_sentences = candidates + references
    tfidf_matrix = compute_tf_idf(vectorizer, all_sentences)

    candidates_tfidf = np.array(tfidf_matrix[:len(candidates)])
    references_tfidf = np.array(tfidf_matrix[len(candidates):])

    cider_scores = []

    for i, candidate_tfidf in enumerate(candidates_tfidf):
        ref_tfidf = references_tfidf[i]
        similarity = compute_cosine_similarity(candidate_tfidf.reshape(1, -1), ref_tfidf.reshape(1, -1))
        cider_scores.append(similarity[0][0])

    return np.mean(cider_scores)

def selectBest(sentences):
    selfBleu = [[] for i in range(len(sentences))]
    for i, s1 in enumerate(sentences):
        for j, s2 in enumerate(sentences):
            score = get_bleu(s1, s2)
            selfBleu[i].append(score)
    for i, s1 in enumerate(sentences):
        selfBleu[i][i] = 0
    idx = np.argmax(np.sum(selfBleu, -1))
    return sentences[idx]


def diversityOfSet(sentences):
    selfBleu = []
    # print(sentences)
    for i, sentence in enumerate(sentences):
        for j in range(i + 1, len(sentences)):
            # print(sentence, sentences[j])
            score = get_bleu(sentence, sentences[j])
            selfBleu.append(score)
    if len(selfBleu) == 0:
        selfBleu.append(0)
    div4 = distinct_n_gram_inter_sent(sentences, 4)
    return np.mean(selfBleu), div4


def distinct_n_gram(hypn, n):
    dist_list = []
    for hyp in hypn:
        hyp_ngrams = []
        hyp_ngrams += nltk.ngrams(hyp.split(), n)
        total_ngrams = len(hyp_ngrams)
        unique_ngrams = len(list(set(hyp_ngrams)))
        if total_ngrams == 0:
            return 0
        dist_list.append(unique_ngrams / total_ngrams)
    return np.mean(dist_list)


def distinct_n_gram_inter_sent(hypn, n):
    hyp_ngrams = []
    for hyp in hypn:
        hyp_ngrams += nltk.ngrams(hyp.split(), n)
    total_ngrams = len(hyp_ngrams)
    unique_ngrams = len(list(set(hyp_ngrams)))
    if total_ngrams == 0:
        return 0
    dist_n = unique_ngrams / total_ngrams
    return dist_n


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='decoding args.')
    parser.add_argument('--folder', type=str, default='config/ema_0.9999_300000.pt.samples', help='path to the folder of decoded texts')
    parser.add_argument('--filename', type=str, default='', help='path to a single decoded text file')
    parser.add_argument('--mbr', action='store_true', help='mbr decoding or not')
    parser.add_argument('--sos', type=str, default='[CLS]', help='start token of the sentence')
    parser.add_argument('--eos', type=str, default='[SEP]', help='end token of the sentence')
    parser.add_argument('--sep', type=str, default='[SEP]', help='sep token of the sentence')
    parser.add_argument('--pad', type=str, default='[PAD]', help='pad token of the sentence')

    args = parser.parse_args()

    # Create separate argparser for tokenizer loading with default values
    arg_defaults = load_defaults_config()
    class DefaultArgs:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
    
    arg = DefaultArgs(**arg_defaults)
    tokenizer = load_tokenizer(arg)

    if args.filename:
        # If a single file is specified, use it directly
        files = [ args.folder + "\\" + args.filename]
    else:
        # Otherwise, use all files in the folder
        files = sorted(glob.glob(f"{args.folder}/*jsonl"))
        print(files)

    print(args.folder)
    sample_num = 0
    with open(files[0], 'r') as f:
        for row in f:
            sample_num += 1

    sentenceDict = defaultdict(list)
    referenceDict = defaultdict(list)
    sourceDict = defaultdict(list)
    for i in range(sample_num):
        sentenceDict[i] = []
        referenceDict[i] = []
        sourceDict[i] = []

    div4 = []
    selfBleu = []

    for path in files:
        print(path)
        sources = []
        references = []
        recovers = []
        bleu = []
        rougel = []
        meteor = []
        cider = []
        avg_len = []
        dist1 = []
       

    total_samples = 0
    with open(path, 'r') as f:
            acc = 0.
            acc_oe = 0.
            acc_yn = 0.
            c_oe = 1e-9
            c_yn = 1e-9
            cnt = 0
            for row in f:
                source = json.loads(row)['question'].strip()
                reference = json.loads(row)['reference_answer'].strip()
                recover = json.loads(row)['generate_answer'].strip()
                source = source.replace(args.eos, '').replace(args.sos, '').strip()
                reference = reference.replace(args.eos, '').replace(args.sos, '').replace(args.sep, '').strip()
                recover = recover.replace(args.eos, '').replace(args.sos, '').replace(args.sep, '').replace(args.pad, '').strip()
                recover_token = word_tokenize(recover)
                reference_token = word_tokenize(reference)
                source_token = word_tokenize(source)
                if recover == reference:
                    acc += 1
                if reference == 'yes' or reference == 'no':
                    if recover == reference:
                        
                        acc_yn += 1
                    c_yn += 1
                elif reference != 'yes' and reference != 'no':
                    if reference == recover:
                        acc_oe += 1
                    c_oe += 1

                sources.append(source)
                references.append(reference)
                recovers.append(recover)
                avg_len.append(len(recover.split(' ')))
                bleu.append(get_bleu(recover, reference))
                rougel.append(rougeScore(recover, reference)['rougeL_fmeasure'].tolist())
                meteor.append(calculate_meteor(recover_token,reference_token))
                # cider.append(calculate_cider(recover,reference))
                dist1.append(distinct_n_gram([recover], 1))

                sentenceDict[cnt].append(recover)
                referenceDict[cnt].append(reference)
                sourceDict[cnt].append(source)
                cnt += 1
            
            total_samples += cnt
            accuracy = acc / cnt

            # Calculate BERT Score only if available
            bert_f1_score = 0.0
            if HAS_BERT_SCORE:
                try:
                    P, R, F1 = score(recovers, references, model_type='microsoft/deberta-xlarge-mnli', lang='en', verbose=True)
                    bert_f1_score = torch.mean(F1).item()
                except Exception as e:
                    print(f"Warning: BERT Score calculation failed: {e}")
                    bert_f1_score = 0.0
            else:
                print("Warning: BERT Score not available, skipping...")
                
            precision, recall, f1_score = calculate_f1(references, recovers)
            CIDer =  cider_score(references, recovers)

            # Calculate Enhanced Metrics if available
            enhanced_results = {}
            if HAS_ENHANCED_METRICS:
                try:
                    print("Calculating enhanced metrics...")
                    enhanced_metrics = EnhancedMetrics()
                    
                    # Extract confidence scores if available
                    confidences = []
                    for recover in recovers:
                        # Try to extract confidence from metadata if available
                        # For now, use a default confidence
                        confidences.append(0.8)  # Default confidence
                    
                    enhanced_results = enhanced_metrics.comprehensive_evaluate(
                        recovers, references, confidences
                    )
                    print("Enhanced metrics calculated successfully!")
                except Exception as e:
                    print(f"Warning: Enhanced metrics calculation failed: {e}")
                    enhanced_results = {}
            else:
                print("Enhanced metrics not available, skipping...")

            import json

            print('*' * 30)
            print('avg BLEU1 score', np.mean(bleu))
            print('avg ROUGE-L score', np.mean(rougel))
            print('avg meteor score', np.mean(meteor))
            print('avg cider score', CIDer)
            print('avg bert_score', bert_f1_score)
            print('avg f1_score', f1_score)
            print('acc', accuracy)
            print('acc_YN',acc_yn/c_yn)
            print('acc_OE', acc_oe/c_oe)
            
            # Print enhanced metrics if available
            if enhanced_results:
                print('*' * 30)
                print('ENHANCED METRICS')
                print('*' * 30)
                for metric_name, value in enhanced_results.items():
                    if isinstance(value, (int, float)):
                        print(f'{metric_name}: {value:.4f}')
                    else:
                        print(f'{metric_name}: {value}')

            results = {
                # Original metrics
                'avg_BLEU1_score': np.mean(bleu),
                'avg_ROUGE_L_score': np.mean(rougel),
                'avg_meteor_score': np.mean(meteor),
                'avg_CIDer': CIDer,
                'avg_bert_score': bert_f1_score,
                'avg_f1_score': f1_score,
                'acc': accuracy,
                'acc_YN':acc_yn/c_yn,
                'acc_OE': acc_oe/c_oe,
                # Enhanced metrics
                **enhanced_results
            }
            with open('ema_0.9999_300000.pt.samples.jsonl', 'w') as f:
                json.dump(results, f, indent=4)
            
            # Export to Excel if available
            if HAS_EXCEL_EXPORT:
                try:
                    print("Exporting results to Excel...")
                    exporter = DiffuVQAExcelExporter()
                    
                    # Determine model and dataset names from folder path
                    model_name = "DiffuVQA"
                    dataset_name = "Medical_VQA"
                    if "slake" in args.folder.lower():
                        dataset_name = "SLAKE"
                    elif "kvasir" in args.folder.lower():
                        dataset_name = "Kvasir-VQA"
                    elif "med" in args.folder.lower():
                        dataset_name = "Med-VQA"
                    
                    # Additional info
                    additional_info = {
                        "Sample Folder": args.folder,
                        "Total Samples": total_samples,
                        "Evaluation Date": str(datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
                        "MBR Mode": args.mbr,
                        "File Count": len(files)
                    }
                    
                    excel_path = exporter.export_evaluation_results(
                        results, model_name, dataset_name, additional_info
                    )
                    print(f"✅ Excel report saved to: {excel_path}")
                    
                except Exception as e:
                    print(f"Warning: Excel export failed: {e}")
            else:
                print("Excel export not available, skipping...")

    if len(files) > 1:
        if not args.mbr:
            print('*' * 30)
            print('Compute diversity...')
            print('*' * 30)
            for k, v in sentenceDict.items():
                if len(v) == 0:
                    continue
                sb, d4 = diversityOfSet(v)
                selfBleu.append(sb)
                div4.append(d4)

            print('avg selfBleu score', np.mean(selfBleu))
            print('avg div4 score', np.mean(div4))

        else:
            print('*' * 30)
            print('MBR...')
            print('*' * 30)
            bleu = []
            rougel = []
            avg_len = []
            dist1 = []
            recovers = []
            references = []
            sources = []

            for k, v in sentenceDict.items():
                if len(v) == 0 or len(referenceDict[k]) == 0:
                    continue

                recovers.append(selectBest(v))
                references.append(referenceDict[k][0])
                sources.append(sourceDict[k][0])

            for (source, reference, recover) in zip(sources, references, recovers):
                bleu.append(get_bleu(recover, reference))
                rougel.append(rougeScore(recover, reference)['rougeL_fmeasure'].tolist())
                avg_len.append(len(recover.split(' ')))
                dist1.append(distinct_n_gram([recover], 1))

            # print(len(recovers), len(references), len(recovers))

            # Calculate BERT Score for MBR if available
            mbr_bert_score = torch.tensor(0.0)
            if HAS_BERT_SCORE:
                try:
                    P, R, F1 = score(recovers, references, model_type='microsoft/deberta-xlarge-mnli', lang='en', verbose=True)
                    mbr_bert_score = torch.mean(F1)
                except Exception as e:
                    print(f"Warning: BERT Score calculation failed in MBR: {e}")
                    mbr_bert_score = torch.tensor(0.0)
            else:
                print("Warning: BERT Score not available for MBR, skipping...")

            print('*' * 30)
            print('avg BLEU score', np.mean(bleu))
            print('avg ROUGE-l score', np.mean(rougel))
            print('avg berscore', mbr_bert_score)
            print('avg dist1 score', np.mean(dist1))

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

        # Record evaluation results in an Excel file
        record_evaluation_data(out, output_dir="", dataset_name="Slake", model_name="DiffuVQA")

        return out