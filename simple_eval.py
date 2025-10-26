#!/usr/bin/env python3
"""
Simple evaluation script for DiffuVQA results
"""
import json
import sys
from collections import defaultdict

def calculate_exact_match(reference, generated):
    """Calculate exact match accuracy"""
    return 1.0 if reference.strip().lower() == generated.strip().lower() else 0.0

def calculate_partial_match(reference, generated):
    """Calculate if reference answer appears in generated answer"""
    ref_clean = reference.strip().lower()
    gen_clean = generated.strip().lower()
    return 1.0 if ref_clean in gen_clean else 0.0

def clean_generated_answer(answer):
    """Clean generated answer by removing noise tokens"""
    # Remove common noise patterns
    cleaned = answer.replace("##uming", "").replace("##imating", "").replace("##lateral", "")
    # Remove extra spaces
    cleaned = " ".join(cleaned.split())
    return cleaned.strip()

def evaluate_file(filepath):
    """Evaluate a single JSONL file"""
    results = {
        'total_samples': 0,
        'exact_match': 0,
        'partial_match': 0,
        'exact_match_cleaned': 0,
        'samples': []
    }
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    sample = json.loads(line)
                    
                    ref_answer = sample.get('reference_answer', '')
                    gen_answer = sample.get('generate_answer', '')
                    confidence = sample.get('confidence', 0.0)
                    
                    # Calculate metrics
                    exact = calculate_exact_match(ref_answer, gen_answer)
                    partial = calculate_partial_match(ref_answer, gen_answer)
                    
                    # Clean the generated answer and recalculate
                    cleaned_gen = clean_generated_answer(gen_answer)
                    exact_cleaned = calculate_exact_match(ref_answer, cleaned_gen)
                    
                    results['total_samples'] += 1
                    results['exact_match'] += exact
                    results['partial_match'] += partial
                    results['exact_match_cleaned'] += exact_cleaned
                    
                    results['samples'].append({
                        'question': sample.get('question', ''),
                        'reference': ref_answer,
                        'generated': gen_answer,
                        'cleaned': cleaned_gen,
                        'exact_match': exact,
                        'partial_match': partial,
                        'exact_match_cleaned': exact_cleaned,
                        'confidence': confidence
                    })
    
    except Exception as e:
        print(f"Error reading file {filepath}: {e}")
        return None
    
    return results

def print_results(results, filename):
    """Print evaluation results"""
    if not results or results['total_samples'] == 0:
        print(f"No valid samples found in {filename}")
        return
    
    total = results['total_samples']
    
    print(f"\n=== Results for {filename} ===")
    print(f"Total samples: {total}")
    print(f"Exact Match: {results['exact_match']}/{total} = {results['exact_match']/total*100:.2f}%")
    print(f"Partial Match: {results['partial_match']}/{total} = {results['partial_match']/total*100:.2f}%")
    print(f"Exact Match (cleaned): {results['exact_match_cleaned']}/{total} = {results['exact_match_cleaned']/total*100:.2f}%")
    
    # Show some examples
    print(f"\n=== Sample Results ===")
    for i, sample in enumerate(results['samples'][:5]):  # Show first 5 samples
        print(f"\nSample {i+1}:")
        print(f"Q: {sample['question']}")
        print(f"Reference: '{sample['reference']}'")
        print(f"Generated: '{sample['generated']}'")
        print(f"Cleaned: '{sample['cleaned']}'")
        print(f"Exact/Partial/Cleaned: {sample['exact_match']:.0f}/{sample['partial_match']:.0f}/{sample['exact_match_cleaned']:.0f}")
        print(f"Confidence: {sample['confidence']:.6f}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python evaluate.py <jsonl_file1> [jsonl_file2] ...")
        sys.exit(1)
    
    for filepath in sys.argv[1:]:
        results = evaluate_file(filepath)
        if results:
            print_results(results, filepath)