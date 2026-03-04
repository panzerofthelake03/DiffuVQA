#!/usr/bin/env python3
"""
Test script for enhanced metrics in DiffuVQA
Demonstrates usage of new evaluation metrics with sample data
"""

import json
import numpy as np
from enhanced_eval_metrics import EnhancedMetrics

def create_sample_data():
    """Create sample VQA data for testing metrics"""
    sample_data = [
        {
            "question": "Is there any abnormality in the heart?",
            "reference_answer": "yes",
            "generate_answer": "yes",
            "confidence": 0.95
        },
        {
            "question": "What color is the tumor?",
            "reference_answer": "red",
            "generate_answer": "reddish",
            "confidence": 0.78
        },
        {
            "question": "How many lesions are visible?",
            "reference_answer": "3",
            "generate_answer": "approximately 3",
            "confidence": 0.82
        },
        {
            "question": "What organ is shown in the image?",
            "reference_answer": "lung",
            "generate_answer": "lung tissue",
            "confidence": 0.91
        },
        {
            "question": "Is the fracture visible?",
            "reference_answer": "no",
            "generate_answer": "no",
            "confidence": 0.88
        },
        {
            "question": "Describe the cardiac function",
            "reference_answer": "normal cardiac function",
            "generate_answer": "the heart shows normal function",
            "confidence": 0.76
        }
    ]
    return sample_data

def test_enhanced_metrics():
    """Test all enhanced metrics with sample data"""
    print("Testing Enhanced Metrics for DiffuVQA")
    print("=" * 50)
    
    # Create sample data
    sample_data = create_sample_data()
    
    # Extract data
    predictions = [item["generate_answer"] for item in sample_data]
    references = [item["reference_answer"] for item in sample_data]
    confidences = [item["confidence"] for item in sample_data]
    
    print(f"Sample size: {len(predictions)} predictions")
    print("\nSample data:")
    for i, item in enumerate(sample_data[:3]):  # Show first 3
        print(f"{i+1}. Q: {item['question']}")
        print(f"   Ref: '{item['reference_answer']}'")
        print(f"   Pred: '{item['generate_answer']}'")
        print(f"   Conf: {item['confidence']:.3f}")
    print("   ...")
    
    # Initialize enhanced metrics
    try:
        enhanced = EnhancedMetrics()
        print("\n✓ Enhanced metrics initialized successfully")
    except Exception as e:
        print(f"\n✗ Failed to initialize enhanced metrics: {e}")
        return
    
    # Test comprehensive evaluation
    print("\nRunning comprehensive evaluation...")
    try:
        results = enhanced.comprehensive_evaluate(predictions, references, confidences)
        
        print("\n" + "=" * 50)
        print("ENHANCED METRICS RESULTS")
        print("=" * 50)
        
        # Original-style metrics
        print(f"Semantic Similarity: {results.get('semantic_similarity', 0):.4f}")
        print(f"Clinical Similarity: {results.get('clinical_similarity', 0):.4f}")
        
        # Answer type accuracy
        print(f"\nAnswer Type Accuracy:")
        print(f"  Yes/No: {results.get('yes_no_acc', 0):.4f} ({results.get('yes_no_count', 0)} samples)")
        print(f"  Numeric: {results.get('numeric_acc', 0):.4f} ({results.get('numeric_count', 0)} samples)")
        print(f"  Descriptive: {results.get('descriptive_acc', 0):.4f} ({results.get('descriptive_count', 0)} samples)")
        
        # BLEU scores
        print(f"\nMulti-BLEU Scores:")
        for i in range(1, 5):
            bleu_key = f'bleu_{i}'
            if bleu_key in results:
                print(f"  BLEU-{i}: {results[bleu_key]:.4f}")
        
        # Medical and confidence metrics
        print(f"\nMedical Concept Accuracy: {results.get('medical_concept_acc', 0):.4f}")
        print(f"Fluency Score: {results.get('fluency_score', 0):.4f}")
        print(f"Entity Overlap: {results.get('entity_overlap', 0):.4f}")
        
        # Confidence metrics
        print(f"\nConfidence Metrics:")
        print(f"  Weighted Accuracy: {results.get('weighted_acc', 0):.4f}")
        print(f"  High Confidence Acc: {results.get('high_conf_acc', 0):.4f}")
        print(f"  Low Confidence Acc: {results.get('low_conf_acc', 0):.4f}")
        
        print("\n✓ All enhanced metrics calculated successfully!")
        
        # Save results
        output_file = "test_enhanced_metrics_results.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"✓ Results saved to {output_file}")
        
    except Exception as e:
        print(f"\n✗ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()

def test_individual_metrics():
    """Test individual metric functions"""
    print("\n" + "=" * 50)
    print("TESTING INDIVIDUAL METRICS")
    print("=" * 50)
    
    enhanced = EnhancedMetrics()
    
    # Test data
    preds = ["yes", "the heart is normal", "3 lesions"]
    refs = ["yes", "normal heart function", "three lesions"]
    confs = [0.9, 0.8, 0.7]
    
    # Test semantic similarity
    try:
        sim = enhanced.sentence_transformer_similarity(preds, refs)
        print(f"✓ Semantic similarity: {sim:.4f}")
    except Exception as e:
        print(f"✗ Semantic similarity failed: {e}")
    
    # Test answer type accuracy
    try:
        type_acc = enhanced.answer_type_accuracy(preds, refs)
        print(f"✓ Answer type accuracy: {type_acc}")
    except Exception as e:
        print(f"✗ Answer type accuracy failed: {e}")
    
    # Test multi-BLEU
    try:
        bleu_scores = enhanced.multi_bleu_scores(preds, refs)
        print(f"✓ Multi-BLEU scores: {bleu_scores}")
    except Exception as e:
        print(f"✗ Multi-BLEU failed: {e}")

def main():
    """Main test function"""
    print("DiffuVQA Enhanced Metrics Test Suite")
    print("====================================")
    
    # Test main comprehensive evaluation
    test_enhanced_metrics()
    
    # Test individual functions
    test_individual_metrics()
    
    print("\n" + "=" * 50)
    print("TEST COMPLETE")
    print("=" * 50)
    print("\nTo integrate with your evaluation:")
    print("1. Install dependencies: pip install sentence-transformers spacy")
    print("2. Run: python eval_DiffuVQA.py --folder your_samples_folder")
    print("3. Enhanced metrics will be included in output automatically")

if __name__ == "__main__":
    main()