# Enhanced Metrics for DiffuVQA

This document describes the enhanced evaluation metrics added to the DiffuVQA project for more comprehensive evaluation of Visual Question Answering (VQA) models, particularly in medical domains.

## Overview

The enhanced metrics system extends the original evaluation framework with modern, domain-specific metrics that provide deeper insights into model performance across different answer types, semantic understanding, and medical concept recognition.

## Enhanced Metrics Categories

### 1. Semantic Similarity Metrics

#### Sentence-BERT Similarity
- **Purpose**: Measures semantic similarity using modern transformer-based embeddings
- **Implementation**: Uses `all-MiniLM-L6-v2` model from sentence-transformers
- **Range**: 0.0 to 1.0 (higher is better)
- **Usage**: Better captures semantic equivalence than simple string matching

```python
semantic_similarity = calculate_semantic_similarity(predictions, references)
```

### 2. Answer Type-Specific Accuracy

#### Yes/No Accuracy
- **Purpose**: Specialized accuracy for binary (yes/no) questions
- **Implementation**: Direct string matching for "yes"/"no" answers
- **Output**: Accuracy and sample count for yes/no questions

#### Numeric Accuracy
- **Purpose**: Accuracy for questions requiring numeric answers
- **Implementation**: Numeric extraction with 10% tolerance
- **Features**: Handles approximate values (e.g., "about 5" vs "5")

#### Descriptive Accuracy
- **Purpose**: Accuracy for open-ended descriptive answers
- **Implementation**: Exact match for non-binary, non-numeric answers

### 3. Multi-BLEU Scores

#### BLEU-1 through BLEU-4
- **Purpose**: N-gram overlap evaluation at different granularities
- **Implementation**: Standard BLEU with smoothing
- **Usage**: BLEU-1 for word overlap, BLEU-4 for phrase-level similarity

```python
multi_bleu_results = calculate_multi_bleu(predictions, references)
# Returns: {'bleu_1': 0.75, 'bleu_2': 0.65, 'bleu_3': 0.55, 'bleu_4': 0.45}
```

### 4. Medical Concept Accuracy

#### Domain-Specific Term Recognition
- **Purpose**: Measures accuracy in medical terminology usage
- **Categories**:
  - **Anatomy**: heart, lung, liver, kidney, brain, etc.
  - **Conditions**: cancer, tumor, lesion, fracture, etc.
  - **Procedures**: surgery, biopsy, scan, x-ray, MRI, etc.
  - **Colors**: red, blue, green, yellow, etc.
  - **Descriptors**: large, small, normal, abnormal, etc.

- **Implementation**: Term overlap calculation between prediction and reference
- **Range**: 0.0 to 1.0 (proportion of medical terms correctly identified)

### 5. Confidence-Weighted Metrics

#### Confidence-Weighted Accuracy
- **Purpose**: Evaluates how well model confidence correlates with actual accuracy
- **Implementation**: Accuracy weighted by model confidence scores
- **Output**: Overall weighted accuracy, high/low confidence accuracy splits

#### Confidence Statistics
- **Metrics**:
  - `confidence_weighted_acc`: Accuracy weighted by confidence
  - `high_confidence_acc`: Accuracy for samples above median confidence
  - `low_confidence_acc`: Accuracy for samples below median confidence
  - `avg_confidence`: Average confidence score
  - `confidence_std`: Standard deviation of confidence scores

## Installation Requirements

Install additional dependencies for enhanced metrics:

```bash
pip install sentence-transformers>=2.0.0
pip install spacy>=3.4.0
python -m spacy download en_core_web_sm  # For NER (optional)
```

## Usage

### Basic Usage
The enhanced metrics are automatically calculated when running the standard evaluation:

```bash
python eval_DiffuVQA.py --folder samples
```

### Sample Output
```
******************************
ENHANCED METRICS
******************************
semantic_similarity: 0.8245
medical_concept_accuracy: 0.7521
Yes/No accuracy: 0.9123 (234 samples)
Numeric accuracy: 0.7456 (89 samples) 
Descriptive accuracy: 0.6789 (156 samples)
bleu_1: 0.7234
bleu_2: 0.6456
bleu_3: 0.5678
bleu_4: 0.4921
confidence_weighted_acc: 0.7891
high_confidence_acc: 0.8456
low_confidence_acc: 0.7234
avg_confidence: 0.8123
confidence_std: 0.1456
```

## Integration with Existing Workflow

The enhanced metrics are seamlessly integrated into the existing evaluation pipeline:

1. **Backward Compatibility**: All original metrics are preserved
2. **Automatic Detection**: Confidence scores are extracted if available in JSON data
3. **Graceful Fallback**: Missing dependencies don't break evaluation
4. **Extended Results**: Enhanced metrics are added to the output JSON file

## Advanced Features

### Medical Domain Adaptation
The medical concept accuracy metric can be extended with:
- Custom medical dictionaries
- UMLS (Unified Medical Language System) integration
- Disease-specific terminology

### Confidence Calibration
Future enhancements may include:
- Expected Calibration Error (ECE)
- Reliability diagrams
- Temperature scaling analysis

## Error Handling

The enhanced metrics system includes robust error handling:
- **Missing Dependencies**: Graceful fallback to available metrics
- **Malformed Data**: Skips problematic samples with warnings
- **Memory Management**: Efficient processing for large datasets

## Performance Considerations

- **Sentence-BERT**: Adds ~2-3 seconds per 100 samples
- **Medical Concepts**: Minimal overhead (<0.1s per 100 samples)
- **Multi-BLEU**: Comparable to original BLEU calculation
- **Memory Usage**: ~50MB additional for sentence transformer model

## Future Enhancements

Planned additions:
1. **Entity-based F1 scores** for medical entity recognition
2. **BARTScore** and **BLEURT** for advanced generation quality
3. **Calibration metrics** for confidence assessment
4. **Cross-domain robustness** evaluation
5. **Answer consistency** across paraphrased questions

## References

1. Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks
2. Zhang, T., et al. (2020). BERTScore: Evaluating Text Generation with BERT
3. Papineni, K., et al. (2002). BLEU: a Method for Automatic Evaluation of Machine Translation
4. Lin, C. Y. (2004). ROUGE: A Package for Automatic Evaluation of Summaries