"""
Enhanced evaluation metrics for DiffuVQA project
Additional metrics that can be integrated into the existing evaluation system
"""

import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModel
import spacy
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from typing import List, Tuple, Dict
import nltk
from nltk.corpus import wordnet
from nltk.translate.bleu_score import sentence_bleu
from sklearn.metrics import precision_recall_fscore_support
import statistics

# Download required NLTK data
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

class EnhancedMetrics:
    def __init__(self):
        """Initialize models for semantic similarity metrics"""
        # Load sentence transformer for semantic similarity
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Load clinical/biomedical sentence transformer if available
        try:
            self.clinical_model = SentenceTransformer('emilyalsentzer/Bio_ClinicalBERT')
        except:
            self.clinical_model = None
            
        # Load spaCy model for named entity recognition
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Warning: spaCy model not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None

    # 1. SEMANTIC SIMILARITY METRICS
    def sentence_transformer_similarity(self, predictions: List[str], references: List[str]) -> float:
        """Calculate semantic similarity using sentence transformers"""
        pred_embeddings = self.sentence_model.encode(predictions)
        ref_embeddings = self.sentence_model.encode(references)
        
        similarities = util.cos_sim(pred_embeddings, ref_embeddings)
        # Get diagonal elements (pair-wise similarities)
        pair_similarities = [similarities[i][i].item() for i in range(len(predictions))]
        return np.mean(pair_similarities)

    def clinical_semantic_similarity(self, predictions: List[str], references: List[str]) -> float:
        """Calculate semantic similarity using clinical BERT if available"""
        if self.clinical_model is None:
            return self.sentence_transformer_similarity(predictions, references)
            
        pred_embeddings = self.clinical_model.encode(predictions)
        ref_embeddings = self.clinical_model.encode(references)
        
        similarities = util.cos_sim(pred_embeddings, ref_embeddings)
        pair_similarities = [similarities[i][i].item() for i in range(len(predictions))]
        return np.mean(pair_similarities)

    # 2. ANSWER TYPE CLASSIFICATION METRICS
    def answer_type_accuracy(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Calculate accuracy by answer type (yes/no, numeric, descriptive)"""
        results = {
            'yes_no_acc': 0.0,
            'numeric_acc': 0.0, 
            'descriptive_acc': 0.0,
            'yes_no_count': 0,
            'numeric_count': 0,
            'descriptive_count': 0
        }
        
        for pred, ref in zip(predictions, references):
            pred_clean = pred.strip().lower()
            ref_clean = ref.strip().lower()
            
            # Yes/No questions
            if ref_clean in ['yes', 'no']:
                results['yes_no_count'] += 1
                if pred_clean == ref_clean:
                    results['yes_no_acc'] += 1
                    
            # Numeric answers
            elif self._is_numeric_answer(ref_clean):
                results['numeric_count'] += 1
                if self._numeric_match(pred_clean, ref_clean):
                    results['numeric_acc'] += 1
                    
            # Descriptive answers
            else:
                results['descriptive_count'] += 1
                if pred_clean == ref_clean:
                    results['descriptive_acc'] += 1
        
        # Calculate accuracies
        if results['yes_no_count'] > 0:
            results['yes_no_acc'] /= results['yes_no_count']
        if results['numeric_count'] > 0:
            results['numeric_acc'] /= results['numeric_count']
        if results['descriptive_count'] > 0:
            results['descriptive_acc'] /= results['descriptive_count']
            
        return results

    def _is_numeric_answer(self, text: str) -> bool:
        """Check if answer is numeric"""
        # Remove common non-numeric words
        text = re.sub(r'\b(approximately|about|around|roughly)\b', '', text)
        # Check for numbers
        return bool(re.search(r'\d+', text))

    def _numeric_match(self, pred: str, ref: str) -> bool:
        """Check if numeric answers match (with some tolerance)"""
        pred_nums = re.findall(r'\d+\.?\d*', pred)
        ref_nums = re.findall(r'\d+\.?\d*', ref)
        
        if not pred_nums or not ref_nums:
            return pred == ref
            
        try:
            pred_val = float(pred_nums[0])
            ref_val = float(ref_nums[0])
            # Allow 10% tolerance for numeric answers
            return abs(pred_val - ref_val) / max(abs(ref_val), 1) <= 0.1
        except:
            return pred == ref

    # 3. ENTITY-BASED METRICS
    def entity_overlap_score(self, predictions: List[str], references: List[str]) -> float:
        """Calculate overlap of named entities between prediction and reference"""
        if self.nlp is None:
            return 0.0
            
        entity_scores = []
        
        for pred, ref in zip(predictions, references):
            pred_doc = self.nlp(pred)
            ref_doc = self.nlp(ref)
            
            pred_entities = set([ent.text.lower() for ent in pred_doc.ents])
            ref_entities = set([ent.text.lower() for ent in ref_doc.ents])
            
            if len(ref_entities) == 0:
                score = 1.0 if len(pred_entities) == 0 else 0.0
            else:
                score = len(pred_entities.intersection(ref_entities)) / len(ref_entities)
            
            entity_scores.append(score)
            
        return np.mean(entity_scores)

    # 4. MULTI-BLEU SCORES
    def multi_bleu_scores(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Calculate BLEU-1, BLEU-2, BLEU-3, BLEU-4 scores"""
        bleu_scores = {f'bleu_{i}': [] for i in range(1, 5)}
        
        for pred, ref in zip(predictions, references):
            pred_tokens = pred.split()
            ref_tokens = [ref.split()]
            
            for n in range(1, 5):
                weights = tuple((1.0/n if i < n else 0.0) for i in range(4))
                try:
                    score = sentence_bleu(ref_tokens, pred_tokens, weights=weights)
                    bleu_scores[f'bleu_{n}'].append(score)
                except:
                    bleu_scores[f'bleu_{n}'].append(0.0)
        
        return {k: np.mean(v) for k, v in bleu_scores.items()}

    # 5. MEDICAL CONCEPT ACCURACY
    def medical_concept_accuracy(self, predictions: List[str], references: List[str]) -> float:
        """Calculate accuracy for medical concepts/terms"""
        medical_terms = {
            'anatomy': ['heart', 'lung', 'liver', 'kidney', 'brain', 'stomach', 'intestine'],
            'conditions': ['cancer', 'tumor', 'lesion', 'fracture', 'inflammation', 'infection'],
            'procedures': ['surgery', 'biopsy', 'scan', 'x-ray', 'mri', 'ct', 'ultrasound'],
            'colors': ['red', 'blue', 'green', 'yellow', 'white', 'black', 'pink', 'brown']
        }
        
        concept_matches = []
        
        for pred, ref in zip(predictions, references):
            pred_lower = pred.lower()
            ref_lower = ref.lower()
            
            # Check if any medical terms are present in reference
            ref_has_medical = any(term in ref_lower for category in medical_terms.values() for term in category)
            
            if ref_has_medical:
                # Calculate term overlap
                pred_terms = set()
                ref_terms = set()
                
                for category, terms in medical_terms.items():
                    for term in terms:
                        if term in pred_lower:
                            pred_terms.add(term)
                        if term in ref_lower:
                            ref_terms.add(term)
                
                if len(ref_terms) > 0:
                    overlap = len(pred_terms.intersection(ref_terms)) / len(ref_terms)
                    concept_matches.append(overlap)
                else:
                    concept_matches.append(1.0 if len(pred_terms) == 0 else 0.0)
        
        return np.mean(concept_matches) if concept_matches else 0.0

    # 6. FLUENCY AND COHERENCE METRICS
    def calculate_fluency_score(self, predictions: List[str]) -> float:
        """Calculate fluency based on sentence structure and grammar"""
        fluency_scores = []
        
        for pred in predictions:
            # Basic fluency indicators
            score = 0.0
            
            # Check for complete sentences
            if pred.strip().endswith(('.', '!', '?')):
                score += 0.3
                
            # Check for reasonable length
            words = pred.split()
            if 3 <= len(words) <= 20:
                score += 0.3
            elif len(words) > 20:
                score += 0.1
                
            # Check for proper capitalization
            if pred.strip() and pred.strip()[0].isupper():
                score += 0.2
                
            # Check for reasonable word repetition
            if len(set(words)) / max(len(words), 1) > 0.7:
                score += 0.2
                
            fluency_scores.append(min(score, 1.0))
            
        return np.mean(fluency_scores)

    # 7. CONFIDENCE-WEIGHTED METRICS
    def confidence_weighted_accuracy(self, predictions: List[str], references: List[str], 
                                   confidences: List[float]) -> Dict[str, float]:
        """Calculate accuracy weighted by model confidence"""
        if len(confidences) != len(predictions):
            return {'weighted_acc': 0.0, 'high_conf_acc': 0.0, 'low_conf_acc': 0.0}
            
        # Calculate weighted accuracy
        correct = [1.0 if p.strip().lower() == r.strip().lower() else 0.0 
                  for p, r in zip(predictions, references)]
        
        weighted_acc = np.average(correct, weights=confidences)
        
        # Split into high and low confidence
        conf_threshold = np.median(confidences)
        high_conf_mask = np.array(confidences) >= conf_threshold
        low_conf_mask = np.array(confidences) < conf_threshold
        
        high_conf_acc = np.mean([correct[i] for i in range(len(correct)) if high_conf_mask[i]]) if np.any(high_conf_mask) else 0.0
        low_conf_acc = np.mean([correct[i] for i in range(len(correct)) if low_conf_mask[i]]) if np.any(low_conf_mask) else 0.0
        
        return {
            'weighted_acc': weighted_acc,
            'high_conf_acc': high_conf_acc,
            'low_conf_acc': low_conf_acc
        }

    # 8. COMPREHENSIVE EVALUATION
    def comprehensive_evaluate(self, predictions: List[str], references: List[str], 
                             confidences: List[float] = None) -> Dict[str, float]:
        """Run all enhanced metrics"""
        results = {}
        
        # Semantic similarity
        results['semantic_similarity'] = self.sentence_transformer_similarity(predictions, references)
        results['clinical_similarity'] = self.clinical_semantic_similarity(predictions, references)
        
        # Answer type accuracy
        type_results = self.answer_type_accuracy(predictions, references)
        results.update(type_results)
        
        # Entity overlap
        results['entity_overlap'] = self.entity_overlap_score(predictions, references)
        
        # Multi-BLEU
        bleu_results = self.multi_bleu_scores(predictions, references)
        results.update(bleu_results)
        
        # Medical concept accuracy
        results['medical_concept_acc'] = self.medical_concept_accuracy(predictions, references)
        
        # Fluency
        results['fluency_score'] = self.calculate_fluency_score(predictions)
        
        # Confidence-weighted metrics
        if confidences is not None:
            conf_results = self.confidence_weighted_accuracy(predictions, references, confidences)
            results.update(conf_results)
        
        return results


# Example usage and integration function
def integrate_enhanced_metrics():
    """Example of how to integrate these metrics into existing eval_DiffuVQA.py"""
    enhanced = EnhancedMetrics()
    
    # Sample data
    predictions = ["yes", "the heart shows normal function", "no visible abnormalities"]
    references = ["yes", "normal cardiac function", "no abnormalities detected"]
    confidences = [0.95, 0.78, 0.82]
    
    # Run comprehensive evaluation
    results = enhanced.comprehensive_evaluate(predictions, references, confidences)
    
    print("Enhanced Metrics Results:")
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")
    
    return results

if __name__ == "__main__":
    integrate_enhanced_metrics()