"""
Excel Export Module for DiffuVQA Evaluation Results and Training Logs
This module creates comprehensive Excel reports for model evaluation and training metrics
"""

import pandas as pd
import numpy as np
import json
import os
import glob
from datetime import datetime
from typing import Dict, List, Any, Optional
import re

class DiffuVQAExcelExporter:
    def __init__(self, output_dir: str = "reports"):
        """
        Initialize the Excel exporter
        
        Args:
            output_dir: Directory to save Excel files
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def export_evaluation_results(self, 
                                 results: Dict[str, Any], 
                                 model_name: str = "DiffuVQA",
                                 dataset_name: str = "Unknown",
                                 additional_info: Dict[str, Any] = None) -> str:
        """
        Export evaluation results to Excel file
        
        Args:
            results: Dictionary containing evaluation metrics
            model_name: Name of the model
            dataset_name: Name of the dataset
            additional_info: Additional information to include
            
        Returns:
            Path to the generated Excel file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{model_name}_{dataset_name}_evaluation_{timestamp}.xlsx"
        filepath = os.path.join(self.output_dir, filename)
        
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            # Sheet 1: Summary Metrics
            self._create_summary_sheet(writer, results, model_name, dataset_name, additional_info)
            
            # Sheet 2: Detailed Metrics
            self._create_detailed_metrics_sheet(writer, results)
            
            # Sheet 3: Answer Type Analysis
            self._create_answer_type_sheet(writer, results)
            
            # Sheet 4: Enhanced Metrics (if available)
            if self._has_enhanced_metrics(results):
                self._create_enhanced_metrics_sheet(writer, results)
                
        print(f"✅ Evaluation results exported to: {filepath}")
        return filepath
    
    def export_training_logs(self, 
                           log_dir: str,
                           model_name: str = "DiffuVQA",
                           include_wandb: bool = True) -> str:
        """
        Export training logs to Excel file
        
        Args:
            log_dir: Directory containing training logs
            model_name: Name of the model
            include_wandb: Whether to include wandb logs
            
        Returns:
            Path to the generated Excel file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{model_name}_training_logs_{timestamp}.xlsx"
        filepath = os.path.join(self.output_dir, filename)
        
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            # Sheet 1: Training Summary
            self._create_training_summary_sheet(writer, log_dir, model_name)
            
            # Sheet 2: Loss Curves
            if include_wandb:
                self._create_loss_curves_sheet(writer, log_dir)
            
            # Sheet 3: Hyperparameters
            self._create_hyperparameters_sheet(writer, log_dir)
            
            # Sheet 4: Model Checkpoints
            self._create_checkpoints_sheet(writer, log_dir)
                
        print(f"✅ Training logs exported to: {filepath}")
        return filepath
    
    def export_comprehensive_report(self,
                                  evaluation_results: Dict[str, Any],
                                  log_dir: str,
                                  model_name: str = "DiffuVQA",
                                  dataset_name: str = "Medical_VQA") -> str:
        """
        Export comprehensive report combining evaluation and training data
        
        Args:
            evaluation_results: Evaluation metrics
            log_dir: Training logs directory
            model_name: Model name
            dataset_name: Dataset name
            
        Returns:
            Path to the generated Excel file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{model_name}_{dataset_name}_comprehensive_report_{timestamp}.xlsx"
        filepath = os.path.join(self.output_dir, filename)
        
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            # Evaluation sheets
            self._create_summary_sheet(writer, evaluation_results, model_name, dataset_name)
            self._create_detailed_metrics_sheet(writer, evaluation_results)
            self._create_answer_type_sheet(writer, evaluation_results)
            
            if self._has_enhanced_metrics(evaluation_results):
                self._create_enhanced_metrics_sheet(writer, evaluation_results)
            
            # Training sheets
            self._create_training_summary_sheet(writer, log_dir, model_name)
            self._create_hyperparameters_sheet(writer, log_dir)
            self._create_checkpoints_sheet(writer, log_dir)
                
        print(f"✅ Comprehensive report exported to: {filepath}")
        return filepath
    
    def _create_summary_sheet(self, writer, results: Dict[str, Any], 
                            model_name: str, dataset_name: str, 
                            additional_info: Dict[str, Any] = None):
        """Create summary metrics sheet"""
        summary_data = []
        
        # Basic info
        summary_data.extend([
            ["Model Name", model_name],
            ["Dataset", dataset_name],
            ["Export Date", datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
            ["", ""]
        ])
        
        # Core metrics
        core_metrics = [
            ("Overall Accuracy", "acc"),
            ("Yes/No Accuracy", "acc_YN"),
            ("Open-Ended Accuracy", "acc_OE"),
            ("BLEU-1 Score", "avg_BLEU1_score"),
            ("ROUGE-L Score", "avg_ROUGE_L_score"),
            ("METEOR Score", "avg_meteor_score"),
            ("CIDEr Score", "avg_CIDer"),
            ("BERT Score", "avg_bert_score"),
            ("F1 Score", "avg_f1_score")
        ]
        
        summary_data.append(["=== CORE METRICS ===", ""])
        for name, key in core_metrics:
            value = results.get(key, "N/A")
            if isinstance(value, (int, float)):
                summary_data.append([name, f"{value:.4f}"])
            else:
                summary_data.append([name, str(value)])
        
        # Enhanced metrics if available
        if self._has_enhanced_metrics(results):
            summary_data.append(["", ""])
            summary_data.append(["=== ENHANCED METRICS ===", ""])
            
            enhanced_metrics = [
                ("Semantic Similarity", "semantic_similarity"),
                ("Clinical Similarity", "clinical_similarity"),
                ("Medical Concept Accuracy", "medical_concept_acc"),
                ("Fluency Score", "fluency_score"),
                ("Confidence Weighted Acc", "weighted_acc")
            ]
            
            for name, key in enhanced_metrics:
                value = results.get(key, "N/A")
                if isinstance(value, (int, float)):
                    summary_data.append([name, f"{value:.4f}"])
                else:
                    summary_data.append([name, str(value)])
        
        # Additional info
        if additional_info:
            summary_data.append(["", ""])
            summary_data.append(["=== ADDITIONAL INFO ===", ""])
            for key, value in additional_info.items():
                summary_data.append([key, str(value)])
        
        df_summary = pd.DataFrame(summary_data, columns=["Metric", "Value"])
        df_summary.to_excel(writer, sheet_name="Summary", index=False)
    
    def _create_detailed_metrics_sheet(self, writer, results: Dict[str, Any]):
        """Create detailed metrics sheet"""
        detailed_data = []
        
        # Group metrics by category
        categories = {
            "Text Generation Metrics": [
                "avg_BLEU1_score", "avg_ROUGE_L_score", "avg_meteor_score", 
                "avg_CIDer", "avg_bert_score"
            ],
            "Accuracy Metrics": [
                "acc", "acc_YN", "acc_OE", "avg_f1_score"
            ],
            "Enhanced Text Metrics": [
                "bleu_1", "bleu_2", "bleu_3", "bleu_4"
            ],
            "Semantic Metrics": [
                "semantic_similarity", "clinical_similarity", "medical_concept_acc"
            ],
            "Quality Metrics": [
                "fluency_score", "entity_overlap"
            ],
            "Confidence Metrics": [
                "weighted_acc", "high_conf_acc", "low_conf_acc"
            ]
        }
        
        for category, metrics in categories.items():
            detailed_data.append([category, "", "", ""])
            for metric in metrics:
                if metric in results:
                    value = results[metric]
                    if isinstance(value, (int, float)):
                        detailed_data.append(["", metric, f"{value:.6f}", self._get_metric_description(metric)])
                    else:
                        detailed_data.append(["", metric, str(value), self._get_metric_description(metric)])
            detailed_data.append(["", "", "", ""])
        
        df_detailed = pd.DataFrame(detailed_data, columns=["Category", "Metric", "Value", "Description"])
        df_detailed.to_excel(writer, sheet_name="Detailed_Metrics", index=False)
    
    def _create_answer_type_sheet(self, writer, results: Dict[str, Any]):
        """Create answer type analysis sheet"""
        answer_type_data = []
        
        # Answer type metrics
        answer_types = [
            ("Yes/No Questions", "yes_no_acc", "yes_no_count"),
            ("Numeric Questions", "numeric_acc", "numeric_count"),
            ("Descriptive Questions", "descriptive_acc", "descriptive_count")
        ]
        
        answer_type_data.append(["Answer Type", "Accuracy", "Sample Count", "Percentage of Total"])
        
        total_samples = sum([results.get(f"{t[2]}", 0) for t in answer_types])
        
        for type_name, acc_key, count_key in answer_types:
            accuracy = results.get(acc_key, 0)
            count = results.get(count_key, 0)
            percentage = (count / total_samples * 100) if total_samples > 0 else 0
            
            answer_type_data.append([
                type_name,
                f"{accuracy:.4f}" if isinstance(accuracy, (int, float)) else str(accuracy),
                int(count) if isinstance(count, (int, float)) else str(count),
                f"{percentage:.2f}%"
            ])
        
        # Add total row
        overall_acc = results.get("acc", 0)
        answer_type_data.append([
            "TOTAL",
            f"{overall_acc:.4f}" if isinstance(overall_acc, (int, float)) else str(overall_acc),
            int(total_samples),
            "100.00%"
        ])
        
        df_answer_types = pd.DataFrame(answer_type_data[1:], columns=answer_type_data[0])
        df_answer_types.to_excel(writer, sheet_name="Answer_Type_Analysis", index=False)
    
    def _create_enhanced_metrics_sheet(self, writer, results: Dict[str, Any]):
        """Create enhanced metrics sheet"""
        enhanced_data = []
        
        # Enhanced metrics categories
        enhanced_categories = {
            "Semantic Understanding": [
                ("Semantic Similarity", "semantic_similarity"),
                ("Clinical Similarity", "clinical_similarity"),
                ("Medical Concept Accuracy", "medical_concept_acc")
            ],
            "Multi-BLEU Analysis": [
                ("BLEU-1", "bleu_1"),
                ("BLEU-2", "bleu_2"),
                ("BLEU-3", "bleu_3"),
                ("BLEU-4", "bleu_4")
            ],
            "Quality Assessment": [
                ("Fluency Score", "fluency_score"),
                ("Entity Overlap", "entity_overlap")
            ],
            "Confidence Analysis": [
                ("Weighted Accuracy", "weighted_acc"),
                ("High Confidence Accuracy", "high_conf_acc"),
                ("Low Confidence Accuracy", "low_conf_acc")
            ]
        }
        
        enhanced_data.append(["Category", "Metric", "Value", "Interpretation"])
        
        for category, metrics in enhanced_categories.items():
            for metric_name, metric_key in metrics:
                value = results.get(metric_key, "N/A")
                interpretation = self._interpret_metric(metric_key, value)
                
                if isinstance(value, (int, float)):
                    enhanced_data.append([category, metric_name, f"{value:.4f}", interpretation])
                else:
                    enhanced_data.append([category, metric_name, str(value), interpretation])
        
        df_enhanced = pd.DataFrame(enhanced_data[1:], columns=enhanced_data[0])
        df_enhanced.to_excel(writer, sheet_name="Enhanced_Metrics", index=False)
    
    def _create_training_summary_sheet(self, writer, log_dir: str, model_name: str):
        """Create training summary sheet"""
        training_data = []
        
        # Basic training info
        training_data.extend([
            ["Model Name", model_name],
            ["Training Date", datetime.now().strftime("%Y-%m-%d")],
            ["Log Directory", log_dir],
            ["", ""]
        ])
        
        # Try to find training args
        try:
            args_file = os.path.join(log_dir, "training_args.json")
            if os.path.exists(args_file):
                with open(args_file, 'r') as f:
                    args = json.load(f)
                
                training_data.append(["=== TRAINING CONFIGURATION ===", ""])
                key_params = [
                    "batch_size", "lr", "learning_steps", "seq_len", 
                    "model", "dataset", "seed", "checkpoint_path"
                ]
                
                for param in key_params:
                    if param in args:
                        training_data.append([param, str(args[param])])
        except Exception as e:
            training_data.append(["Training Args", f"Error loading: {e}"])
        
        # Look for checkpoint files
        checkpoint_pattern = os.path.join(log_dir, "*.pt")
        checkpoints = glob.glob(checkpoint_pattern)
        
        if checkpoints:
            training_data.append(["", ""])
            training_data.append(["=== CHECKPOINTS ===", ""])
            training_data.append(["Total Checkpoints", str(len(checkpoints))])
            training_data.append(["Latest Checkpoint", os.path.basename(sorted(checkpoints)[-1]) if checkpoints else "None"])
        
        df_training = pd.DataFrame(training_data, columns=["Parameter", "Value"])
        df_training.to_excel(writer, sheet_name="Training_Summary", index=False)
    
    def _create_hyperparameters_sheet(self, writer, log_dir: str):
        """Create hyperparameters sheet"""
        try:
            args_file = os.path.join(log_dir, "training_args.json")
            if os.path.exists(args_file):
                with open(args_file, 'r') as f:
                    args = json.load(f)
                
                # Convert to DataFrame
                hyperparams_data = []
                for key, value in sorted(args.items()):
                    hyperparams_data.append([key, str(value), self._get_param_description(key)])
                
                df_hyperparams = pd.DataFrame(hyperparams_data, columns=["Parameter", "Value", "Description"])
                df_hyperparams.to_excel(writer, sheet_name="Hyperparameters", index=False)
            else:
                # Create empty sheet with message
                df_empty = pd.DataFrame([["No training_args.json found", "", ""]], 
                                      columns=["Parameter", "Value", "Description"])
                df_empty.to_excel(writer, sheet_name="Hyperparameters", index=False)
        except Exception as e:
            df_error = pd.DataFrame([["Error loading hyperparameters", str(e), ""]], 
                                  columns=["Parameter", "Value", "Description"])
            df_error.to_excel(writer, sheet_name="Hyperparameters", index=False)
    
    def _create_checkpoints_sheet(self, writer, log_dir: str):
        """Create checkpoints information sheet"""
        checkpoint_data = []
        
        # Find checkpoint files
        checkpoint_patterns = ["*.pt", "model_*.pth", "ema_*.pt"]
        checkpoints = []
        
        for pattern in checkpoint_patterns:
            checkpoints.extend(glob.glob(os.path.join(log_dir, pattern)))
        
        if checkpoints:
            checkpoint_data.append(["Checkpoint Name", "Size (MB)", "Modified Date", "Type"])
            
            for checkpoint in sorted(checkpoints):
                try:
                    stat = os.stat(checkpoint)
                    size_mb = stat.st_size / (1024 * 1024)
                    modified = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
                    
                    # Determine checkpoint type
                    filename = os.path.basename(checkpoint)
                    if "ema" in filename.lower():
                        ckpt_type = "EMA"
                    elif "model" in filename.lower():
                        ckpt_type = "Model"
                    elif "optim" in filename.lower():
                        ckpt_type = "Optimizer"
                    else:
                        ckpt_type = "Unknown"
                    
                    checkpoint_data.append([filename, f"{size_mb:.2f}", modified, ckpt_type])
                    
                except Exception as e:
                    checkpoint_data.append([os.path.basename(checkpoint), "Error", "Error", "Error"])
        else:
            checkpoint_data.append(["No checkpoints found", "", "", ""])
        
        df_checkpoints = pd.DataFrame(checkpoint_data[1:] if len(checkpoint_data) > 1 else checkpoint_data,
                                    columns=checkpoint_data[0] if len(checkpoint_data) > 1 else ["Message", "", "", ""])
        df_checkpoints.to_excel(writer, sheet_name="Checkpoints", index=False)
    
    def _create_loss_curves_sheet(self, writer, log_dir: str):
        """Create loss curves sheet (placeholder for now)"""
        # This would parse wandb logs or other training logs
        # For now, create a placeholder
        loss_data = [
            ["Step", "Training Loss", "Validation Loss", "Notes"],
            ["Placeholder", "Parse wandb logs", "Or training logs", "Future implementation"]
        ]
        
        df_loss = pd.DataFrame(loss_data[1:], columns=loss_data[0])
        df_loss.to_excel(writer, sheet_name="Loss_Curves", index=False)
    
    def _has_enhanced_metrics(self, results: Dict[str, Any]) -> bool:
        """Check if results contain enhanced metrics"""
        enhanced_keys = ["semantic_similarity", "clinical_similarity", "bleu_1", "medical_concept_acc"]
        return any(key in results for key in enhanced_keys)
    
    def _get_metric_description(self, metric_key: str) -> str:
        """Get description for a metric"""
        descriptions = {
            "acc": "Overall exact match accuracy",
            "acc_YN": "Accuracy on Yes/No questions",
            "acc_OE": "Accuracy on open-ended questions",
            "avg_BLEU1_score": "BLEU-1 score for text generation quality",
            "avg_ROUGE_L_score": "ROUGE-L score for text similarity",
            "avg_meteor_score": "METEOR score for semantic similarity",
            "avg_CIDer": "CIDEr score for consensus evaluation",
            "avg_bert_score": "BERT-based semantic similarity score",
            "semantic_similarity": "Sentence transformer semantic similarity",
            "clinical_similarity": "Clinical domain semantic similarity",
            "medical_concept_acc": "Medical terminology accuracy",
            "fluency_score": "Text fluency and coherence score",
            "bleu_1": "BLEU-1 (unigram overlap)",
            "bleu_2": "BLEU-2 (bigram overlap)",
            "bleu_3": "BLEU-3 (trigram overlap)",
            "bleu_4": "BLEU-4 (4-gram overlap)",
            "weighted_acc": "Confidence-weighted accuracy"
        }
        return descriptions.get(metric_key, "No description available")
    
    def _interpret_metric(self, metric_key: str, value: Any) -> str:
        """Interpret metric value"""
        if not isinstance(value, (int, float)):
            return "N/A"
        
        interpretations = {
            "semantic_similarity": "Excellent" if value > 0.8 else "Good" if value > 0.6 else "Fair" if value > 0.4 else "Poor",
            "clinical_similarity": "Excellent" if value > 0.8 else "Good" if value > 0.6 else "Fair" if value > 0.4 else "Poor",
            "medical_concept_acc": "Perfect" if value >= 0.95 else "Excellent" if value > 0.8 else "Good" if value > 0.6 else "Needs improvement",
            "fluency_score": "Excellent" if value > 0.8 else "Good" if value > 0.6 else "Fair" if value > 0.4 else "Poor",
            "acc": "Excellent" if value > 0.8 else "Good" if value > 0.6 else "Fair" if value > 0.4 else "Poor"
        }
        
        return interpretations.get(metric_key, f"Value: {value:.4f}")
    
    def _get_param_description(self, param_key: str) -> str:
        """Get description for hyperparameter"""
        descriptions = {
            "batch_size": "Number of samples per training batch",
            "lr": "Learning rate for optimization",
            "learning_steps": "Total number of training steps",
            "seq_len": "Maximum sequence length",
            "model": "Model architecture type",
            "dataset": "Training dataset name",
            "seed": "Random seed for reproducibility",
            "checkpoint_path": "Directory for saving checkpoints",
            "ema_rate": "Exponential moving average rate",
            "weight_decay": "L2 regularization weight decay",
            "gradient_clipping": "Gradient clipping threshold"
        }
        return descriptions.get(param_key, "Training parameter")


# Convenience functions for easy usage
def export_eval_results_to_excel(results_dict: Dict[str, Any], 
                                model_name: str = "DiffuVQA",
                                dataset_name: str = "Medical_VQA",
                                output_dir: str = "reports") -> str:
    """
    Quick function to export evaluation results to Excel
    
    Args:
        results_dict: Dictionary containing evaluation metrics
        model_name: Name of the model
        dataset_name: Name of the dataset  
        output_dir: Output directory for Excel file
        
    Returns:
        Path to generated Excel file
    """
    exporter = DiffuVQAExcelExporter(output_dir)
    return exporter.export_evaluation_results(results_dict, model_name, dataset_name)


def export_training_logs_to_excel(log_directory: str,
                                 model_name: str = "DiffuVQA", 
                                 output_dir: str = "reports") -> str:
    """
    Quick function to export training logs to Excel
    
    Args:
        log_directory: Directory containing training logs
        model_name: Name of the model
        output_dir: Output directory for Excel file
        
    Returns:
        Path to generated Excel file
    """
    exporter = DiffuVQAExcelExporter(output_dir)
    return exporter.export_training_logs(log_directory, model_name)


if __name__ == "__main__":
    # Example usage
    print("DiffuVQA Excel Export Module")
    print("Use this module to export evaluation results and training logs to Excel format")
    
    # Example: Export sample results
    sample_results = {
        "acc": 0.7420,
        "acc_YN": 0.7163,
        "acc_OE": 0.7550,
        "avg_BLEU1_score": 0.8837,
        "avg_ROUGE_L_score": 0.9231,
        "semantic_similarity": 0.8671,
        "medical_concept_acc": 1.0000
    }
    
    exporter = DiffuVQAExcelExporter()
    filepath = exporter.export_evaluation_results(
        sample_results, 
        "DiffuVQA_Sample", 
        "Medical_VQA_Test"
    )
    print(f"Sample Excel file created: {filepath}")