"""
DistilBERT Clickbait Model Evaluator
===================================

This module evaluates a pre-trained DistilBERT model for clickbait classification.
It loads the saved model, performs evaluation on test data, and generates 
classification reports and predictions.

Features:
- Model performance evaluation
- Classification report generation
- Prediction export to CSV
- Comprehensive metrics analysis

Author: Your Name
Date: 2025
"""

import pandas as pd
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import numpy as np
import os
from datetime import datetime
from typing import List, Tuple, Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns


class ModelEvaluator:
    """Comprehensive evaluation of trained DistilBERT clickbait classifier."""
    
    def __init__(self, model_path: str = './clickbait_model', data_path: str = 'clickbait_data.csv'):
        """
        Initialize evaluator with model and data paths.
        
        Args:
            model_path: Path to saved model directory
            data_path: Path to labeled dataset CSV file
        """
        self.model_path = model_path
        self.data_path = data_path
        self.tokenizer = None
        self.model = None
        self.trainer = None
        self.df = None
        
        # Load and prepare data
        self._load_data()
        self._load_model()
        self._prepare_datasets()
    
    def _load_data(self) -> None:
        """Load and validate dataset."""
        try:
            self.df = pd.read_csv(self.data_path, encoding='utf-8')
            self.df = self.df.dropna(subset=['headline', 'label'])
            print(f"✅ Loaded {len(self.df)} rows from dataset")
            
            # Validate data structure
            if 'headline' not in self.df.columns or 'label' not in self.df.columns:
                raise ValueError("Dataset must contain 'headline' and 'label' columns")
                
        except FileNotFoundError:
            print(f"❌ Dataset file not found: {self.data_path}")
            print("Please ensure the clickbait_data.csv file exists in the project directory")
            exit()
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            exit()
    
    def _load_model(self) -> None:
        """Load pre-trained tokenizer and model."""
        try:
            self.tokenizer = DistilBertTokenizer.from_pretrained(self.model_path)
            self.model = DistilBertForSequenceClassification.from_pretrained(self.model_path)
            print("✅ Model and tokenizer loaded successfully")
            
            # Setup trainer for evaluation
            training_args = TrainingArguments(
                output_dir='./evaluation_results',
                per_device_eval_batch_size=8,
                logging_dir='./evaluation_logs',
                dataloader_drop_last=False,
            )
            
            self.trainer = Trainer(
                model=self.model,
                args=training_args,
            )
            
        except Exception as e:
            print(f"❌ Error loading model from {self.model_path}: {e}")
            print("Please ensure the model directory exists and contains trained model files")
            exit()
    
    def _prepare_datasets(self) -> None:
        """Prepare train/test splits and tokenize data."""
        # Extract features and labels
        X = self.df['headline'].tolist()
        y = self.df['label'].astype(int).tolist()
        
        # Split data (80% train, 20% test)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"📊 Dataset split:")
        print(f"  Training set: {len(self.X_train)} samples")
        print(f"  Test set: {len(self.X_test)} samples")
        
        # Tokenize datasets
        self.train_encodings = self._tokenize_data(self.X_train)
        self.test_encodings = self._tokenize_data(self.X_test)
        
        # Create PyTorch datasets
        self.train_dataset = ClickbaitDataset(self.train_encodings, self.y_train)
        self.test_dataset = ClickbaitDataset(self.test_encodings, self.y_test)
    
    def _tokenize_data(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """
        Tokenize text data for model input.
        
        Args:
            texts: List of text strings
            
        Returns:
            Dictionary of tokenized tensors
        """
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=64,
            return_tensors='pt'
        )
    
    def evaluate_model(self) -> Dict[str, Any]:
        """
        Comprehensive model evaluation on test set.
        
        Returns:
            Dictionary containing evaluation metrics
        """
        print("\n🔍 Evaluating model on test set...")
        
        # Make predictions
        predictions = self.trainer.predict(self.test_dataset)
        y_pred = np.argmax(predictions.predictions, axis=1)
        
        # Calculate metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred, average='weighted')
        
        # Generate classification report
        class_report = classification_report(
            self.y_test, 
            y_pred, 
            target_names=['Non-clickbait', 'Clickbait'],
            output_dict=True
        )
        
        # Print results
        print("\n📈 Model Performance Results:")
        print("=" * 40)
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1-Score (weighted): {f1:.4f}")
        print("\nDetailed Classification Report:")
        print(classification_report(
            self.y_test, 
            y_pred, 
            target_names=['Non-clickbait', 'Clickbait']
        ))
        
        # Generate confusion matrix
        self._plot_confusion_matrix(self.y_test, y_pred)
        
        return {
            'accuracy': accuracy,
            'f1_score': f1,
            'classification_report': class_report,
            'predictions': y_pred,
            'true_labels': self.y_test
        }
    
    def _plot_confusion_matrix(self, y_true: List[int], y_pred: List[int]) -> None:
        """
        Generate and save confusion matrix plot.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
        """
        try:
            cm = confusion_matrix(y_true, y_pred)
            
            plt.figure(figsize=(8, 6))
            sns.heatmap(
                cm, 
                annot=True, 
                fmt='d', 
                cmap='Blues',
                xticklabels=['Non-clickbait', 'Clickbait'],
                yticklabels=['Non-clickbait', 'Clickbait']
            )
            plt.title('Confusion Matrix - Clickbait Classification')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            
            # Save plot
            os.makedirs('evaluation_results', exist_ok=True)
            plt.savefig('evaluation_results/confusion_matrix.png', dpi=300, bbox_inches='tight')
            plt.show()
            print("📊 Confusion matrix saved to evaluation_results/confusion_matrix.png")
            
        except ImportError:
            print("⚠️ Matplotlib/Seaborn not available - skipping confusion matrix plot")
    
    def predict_all_data(self) -> pd.DataFrame:
        """
        Generate predictions for entire dataset.
        
        Returns:
            DataFrame with original data and predictions
        """
        print("\n🤖 Generating predictions for entire dataset...")
        
        # Tokenize all data
        all_encodings = self._tokenize_data(self.df['headline'].tolist())
        all_dataset = ClickbaitDataset(all_encodings, self.df['label'].tolist())
        
        # Make predictions
        all_predictions = self.trainer.predict(all_dataset)
        predicted_labels = np.argmax(all_predictions.predictions, axis=1)
        prediction_probs = torch.softmax(torch.tensor(all_predictions.predictions), dim=1)
        
        # Add predictions to dataframe
        result_df = self.df.copy()
        result_df['predicted_label'] = predicted_labels
        result_df['predicted_class'] = result_df['predicted_label'].map({
            0: 'Non-clickbait', 
            1: 'Clickbait'
        })
        result_df['confidence_score'] = prediction_probs.max(dim=1)[0].numpy()
        
        # Calculate accuracy on full dataset
        accuracy = accuracy_score(result_df['label'], result_df['predicted_label'])
        print(f"✅ Overall dataset accuracy: {accuracy:.4f}")
        
        return result_df
    
    def save_predictions(self, predictions_df: pd.DataFrame) -> str:
        """
        Save predictions to CSV file.
        
        Args:
            predictions_df: DataFrame with predictions
            
        Returns:
            Path to saved file
        """
        # Create output directory
        os.makedirs('evaluation_results', exist_ok=True)
        
        # Generate filename with timestamp
        current_date = datetime.now().strftime("%Y-%m-%d")
        output_file = f'evaluation_results/model_predictions_{current_date}.csv'
        
        # Save to CSV
        predictions_df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"💾 Predictions saved to: {output_file}")
        
        return output_file
    
    def generate_evaluation_summary(self, metrics: Dict[str, Any]) -> None:
        """
        Generate and save evaluation summary report.
        
        Args:
            metrics: Dictionary containing evaluation metrics
        """
        summary = {
            'evaluation_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'model_path': self.model_path,
            'dataset_path': self.data_path,
            'total_samples': len(self.df),
            'test_samples': len(self.y_test),
            'accuracy': metrics['accuracy'],
            'f1_score': metrics['f1_score'],
            'classification_report': metrics['classification_report']
        }
        
        # Save summary as text file
        os.makedirs('evaluation_results', exist_ok=True)
        summary_file = 'evaluation_results/evaluation_summary.txt'
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("DistilBERT Clickbait Model Evaluation Summary\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Evaluation Date: {summary['evaluation_date']}\n")
            f.write(f"Model Path: {summary['model_path']}\n")
            f.write(f"Dataset Path: {summary['dataset_path']}\n")
            f.write(f"Total Samples: {summary['total_samples']}\n")
            f.write(f"Test Samples: {summary['test_samples']}\n")
            f.write(f"Accuracy: {summary['accuracy']:.4f}\n")
            f.write(f"F1-Score: {summary['f1_score']:.4f}\n\n")
            f.write("Detailed Classification Report:\n")
            f.write("-" * 30 + "\n")
            
            # Write classification report details
            report = summary['classification_report']
            for class_name in ['Non-clickbait', 'Clickbait']:
                if class_name in report:
                    class_metrics = report[class_name]
                    f.write(f"\n{class_name}:\n")
                    f.write(f"  Precision: {class_metrics['precision']:.4f}\n")
                    f.write(f"  Recall: {class_metrics['recall']:.4f}\n")
                    f.write(f"  F1-Score: {class_metrics['f1-score']:.4f}\n")
                    f.write(f"  Support: {class_metrics['support']}\n")
        
        print(f"📋 Evaluation summary saved to: {summary_file}")
    
    def run_complete_evaluation(self) -> None:
        """Run complete evaluation pipeline."""
        print("🚀 Starting Model Evaluation Pipeline")
        print("=" * 50)
        
        # Evaluate model performance
        metrics = self.evaluate_model()
        
        # Generate predictions for all data
        predictions_df = self.predict_all_data()
        
        # Save results
        self.save_predictions(predictions_df)
        self.generate_evaluation_summary(metrics)
        
        print("\n✅ Evaluation completed successfully!")
        print("Check 'evaluation_results/' directory for detailed results.")


class ClickbaitDataset(torch.utils.data.Dataset):
    """PyTorch Dataset for clickbait classification."""
    
    def __init__(self, encodings: Dict[str, torch.Tensor], labels: List[int]):
        """
        Initialize dataset with encodings and labels.
        
        Args:
            encodings: Tokenized text encodings
            labels: List of integer labels
        """
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get item at specified index.
        
        Args:
            idx: Index to retrieve
            
        Returns:
            Dictionary with encodings and label
        """
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.labels)


def main():
    """Main execution function."""
    try:
        # Initialize evaluator
        evaluator = ModelEvaluator()
        
        # Run complete evaluation
        evaluator.run_complete_evaluation()
        
    except KeyboardInterrupt:
        print("\n⚠️ Evaluation interrupted by user")
    except Exception as e:
        print(f"❌ Unexpected error during evaluation: {e}")
        raise


if __name__ == "__main__":
    main()
