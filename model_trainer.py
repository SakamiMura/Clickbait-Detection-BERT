"""
DistilBERT Clickbait Model Trainer
=================================

This module trains a DistilBERT model for clickbait classification from scratch.
It handles data loading, preprocessing, model training, validation, and saving
the trained model for future use.

Features:
- Data preprocessing and validation
- Model training with customizable hyperparameters
- Training progress monitoring
- Model evaluation and validation
- Automatic model saving
- Training metrics visualization

Author: Your Name
Date: 2025
"""

import pandas as pd
import torch
from transformers import (
    DistilBertTokenizer, 
    DistilBertForSequenceClassification, 
    Trainer, 
    TrainingArguments,
    EarlyStoppingCallback
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import json
import matplotlib.pyplot as plt
import seaborn as sns


class DataPreprocessor:
    """Handles data loading and preprocessing for model training."""
    
    def __init__(self, data_path: str = 'clickbait_data.csv'):
        """
        Initialize data preprocessor.
        
        Args:
            data_path: Path to the labeled dataset CSV file
        """
        self.data_path = data_path
        self.df = None
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
    
    def load_and_validate_data(self) -> pd.DataFrame:
        """
        Load dataset and perform basic validation.
        
        Returns:
            Validated DataFrame
            
        Raises:
            FileNotFoundError: If dataset file doesn't exist
            ValueError: If required columns are missing
        """
        try:
            self.df = pd.read_csv(self.data_path, encoding='utf-8')
            print(f"✅ Loaded dataset with {len(self.df)} rows")
            
            # Validate required columns
            required_columns = ['headline', 'label']
            missing_columns = [col for col in required_columns if col not in self.df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Remove rows with missing values
            initial_size = len(self.df)
            self.df = self.df.dropna(subset=required_columns)
            removed_rows = initial_size - len(self.df)
            if removed_rows > 0:
                print(f"⚠️ Removed {removed_rows} rows with missing values")
            
            # Validate label values
            unique_labels = self.df['label'].unique()
            if not all(label in [0, 1] for label in unique_labels):
                raise ValueError("Labels must be binary (0 or 1)")
            
            # Print dataset statistics
            self._print_dataset_stats()
            
            return self.df
            
        except FileNotFoundError:
            print(f"❌ Dataset file not found: {self.data_path}")
            print("Please ensure clickbait_data.csv exists in the project directory")
            raise
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            raise
    
    def _print_dataset_stats(self) -> None:
        """Print basic dataset statistics."""
        print("\n📊 Dataset Statistics:")
        print("-" * 30)
        print(f"Total samples: {len(self.df)}")
        print(f"Clickbait samples: {len(self.df[self.df['label'] == 1])}")
        print(f"Non-clickbait samples: {len(self.df[self.df['label'] == 0])}")
        
        # Calculate class distribution
        class_dist = self.df['label'].value_counts(normalize=True)
        print(f"Class distribution:")
        print(f"  Non-clickbait: {class_dist[0]:.2%}")
        print(f"  Clickbait: {class_dist[1]:.2%}")
        
        # Check for class imbalance
        if abs(class_dist[0] - class_dist[1]) > 0.3:
            print("⚠️ Significant class imbalance detected - consider using weighted loss")
    
    def split_data(self, test_size: float = 0.2, val_size: float = 0.1, random_state: int = 42) -> None:
        """
        Split data into train/validation/test sets.
        
        Args:
            test_size: Fraction of data for test set
            val_size: Fraction of data for validation set
            random_state: Random seed for reproducibility
        """
        if self.df is None:
            raise ValueError("Data must be loaded first using load_and_validate_data()")
        
        X = self.df['headline'].tolist()
        y = self.df['label'].astype(int).tolist()
        
        # First split: train+val vs test
        X_temp, self.X_test, y_temp, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Second split: train vs validation
        val_size_adjusted = val_size / (1 - test_size)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state, stratify=y_temp
        )
        
        print(f"\n📂 Data split completed:")
        print(f"  Training set: {len(self.X_train)} samples")
        print(f"  Validation set: {len(self.X_val)} samples")
        print(f"  Test set: {len(self.X_test)} samples")


class ModelTrainer:
    """Handles DistilBERT model training for clickbait classification."""
    
    def __init__(self, model_name: str = 'distilbert-base-uncased'):
        """
        Initialize model trainer.
        
        Args:
            model_name: Pre-trained model name from HuggingFace
        """
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.trainer = None
        self.training_history = {}
        
        # Initialize tokenizer and model
        self._initialize_model()
    
    def _initialize_model(self) -> None:
        """Initialize tokenizer and model."""
        try:
            print(f"🔧 Initializing {self.model_name}...")
            self.tokenizer = DistilBertTokenizer.from_pretrained(self.model_name)
            self.model = DistilBertForSequenceClassification.from_pretrained(
                self.model_name, 
                num_labels=2
            )
            print("✅ Model and tokenizer initialized successfully")
            
        except Exception as e:
            print(f"❌ Error initializing model: {e}")
            raise
    
    def tokenize_data(self, texts: List[str], max_length: int = 128) -> Dict[str, torch.Tensor]:
        """
        Tokenize text data for model input.
        
        Args:
            texts: List of text strings
            max_length: Maximum sequence length
            
        Returns:
            Dictionary of tokenized tensors
        """
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )
    
    def create_datasets(self, preprocessor: DataPreprocessor, max_length: int = 128) -> Tuple:
        """
        Create PyTorch datasets from preprocessed data.
        
        Args:
            preprocessor: Data preprocessor with split data
            max_length: Maximum sequence length for tokenization
            
        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        print("🔤 Tokenizing datasets...")
        
        # Tokenize all splits
        train_encodings = self.tokenize_data(preprocessor.X_train, max_length)
        val_encodings = self.tokenize_data(preprocessor.X_val, max_length)
        test_encodings = self.tokenize_data(preprocessor.X_test, max_length)
        
        # Create datasets
        train_dataset = ClickbaitDataset(train_encodings, preprocessor.y_train)
        val_dataset = ClickbaitDataset(val_encodings, preprocessor.y_val)
        test_dataset = ClickbaitDataset(test_encodings, preprocessor.y_test)
        
        print("✅ Datasets created successfully")
        return train_dataset, val_dataset, test_dataset
    
    def setup_training_args(self, 
                           output_dir: str = './training_results',
                           num_epochs: int = 3,
                           batch_size: int = 16,
                           learning_rate: float = 2e-5,
                           warmup_steps: int = 500,
                           weight_decay: float = 0.01,
                           early_stopping_patience: int = 2) -> TrainingArguments:
        """
        Setup training arguments with best practices.
        
        Args:
            output_dir: Output directory for model checkpoints
            num_epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate for optimizer
            warmup_steps: Number of warmup steps
            weight_decay: Weight decay for regularization
            early_stopping_patience: Early stopping patience
            
        Returns:
            TrainingArguments object
        """
        return TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=warmup_steps,
            weight_decay=weight_decay,
            learning_rate=learning_rate,
            logging_dir=f'{output_dir}/logs',
            logging_steps=100,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_f1",
            greater_is_better=True,
            remove_unused_columns=False,
            push_to_hub=False,
            report_to=None,  # Disable wandb/tensorboard
        )
    
    def compute_metrics(self, eval_pred) -> Dict[str, float]:
        """
        Compute evaluation metrics during training.
        
        Args:
            eval_pred: Predictions from model
            
        Returns:
            Dictionary of computed metrics
        """
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='weighted'
        )
        accuracy = accuracy_score(labels, predictions)
        
        return {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
    
    def train_model(self, 
                   train_dataset, 
                   val_dataset, 
                   training_args: TrainingArguments,
                   early_stopping_patience: int = 2) -> None:
        """
        Train the model with given datasets and arguments.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            training_args: Training configuration
            early_stopping_patience: Early stopping patience
        """
        print("🚀 Starting model training...")
        
        # Setup trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)]
        )
        
        # Train the model
        train_result = self.trainer.train()
        
        # Store training history
        self.training_history = {
            'train_runtime': train_result.metrics['train_runtime'],
            'train_samples_per_second': train_result.metrics['train_samples_per_second'],
            'train_loss': train_result.metrics['train_loss'],
            'total_training_steps': train_result.metrics.get('train_steps_per_second', 0)
        }
        
        print("✅ Training completed successfully!")
        print(f"📊 Training loss: {train_result.metrics['train_loss']:.4f}")
        print(f"⏱️ Training time: {train_result.metrics['train_runtime']:.2f} seconds")
    
    def evaluate_model(self, test_dataset) -> Dict[str, float]:
        """
        Evaluate trained model on test dataset.
        
        Args:
            test_dataset: Test dataset for evaluation
            
        Returns:
            Dictionary of evaluation metrics
        """
        print("📊 Evaluating model on test set...")
        
        eval_results = self.trainer.evaluate(test_dataset)
        
        print("✅ Evaluation completed!")
        print(f"Test Accuracy: {eval_results['eval_accuracy']:.4f}")
        print(f"Test F1-Score: {eval_results['eval_f1']:.4f}")
        print(f"Test Precision: {eval_results['eval_precision']:.4f}")
        print(f"Test Recall: {eval_results['eval_recall']:.4f}")
        
        return eval_results
    
    def save_model(self, save_path: str = './clickbait_model') -> None:
        """
        Save trained model and tokenizer.
        
        Args:
            save_path: Directory path to save model
        """
        print(f"💾 Saving model to {save_path}...")
        
        # Create directory if it doesn't exist
        os.makedirs(save_path, exist_ok=True)
        
        # Save model and tokenizer
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        
        # Save training configuration
        config = {
            'model_name': self.model_name,
            'training_history': self.training_history,
            'save_date': datetime.now().isoformat(),
            'num_labels': 2,
            'label_mapping': {0: 'Non-clickbait', 1: 'Clickbait'}
        }
        
        with open(os.path.join(save_path, 'training_config.json'), 'w') as f:
            json.dump(config, f, indent=2)
        
        print("✅ Model saved successfully!")
    
    def plot_training_history(self, save_path: Optional[str] = None) -> None:
        """
        Plot training history if available.
        
        Args:
            save_path: Path to save the plot (optional)
        """
        try:
            if hasattr(self.trainer, 'state') and self.trainer.state.log_history:
                log_history = self.trainer.state.log_history
                
                # Extract training and validation metrics
                train_losses = [log['train_loss'] for log in log_history if 'train_loss' in log]
                eval_losses = [log['eval_loss'] for log in log_history if 'eval_loss' in log]
                eval_f1 = [log['eval_f1'] for log in log_history if 'eval_f1' in log]
                
                if train_losses and eval_losses:
                    epochs = range(1, len(eval_losses) + 1)
                    
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
                    
                    # Plot loss
                    ax1.plot(epochs, eval_losses, 'b-', label='Validation Loss')
                    ax1.set_title('Training and Validation Loss')
                    ax1.set_xlabel('Epoch')
                    ax1.set_ylabel('Loss')
                    ax1.legend()
                    ax1.grid(True)
                    
                    # Plot F1 score
                    ax2.plot(epochs, eval_f1, 'r-', label='Validation F1')
                    ax2.set_title('Validation F1 Score')
                    ax2.set_xlabel('Epoch')
                    ax2.set_ylabel('F1 Score')
                    ax2.legend()
                    ax2.grid(True)
                    
                    plt.tight_layout()
                    
                    if save_path:
                        plt.savefig(save_path, dpi=300, bbox_inches='tight')
                        print(f"📊 Training plot saved to {save_path}")
                    
                    plt.show()
                    
        except ImportError:
            print("⚠️ Matplotlib not available - skipping training plot")
        except Exception as e:
            print(f"⚠️ Error creating training plot: {e}")


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
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.labels)


class ClickbaitModelTrainingPipeline:
    """Complete training pipeline for clickbait classification model."""
    
    def __init__(self, 
                 data_path: str = 'clickbait_data.csv',
                 model_name: str = 'distilbert-base-uncased'):
        """
        Initialize training pipeline.
        
        Args:
            data_path: Path to labeled dataset
            model_name: Pre-trained model name
        """
        self.data_path = data_path
        self.model_name = model_name
        self.preprocessor = DataPreprocessor(data_path)
        self.trainer_model = ModelTrainer(model_name)
    
    def run_training_pipeline(self,
                            num_epochs: int = 3,
                            batch_size: int = 16,
                            learning_rate: float = 2e-5,
                            max_length: int = 128,
                            early_stopping_patience: int = 2) -> None:
        """
        Run complete training pipeline.
        
        Args:
            num_epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate
            max_length: Maximum sequence length
            early_stopping_patience: Early stopping patience
        """
        print("🚀 Starting Clickbait Model Training Pipeline")
        print("=" * 60)
        
        try:
            # Step 1: Load and preprocess data
            print("\n📂 Step 1: Loading and preprocessing data...")
            self.preprocessor.load_and_validate_data()
            self.preprocessor.split_data()
            
            # Step 2: Create datasets
            print("\n🔤 Step 2: Creating datasets...")
            train_dataset, val_dataset, test_dataset = self.trainer_model.create_datasets(
                self.preprocessor, max_length
            )
            
            # Step 3: Setup training arguments
            print("\n⚙️ Step 3: Setting up training configuration...")
            training_args = self.trainer_model.setup_training_args(
                num_epochs=num_epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                early_stopping_patience=early_stopping_patience
            )
            
            # Step 4: Train model
            print("\n🏋️ Step 4: Training model...")
            self.trainer_model.train_model(
                train_dataset, 
                val_dataset, 
                training_args,
                early_stopping_patience
            )
            
            # Step 5: Evaluate model
            print("\n📊 Step 5: Evaluating model...")
            eval_results = self.trainer_model.evaluate_model(test_dataset)
            
            # Step 6: Save model
            print("\n💾 Step 6: Saving model...")
            self.trainer_model.save_model()
            
            # Step 7: Create training plot
            print("\n📈 Step 7: Creating training visualization...")
            self.trainer_model.plot_training_history('training_results/training_history.png')
            
            print("\n✅ Training pipeline completed successfully!")
            print("🎉 Your clickbait classification model is ready to use!")
            
        except Exception as e:
            print(f"❌ Training pipeline failed: {e}")
            raise


def main():
    """Main execution function."""
    try:
        # Initialize and run training pipeline
        pipeline = ClickbaitModelTrainingPipeline()
        
        # Run with custom parameters (adjust as needed)
        pipeline.run_training_pipeline(
            num_epochs=3,
            batch_size=16,
            learning_rate=2e-5,
            max_length=128,
            early_stopping_patience=2
        )
        
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
    except Exception as e:
        print(f"❌ Unexpected error during training: {e}")
        raise


if __name__ == "__main__":
    main()
