# Clickbait Analysis System 🎯

A comprehensive machine learning system for detecting clickbait headlines using DistilBERT transformer model. The system scrapes headlines from major news portals and classifies them in real-time.

## 📋 Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Project Workflow](#project-workflow)
- [File Structure](#file-structure)
- [Installation](#installation)
- [Usage Guide](#usage-guide)
- [Model Performance](#model-performance)
- [Supported News Portals](#supported-news-portals)
- [Technical Details](#technical-details)
- [Contributing](#contributing)

## 🔍 Overview

This project implements a three-phase pipeline for clickbait detection:

1. **Training Phase**: Train a DistilBERT model on labeled clickbait dataset
2. **Scraping & Classification Phase**: Real-time headline scraping and classification  
3. **Evaluation Phase**: Comprehensive model performance analysis

The system achieves high accuracy in distinguishing between clickbait and legitimate news headlines across multiple news sources.

## 🏗️ System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   TRAINING      │    │    SCRAPING     │    │   EVALUATION    │
│                 │    │                 │    │                 │
│ model_trainer.py│───▶│headlines_scraper│───▶│model_evaluator  │
│                 │    │      .py        │    │      .py        │
│ ┌─────────────┐ │    │ ┌─────────────┐ │    │ ┌─────────────┐ │
│ │ClickbaitData│ │    │ │  Selenium   │ │    │ │Metrics &    │ │
│ │   Dataset   │ │    │ │BeautifulSoup│ │    │ │Reports      │ │
│ └─────────────┘ │    │ └─────────────┘ │    │ └─────────────┘ │
└─────────────────┘    └─────────────────┘    └─────────────────┘
        │                        │                        │
        ▼                        ▼                        ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Trained Model   │    │ Classified      │    │ Performance     │
│ (DistilBERT)    │    │ Headlines       │    │ Reports         │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🔄 Project Workflow

### Phase 1: Model Training (`model_trainer.py`)

**Purpose**: Train a DistilBERT model for binary clickbait classification

**Input**: 
- `clickbait_data.csv` - Pre-labeled dataset with headlines and binary labels (0=non-clickbait, 1=clickbait)
- **Note**: The training uses an existing labeled dataset, NOT the scraped headlines

**Process**:
1. Load and validate labeled training data
2. Split data into train/validation/test sets (70%/15%/15%)
3. Tokenize headlines using DistilBERT tokenizer
4. Fine-tune pre-trained DistilBERT model
5. Apply early stopping and model checkpointing
6. Save trained model to `./clickbait_model/`

**Output**: 
- Trained DistilBERT model files
- Training metrics and visualizations
- Model configuration files

### Phase 2: Real-time Scraping & Classification (`headlines_scraper.py`)

**Purpose**: Scrape fresh headlines and classify them using the trained model

**Process**:
1. **Web Scraping**: Use Selenium + BeautifulSoup to extract headlines
2. **Text Preprocessing**: Clean and normalize headline text
3. **Real-time Classification**: Apply trained model to new headlines
4. **Results Export**: Save classified headlines with confidence scores

**Supported Portals**:
- **BuzzFeed** (`buzzfeed.com`) - Known for clickbait content
- **Nature** (`nature.com`) - Scientific journal articles  
- **New Scientist** (`newscientist.com`) - Science news
- **Daily Mail** (`dailymail.co.uk`) - Tabloid newspaper
- **Phys.org** (`phys.org`) - Physics and science news

**Output**: 
- `scraped_results/scraped_headlines_predictions_YYYY-MM-DD.csv`
- Real-time classification results with confidence scores

### Phase 3: Model Evaluation (`model_evaluator.py`)

**Purpose**: Comprehensive analysis of model performance on test data

**Process**:
1. Load trained model and test dataset
2. Generate predictions on held-out test set
3. Calculate performance metrics (accuracy, F1-score, precision, recall)
4. Create confusion matrix visualization
5. Generate detailed classification reports
6. Export evaluation results

**Output**:
- Performance metrics and classification reports
- Confusion matrix visualization
- Detailed evaluation summary
- `evaluation_results/model_predictions_YYYY-MM-DD.csv`

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- Chrome browser (for Selenium WebDriver)
- 8GB+ RAM recommended for model training

### Setup Instructions

1. **Clone the repository**:
```bash
git clone <repository-url>
cd CLICKBAIT-ANALYSYS
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Download NLTK data**:
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

4. **Prepare training data**:
   - Ensure `clickbait_data.csv` exists with columns: `headline`, `label`
   - Labels should be binary: 0 (non-clickbait), 1 (clickbait)

## 📖 Usage Guide

### 1. Train the Model (First Time Setup)

```bash
python model_trainer.py
```

**Expected Output**:
- Training progress with loss metrics
- Validation scores during training
- Saved model in `./clickbait_model/`
- Training visualization plots

### 2. Scrape and Classify Headlines

```bash
python headlines_scraper.py
```

**Features**:
- Scrapes headlines from 5 major news portals
- Real-time clickbait classification
- Automatic text preprocessing
- Duplicate removal and filtering
- Results saved with timestamps

**Sample Output**:
```
🔍 Scraping BUZZFEED...
✅ Scraped 45 unique headlines from BUZZFEED
🔍 Scraping NATURE...
✅ Scraped 32 unique headlines from NATURE
🤖 Classifying headlines...
💾 Results saved to: scraped_results/scraped_headlines_predictions_2025-01-15.csv
```

### 3. Evaluate Model Performance

```bash
python model_evaluator.py
```

**Features**:
- Comprehensive performance metrics
- Confusion matrix visualization
- Per-class precision/recall analysis  
- Confidence score distribution
- Detailed evaluation reports

## 📊 Model Performance

### Training Configuration
- **Base Model**: DistilBERT (distilbert-base-uncased)
- **Task**: Binary sequence classification
- **Max Sequence Length**: 128 tokens
- **Batch Size**: 16
- **Learning Rate**: 2e-5
- **Training Epochs**: 3-5 (with early stopping)

### Expected Performance Metrics
- **Accuracy**: ~92-95%
- **F1-Score**: ~0.93-0.96
- **Precision**: ~0.91-0.94
- **Recall**: ~0.92-0.95

### Performance by Portal
| Portal | Clickbait Rate | Model Confidence |
|--------|---------------|------------------|
| BuzzFeed | ~85% | High (0.9+) |
| Daily Mail | ~60% | Medium (0.7+) |
| Nature | ~5% | High (0.9+) |
| New Scientist | ~15% | High (0.8+) |
| Phys.org | ~10% | High (0.8+) |

## 🌐 Supported News Portals

### High Clickbait Tendency
- **BuzzFeed**: Entertainment, lifestyle, viral content
- **Daily Mail**: Tabloid news, sensationalized headlines

### Low Clickbait Tendency  
- **Nature**: Scientific journal, peer-reviewed articles
- **Phys.org**: Physics and science news
- **New Scientist**: Science journalism, research news

### Portal-Specific Features
- **Adaptive CSS selectors** for different site structures
- **Rate limiting** to respect robots.txt
- **Error handling** for dynamic content loading
- **Text normalization** for consistent processing

## 🔧 Technical Details

### Text Preprocessing Pipeline
1. **HTML tag removal** and special character cleaning
2. **Lowercase normalization** and whitespace cleanup
3. **Stop word removal** using NLTK English stopwords
4. **Tokenization** with DistilBERT tokenizer
5. **Sequence padding/truncation** to fixed length

### Model Architecture
- **Transformer**: DistilBERT (6 layers, 768 hidden size)
- **Classification Head**: Linear layer with 2 outputs
- **Activation**: Softmax for probability distribution
- **Loss Function**: CrossEntropyLoss
- **Optimizer**: AdamW with weight decay

### Scraping Technology
- **Selenium WebDriver**: JavaScript rendering and dynamic content
- **BeautifulSoup**: HTML parsing and element extraction
- **CSS Selectors**: Portal-specific headline identification
- **Rate Limiting**: Respectful scraping with delays
