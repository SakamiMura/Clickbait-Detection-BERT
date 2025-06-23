"""
Headlines Scraper with Clickbait Classification
==============================================

This module scrapes headlines from various news portals and classifies them
as clickbait or non-clickbait using a pre-trained DistilBERT model.

Supported portals:
- BuzzFeed
- Nature 
- New Scientist
- Daily Mail
- Phys.org

Author: Your Name
Date: 2025
"""

import pandas as pd
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import time
import os
from datetime import datetime
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import numpy as np
import csv
from typing import List, Dict, Tuple, Optional


class TextProcessor:
    """Handles text cleaning and preprocessing operations."""
    
    def __init__(self):
        """Initialize NLTK resources."""
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        self.stop_words = set(stopwords.words('english'))
    
    def clean_text(self, text: str) -> str:
        """
        Clean text by removing digits, special characters and extra spaces.
        
        Args:
            text: Raw text string
            
        Returns:
            Cleaned text in lowercase
        """
        text = text.lower()
        text = re.sub(r'[^a-z\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def process_text(self, text: str) -> str:
        """
        Tokenize text and remove English stop words.
        
        Args:
            text: Input text string
            
        Returns:
            Processed text without stop words
        """
        tokens = word_tokenize(text)
        tokens = [token for token in tokens if token.lower() not in self.stop_words]
        return ' '.join(tokens)


class HeadlinesScraper:
    """Web scraper for news headlines from various portals."""
    
    PORTAL_MAPPING = {
        'https://www.buzzfeed.com/': 'buzzfeed',
        'https://www.nature.com/': 'nature',
        'https://www.newscientist.com/': 'newscientist',
        'https://www.dailymail.co.uk/home/index.html/': 'dailymail',
        'https://phys.org/': 'physorg',
    }
    
    EXCLUDE_KEYWORDS = [
        'newsletter', 'signup', 'loading', 'success', 'privacy', 
        'consent', 'email', 'manage', 'preferences', 'cookie', 'advertising'
    ]
    
    def __init__(self, text_processor: TextProcessor):
        """
        Initialize scraper with text processor.
        
        Args:
            text_processor: Instance of TextProcessor for cleaning text
        """
        self.text_processor = text_processor
        self.chrome_options = self._setup_chrome_options()
    
    def _setup_chrome_options(self) -> Options:
        """Configure Chrome browser options for headless scraping."""
        options = Options()
        options.add_argument("--headless=new")
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        options.add_argument("--window-size=1920,1080")
        options.add_argument("--disable-notifications")
        options.add_argument(
            "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        )
        return options
    
    def _get_selectors(self, url: str) -> str:
        """
        Get appropriate CSS selectors for different portals.
        
        Args:
            url: Portal URL
            
        Returns:
            CSS selector string
        """
        if 'buzzfeed' in url:
            return '[class*="title"], [class*="headline"], h2, h3'
        elif 'newscientist' in url:
            return 'h3, h2, [class*="title"], [class*="headline"], .article-title'
        elif 'dailymail' in url:
            return 'h2, h3, [class*="headline"], [class*="link-text"], .linkro-darkred'
        elif 'phys.org' in url:
            return 'h2.text-m, h1.text-l, [class*="title"], .news-title'
        else:  # nature and default
            return 'h3.c-card__title, h2.c-card__title, [class*="title"]'
    
    def _is_valid_headline(self, text: str, processed_text: str) -> bool:
        """
        Validate if text is a proper headline.
        
        Args:
            text: Original text
            processed_text: Processed text
            
        Returns:
            True if valid headline, False otherwise
        """
        return (
            processed_text and
            15 < len(text) < 200 and
            len(processed_text.split()) >= 5 and
            not any(kw in text.lower() for kw in self.EXCLUDE_KEYWORDS) and
            not re.match(r'.*\•.*(ago|hours|minutes)', text, re.IGNORECASE)
        )
    
    def scrape_headlines(self, url: str, selectors: Optional[str] = None) -> Tuple[List[Dict], str]:
        """
        Scrape headlines from given URL using Selenium and BeautifulSoup.
        
        Args:
            url: Website URL to scrape
            selectors: Custom CSS selectors (optional)
            
        Returns:
            Tuple of (headlines list, portal name)
        """
        portal_name = self.PORTAL_MAPPING.get(url, url.split('//')[-1].split('.')[1])
        
        if selectors is None:
            selectors = self._get_selectors(url)
        
        driver = webdriver.Chrome(options=self.chrome_options)
        print(f"🔍 Scraping {portal_name.upper()}...")
        
        try:
            driver.get(url)
            time.sleep(5)
            
            # Scroll page for dynamic content
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)
            
            html_content = driver.page_source
            soup = BeautifulSoup(html_content, 'html.parser')
            title_elements = soup.select(selectors)
            
            headlines = []
            for element in title_elements:
                text = element.text.strip()
                if text:
                    cleaned_text = self.text_processor.clean_text(text)
                    processed_text = self.text_processor.process_text(cleaned_text)
                    
                    if self._is_valid_headline(text, processed_text):
                        headlines.append({
                            "site": portal_name,
                            "headline": text,
                            "processed_headline": processed_text
                        })
            
            unique_headlines = self._remove_duplicates(headlines)
            print(f"✅ Scraped {len(unique_headlines)} unique headlines from {portal_name.upper()}")
            return unique_headlines, portal_name
            
        except Exception as e:
            print(f"❌ Error scraping {url}: {str(e)}")
            return [], portal_name
        finally:
            driver.quit()
    
    def _remove_duplicates(self, headlines: List[Dict]) -> List[Dict]:
        """
        Remove duplicate headlines based on original headline text.
        
        Args:
            headlines: List of headline dictionaries
            
        Returns:
            List of unique headlines
        """
        seen = set()
        unique_headlines = []
        for headline in headlines:
            if headline["headline"] not in seen:
                seen.add(headline["headline"])
                unique_headlines.append(headline)
        return unique_headlines


class ClickbaitClassifier:
    """Clickbait classification using pre-trained DistilBERT model."""
    
    def __init__(self, model_path: str = './clickbait_model'):
        """
        Initialize classifier with pre-trained model.
        
        Args:
            model_path: Path to saved model directory
        """
        self.model_path = model_path
        self.tokenizer = None
        self.model = None
        self.trainer = None
        self._load_model()
    
    def _load_model(self):
        """Load tokenizer and model from saved directory."""
        try:
            self.tokenizer = DistilBertTokenizer.from_pretrained(self.model_path)
            self.model = DistilBertForSequenceClassification.from_pretrained(self.model_path)
            
            # Setup trainer for inference
            training_args = TrainingArguments(
                output_dir='./results',
                per_device_eval_batch_size=8,
                logging_dir='./logs',
            )
            
            self.trainer = Trainer(
                model=self.model,
                args=training_args,
            )
            print("✅ Model and tokenizer loaded successfully")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("Make sure 'clickbait_model' folder exists with trained model files.")
            exit()
    
    def tokenize_data(self, texts: List[str]) -> dict:
        """
        Tokenize text data for model input.
        
        Args:
            texts: List of text strings
            
        Returns:
            Tokenized data dictionary
        """
        return self.tokenizer(
            texts, 
            padding=True, 
            truncation=True, 
            max_length=64, 
            return_tensors='pt'
        )
    
    def predict(self, headlines_df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict clickbait labels for headlines.
        
        Args:
            headlines_df: DataFrame with headlines
            
        Returns:
            DataFrame with predictions added
        """
        headlines = headlines_df['headline'].tolist()
        encodings = self.tokenize_data(headlines)
        dataset = HeadlineDataset(encodings)
        
        # Make predictions
        predictions = self.trainer.predict(dataset)
        y_pred = np.argmax(predictions.predictions, axis=1)
        
        # Add predictions to dataframe
        headlines_df['prediction'] = y_pred
        headlines_df['prediction_label'] = headlines_df['prediction'].map({
            0: 'Nie-clickbait', 
            1: 'Clickbait'
        })
        
        return headlines_df


class HeadlineDataset(torch.utils.data.Dataset):
    """PyTorch Dataset for headline classification."""
    
    def __init__(self, encodings):
        """
        Initialize dataset with tokenized encodings.
        
        Args:
            encodings: Tokenized text encodings
        """
        self.encodings = encodings

    def __getitem__(self, idx):
        """Get item at index."""
        return {key: val[idx] for key, val in self.encodings.items()}

    def __len__(self):
        """Get dataset length."""
        return len(self.encodings.input_ids)


class HeadlinesAnalyzer:
    """Main class orchestrating the headline scraping and classification pipeline."""
    
    DEFAULT_PORTALS = [
        {"url": "https://www.buzzfeed.com/", "selectors": None},
        {"url": "https://www.nature.com/", "selectors": None},
        {"url": "https://www.newscientist.com/", "selectors": None},
        {"url": "https://www.dailymail.co.uk/home/index.html", "selectors": None},
        {"url": "https://phys.org/", "selectors": None},
    ]
    
    def __init__(self):
        """Initialize analyzer with required components."""
        self.text_processor = TextProcessor()
        self.scraper = HeadlinesScraper(self.text_processor)
        self.classifier = ClickbaitClassifier()
    
    def run_analysis(self, portals: Optional[List[Dict]] = None) -> None:
        """
        Run complete analysis pipeline: scrape headlines and classify them.
        
        Args:
            portals: List of portal configurations (optional)
        """
        if portals is None:
            portals = self.DEFAULT_PORTALS
        
        print("🚀 Starting Headlines Analysis Pipeline")
        print("=" * 50)
        
        # Scrape headlines from all portals
        all_headlines = []
        for portal in portals:
            headlines, portal_name = self.scraper.scrape_headlines(
                portal["url"], 
                portal["selectors"]
            )
            all_headlines.extend(headlines)
            time.sleep(2)  # Be respectful to servers
        
        if not all_headlines:
            print("❌ No headlines scraped. Check connection, websites or selectors.")
            return
        
        print(f"\n📊 Total headlines scraped: {len(all_headlines)}")
        
        # Classify headlines
        print("\n🤖 Classifying headlines...")
        headlines_df = pd.DataFrame(all_headlines)
        headlines_df = self.classifier.predict(headlines_df)
        
        # Save results
        self._save_results(headlines_df)
        self._print_summary(headlines_df)
    
    def _save_results(self, headlines_df: pd.DataFrame) -> None:
        """
        Save analysis results to CSV file.
        
        Args:
            headlines_df: DataFrame with analysis results
        """
        results_dir = "scraped_results"
        os.makedirs(results_dir, exist_ok=True)
        
        current_date = datetime.now().strftime("%Y-%m-%d")
        output_file = os.path.join(
            results_dir, 
            f"scraped_headlines_predictions_{current_date}.csv"
        )
        
        headlines_df.to_csv(
            output_file, 
            index=False, 
            encoding='utf-8-sig', 
            quoting=csv.QUOTE_MINIMAL
        )
        print(f"💾 Results saved to: {output_file}")
    
    def _print_summary(self, headlines_df: pd.DataFrame) -> None:
        """
        Print analysis summary statistics.
        
        Args:
            headlines_df: DataFrame with analysis results
        """
        print("\n📈 Analysis Summary:")
        print("-" * 30)
        
        total = len(headlines_df)
        clickbait_count = len(headlines_df[headlines_df['prediction'] == 1])
        non_clickbait_count = total - clickbait_count
        
        print(f"Total headlines analyzed: {total}")
        print(f"Clickbait: {clickbait_count} ({clickbait_count/total*100:.1f}%)")
        print(f"Non-clickbait: {non_clickbait_count} ({non_clickbait_count/total*100:.1f}%)")
        
        # Portal breakdown
        print("\nBreakdown by portal:")
        portal_stats = headlines_df.groupby('site')['prediction'].agg(['count', 'sum']).round(2)
        portal_stats.columns = ['Total', 'Clickbait']
        portal_stats['Clickbait_Rate'] = (portal_stats['Clickbait'] / portal_stats['Total'] * 100).round(1)
        print(portal_stats)


def main():
    """Main execution function."""
    try:
        analyzer = HeadlinesAnalyzer()
        analyzer.run_analysis()
    except KeyboardInterrupt:
        print("\n⚠️ Analysis interrupted by user")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")


if __name__ == "__main__":
    main()
