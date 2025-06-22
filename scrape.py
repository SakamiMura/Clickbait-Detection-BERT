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

# Pobranie zasobów NLTK
nltk.download('punkt')
nltk.download('stopwords')

# Funkcja do czyszczenia tekstu
def clean_text(text):
    """
    Czyści tekst, usuwając cyfry, znaki specjalne i nadmiarowe spacje.
    Zamienia tekst na małe litery.
    """
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Funkcja do tokenizacji i usuwania stop words
def process_text(text):
    """
    Tokenizuje tekst i usuwa angielskie stop words.
    """
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token.lower() not in stop_words]
    return ' '.join(tokens)

# Funkcja do usuwania duplikatów
def remove_duplicates(headlines):
    """
    Usuwa duplikaty na podstawie oryginalnego nagłówka.
    """
    seen = set()
    unique_headlines = []
    for headline in headlines:
        if headline["headline"] not in seen:
            seen.add(headline["headline"])
            unique_headlines.append(headline)
    return unique_headlines

# Funkcja do scrapowania nagłówków
def scrape_headlines(url, selectors=None):
    """
    Scrapuje nagłówki z podanego URL za pomocą Selenium i BeautifulSoup.
    Args:
        url: Adres strony
        selectors: Selektory CSS do wyszukiwania nagłówków
    Returns:
        list: Lista słowników z nagłówkami
        str: Nazwa portalu
    """
    portal_mapping = {
        'https://www.buzzfeed.com/': 'buzzfeed',
        'https://www.nature.com/': 'nature',
        'https://www.newscientist.com/' : 'newscientist',
        'https://www.dailymail.co.uk/home/index.html/': 'dailymail',
        'https://phys.org/': 'physorg',
    }
    portal_name = portal_mapping.get(url, url.split('//')[-1].split('.')[1])

    if selectors is None:
        if 'buzzfeed' in url:
            selectors = '[class*="title"], [class*="headline"], h2, h3'
        elif 'newscientist' in url:
            selectors = 'h3, h2, [class*="title"], [class*="headline"], .article-title'
        elif 'dailymail' in url:
            selectors = 'h2, h3, [class*="headline"], [class*="link-text"], .linkro-darkred'
        elif 'phys.org' in url:
            selectors = 'h2.text-m, h1.text-l, [class*="title"], .news-title'
        else:  # nature and default
            selectors = 'h3.c-card__title, h2.c-card__title, [class*="title"]'

    chrome_options = Options()
    chrome_options.add_argument("--headless=new")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument("--disable-notifications")
    chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")

    driver = webdriver.Chrome(options=chrome_options)
    print(f"Scrapowanie {portal_name.upper()}...")

    try:
        driver.get(url)
        time.sleep(5)
        # Przewiń stronę dla dynamicznych treści
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)
        html_content = driver.page_source
        soup = BeautifulSoup(html_content, 'html.parser')
        title_elements = soup.select(selectors)

        headlines = []
        exclude_keywords = ['newsletter', 'signup', 'loading', 'success', 'privacy', 'consent', 'email', 'manage', 'preferences', 'cookie', 'advertising']
        for element in title_elements:
            text = element.text.strip()
            if text and 15 < len(text) < 200:
                cleaned_text = clean_text(text)
                processed_text = process_text(cleaned_text)
                if (processed_text and
                    len(processed_text.split()) >= 5 and
                    not any(kw in text.lower() for kw in exclude_keywords) and
                    not re.match(r'.*\•.*(ago|hours|minutes)', text, re.IGNORECASE)):
                    headlines.append({
                        "site": portal_name,
                        "headline": text,
                        "processed_headline": processed_text
                    })

        unique_headlines = remove_duplicates(headlines)
        print(f"Zescrapowano {len(unique_headlines)} unikalnych nagłówków z {portal_name.upper()}")
        return unique_headlines, portal_name

    except Exception as e:
        print(f"Błąd podczas scrapowania {url}: {str(e)}")
        return [], portal_name

    finally:
        driver.quit()

# Wczytaj model i tokenizer
model_path = './clickbait_model'
try:
    tokenizer = DistilBertTokenizer.from_pretrained(model_path)
    model = DistilBertForSequenceClassification.from_pretrained(model_path)
except Exception as e:
    print(f"Błąd wczytywania modelu: {e}. Upewnij się, że folder 'clickbait_model' istnieje.")
    exit()

# Funkcja do tokenizacji dla modelu
def tokenize_data(texts):
    return tokenizer(texts, padding=True, truncation=True, max_length=64, return_tensors='pt')

# Klasa datasetu dla PyTorch
class HeadlineDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        return item

    def __len__(self):
        return len(self.encodings.input_ids)

# Ustaw parametry dla Trainer
training_args = TrainingArguments(
    output_dir='./results',
    per_device_eval_batch_size=8,
    logging_dir='./logs',
)

# Stwórz Trainer
trainer = Trainer(
    model=model,
    args=training_args,
)

# Główna funkcja
def main():
    # Lista portali do scrapowania
    portals = [
        {"url": "https://www.buzzfeed.com/", "selectors": None},
        {"url": "https://www.nature.com/", "selectors": None},
        {"url": "https://www.newscientist.com/", "selectors": None},
        {"url": "https://www.dailymail.co.uk/home/index.html", "selectors": None},  # Removed trailing slash
        {"url": "https://phys.org/", "selectors": None},
    ]

    all_headlines = []
    for portal in portals:
        headlines, portal_name = scrape_headlines(portal["url"], portal["selectors"])
        all_headlines.extend(headlines)
        time.sleep(2)

    if not all_headlines:
        print("Nie zescrapowano żadnych nagłówków. Sprawdź połączenie, strony lub selektory.")
        return

    # Przygotuj dane do klasyfikacji
    headlines_df = pd.DataFrame(all_headlines)
    headlines = headlines_df['headline'].tolist()
    encodings = tokenize_data(headlines)
    dataset = HeadlineDataset(encodings)

    # Przewiduj etykiety
    predictions = trainer.predict(dataset)
    y_pred = np.argmax(predictions.predictions, axis=1)
    headlines_df['prediction'] = y_pred
    headlines_df['prediction_label'] = headlines_df['prediction'].map({0: 'Nie-clickbait', 1: 'Clickbait'})

    # Zapisz wyniki do CSV
    results_dir = "scraped_results"
    os.makedirs(results_dir, exist_ok=True)
    current_date = datetime.now().strftime("%Y-%m-%d")
    output_file = os.path.join(results_dir, f"scraped_headlines_predictions_{current_date}.csv")
    headlines_df.to_csv(output_file, index=False, encoding='utf-8-sig', quoting=csv.QUOTE_MINIMAL)
    print(f"Zapisano wyniki do pliku: {output_file}")

if __name__ == "__main__":
    main()