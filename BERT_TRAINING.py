import pandas as pd
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np

# 1. Wczytaj dane
df = pd.read_csv('clickbait_data.csv', encoding='utf-8')
df = df.dropna(subset=['headline', 'label'])
print(f"Wczytano {len(df)} wierszy")

# 2. Przygotuj dane
X = df['headline'].tolist()
y = df['label'].astype(int).tolist()

# 3. Podziel dane na treningowe (80%) i testowe (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Zbiór treningowy: {len(X_train)} wierszy")
print(f"Zbiór testowy: {len(X_test)} wierszy")

# 4. Wczytaj tokenizer i model DistilBERT
model_name = "distilbert-base-uncased"
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=2)

# 5. Tokenizuj dane
def tokenize_data(texts):
    return tokenizer(texts, padding=True, truncation=True, max_length=64, return_tensors='pt')

train_encodings = tokenize_data(X_train)
test_encodings = tokenize_data(X_test)

# 6. Przygotuj dataset dla PyTorch
class ClickbaitDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = ClickbaitDataset(train_encodings, y_train)
test_dataset = ClickbaitDataset(test_encodings, y_test)

# 7. Ustaw parametry trenowania
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

# 8. Trenowanie modelu
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

trainer.train()

# 9. Oceń model na danych testowych
predictions = trainer.predict(test_dataset)
y_pred = np.argmax(predictions.predictions, axis=1)
print("\nWyniki modelu na zbiorze testowym:")
print(classification_report(y_test, y_pred, target_names=['Nie-clickbait', 'Clickbait']))

# 10. Przewidź etykiety dla wszystkich danych
all_encodings = tokenize_data(X)
all_dataset = ClickbaitDataset(all_encodings, y)
all_predictions = trainer.predict(all_dataset)
df['predicted_clickbait'] = np.argmax(all_predictions.predictions, axis=1)

# 11. Zapisz wyniki
output_file = 'predicted_clickbait_2025-05-04.csv'
df.to_csv(output_file, index=False, encoding='utf-8')
print(f"Zapisano wyniki do pliku: {output_file}")

# 12. Zapisz model i tokenizer
model.save_pretrained('./clickbait_model')
tokenizer.save_pretrained('./clickbait_model')
print("Zapisano model i tokenizer do folderu: clickbait_model")