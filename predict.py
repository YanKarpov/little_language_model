from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tokenizer = AutoTokenizer.from_pretrained('results')  
model = AutoModelForSequenceClassification.from_pretrained('results')  

texts = [
    "I enjoy programming.",
    "I dislike bugs in my code.",
    "The new update works smoothly.",
    "This software is too slow.",
    "I love learning about artificial intelligence.",
    "The interface is really hard to understand.",
    "Customer support was very helpful.",
    "I hate how often this app crashes.",
    "The features in this app are very useful.",
    "I am frustrated with the constant bugs."
]

encodings = tokenizer(texts, truncation=True, padding=True, return_tensors='pt')

with torch.no_grad():
    outputs = model(**encodings)
    predictions = torch.argmax(outputs.logits, dim=-1)

for text, prediction in zip(texts, predictions):
    print(f"Text: {text} | Prediction: {prediction.item()}")