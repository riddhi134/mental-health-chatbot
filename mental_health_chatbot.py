

# ==================== STEP 1: Install Libraries ====================
# !pip install transformers datasets torch scikit-learn

# ==================== STEP 2: Load & Preprocess ====================
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
import torch
from transformers import BertTokenizer

raw_datasets = load_dataset("daily_dialog", trust_remote_code=True)
train_data = raw_datasets["train"]

texts, labels = [], []
for dialog, emotions in zip(train_data["dialog"], train_data["emotion"]):
    for sentence, label in zip(dialog, emotions):
        if label != -1:
            texts.append(sentence)
            labels.append(label)

label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

class EmotionDataset(Dataset):
    def __init__(self, texts, labels):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=64)
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

X_train, X_val, y_train, y_val = train_test_split(texts, labels, test_size=0.2, random_state=42)
train_dataset = EmotionDataset(X_train, y_train)
val_dataset = EmotionDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

# ==================== STEP 3: Train the Model ====================
from transformers import BertForSequenceClassification, AdamW

model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(set(labels)))
optimizer = AdamW(model.parameters(), lr=2e-5)

model.train()
for epoch in range(3):
    print(f"Epoch {epoch+1}")
    for batch in train_loader:
        optimizer.zero_grad()
        outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['labels'])
        loss = outputs.loss
        loss.backward()
        optimizer.step()

model.save_pretrained("emotion_model")
tokenizer.save_pretrained("emotion_model")

# Save label encoder
import pickle
with open("emotion_model/label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

# ---------------------------------------------------------------

# File: mental_health_chatbot.py

import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification, AutoModelForCausalLM, AutoTokenizer
import pickle

st.set_page_config(page_title="Mental Health Chatbot")
st.title("🧠 Mental Health Chatbot")

if "chat_history_ids" not in st.session_state:
    st.session_state.chat_history_ids = None

# Load emotion model and label encoder
emotion_model = BertForSequenceClassification.from_pretrained("emotion_model")
emotion_tokenizer = BertTokenizer.from_pretrained("emotion_model")
with open("emotion_model/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# Load DialogGPT
dialog_model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")
dialog_tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")

def detect_emotion(text):
    inputs = emotion_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=64)
    with torch.no_grad():
        outputs = emotion_model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    label_id = torch.argmax(probs).item()
    return label_encoder.inverse_transform([label_id])[0]

def generate_response(user_input, chat_history_ids=None):
    input_ids = dialog_tokenizer.encode(user_input + dialog_tokenizer.eos_token, return_tensors='pt')
    if chat_history_ids is not None:
        input_ids = torch.cat([chat_history_ids, input_ids], dim=-1)
    chat_history_ids = dialog_model.generate(input_ids, max_length=1000, pad_token_id=dialog_tokenizer.eos_token_id)
    response = dialog_tokenizer.decode(chat_history_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
    return response, chat_history_ids

def get_recommendations(emotion):
    recs = {
        "joy": ["Watch a comedy movie", "Listen to upbeat music"],
        "sadness": ["Listen to soft music", "Read uplifting stories"],
        "anger": ["Try meditation", "Watch relaxing videos"],
        "fear": ["Watch confidence talks", "Practice breathing"],
        "love": ["Talk to someone you care about", "Watch feel-good content"]
    }
    return recs.get(emotion, ["Take a walk", "Do something creative"])

user_input = st.text_input("You:", "")
if user_input:
    emotion = detect_emotion(user_input)
    response, st.session_state.chat_history_ids = generate_response(user_input, st.session_state.chat_history_ids)
    st.markdown(f"**🤖 Bot:** {response}")
    st.markdown(f"**🧠 Detected Emotion:** `{emotion}`")

    with st.expander("💡 Recommendations"):
        for rec in get_recommendations(emotion):
            st.markdown(f"- {rec}")

# ------------------ requirements.txt ------------------
# streamlit
# transformers
# datasets
# torch
# scikit-learn
# ------------------------------------------------------

# Add README.md separately if you'd like help generating it.
