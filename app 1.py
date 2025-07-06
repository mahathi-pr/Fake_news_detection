# app.py
import streamlit as st
import joblib
import re
from sentence_transformers import SentenceTransformer
# Load model
model = joblib.load("Fake news prediction final.pkl")
bert_model = SentenceTransformer("all-MiniLM-L6-v2")
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r"[^a-zA-Z ]", '', text)
    text = re.sub(r"\s+", ' ', text).strip()
    return text
def predict_news(text):
    cleaned = clean_text(text)
    emb = bert_model.encode([cleaned])
    prob = model.predict_proba(emb)[0]
    return "REAL" if prob[1] > 0.5 else "FAKE", prob
# Streamlit App Layout
st.set_page_config(page_title="Fake News Detector", layout="centered")
st.title("📰 Fake News Detection Web App")
user_input = st.text_area("Enter news headline or snippet:")
if st.button("Predict"):
    if not user_input.strip():
        st.warning("Please enter text!")
    else:
        label, prob = predict_news(user_input)
        st.success(f"🧠 Prediction: **{label}**")
        st.write(f"Confidence → REAL: `{prob[1]:.2f}`, FAKE: `{prob[0]:.2f}`")
