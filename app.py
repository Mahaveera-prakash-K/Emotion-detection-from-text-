import streamlit as st
import joblib
from src.preprocess import preprocess

# Load model and vectorizer
model = joblib.load("models/emotion_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

# App UI
st.set_page_config(page_title="Emotion Detector", page_icon="😊")
st.title("🧠 Emotion Detection from Text")
st.write("Enter a sentence and the model will predict the emotion.")

# Input box
user_input = st.text_area("✍️ Enter text here:")

# Predict button
if st.button("🔍 Detect Emotion"):
    if user_input.strip() == "":
        st.warning("Please enter some text!")
    else:
        cleaned = preprocess(user_input)
        vec = vectorizer.transform([cleaned])
        prediction = model.predict(vec)[0]
        st.success(f"💬 **Predicted Emotion:** `{prediction}`")
