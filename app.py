# Step 1: Imports
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st
import os

# -----------------------------
# Safe Cached Model Loader
# -----------------------------
@st.cache_resource
def load_my_model():
    return load_model("rnn.keras", compile=False)

# Load model safely
model = load_my_model()

# -----------------------------
# Load IMDB Word Index
# -----------------------------
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# -----------------------------
# Helper Functions
# -----------------------------
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("IMDB Movie Review Sentiment Analysis")
st.write("Enter a movie review to classify it as positive or negative.")

user_input = st.text_area("Movie Review")

if st.button("Classify") and user_input.strip() != "":
    with st.spinner("Analyzing..."):
        preprocessed_input = preprocess_text(user_input)
        prediction = model.predict(preprocessed_input)
        score = float(prediction[0][0])

        sentiment = "Positive" if score > 0.5 else "Negative"

    st.success(f"Sentiment: {sentiment}")
    st.write(f"Prediction Score: {score:.4f}")
