# =============================
# Mood-Aware Chatbot - Flask web app
# =============================
# Run main.py first (it creates the model files), then:  python app.py
# Open http://127.0.0.1:5000 in your browser.

import re

import joblib
import nltk
from flask import Flask, jsonify, render_template, request
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer

app = Flask(__name__)

# -----------------------------
# NLTK resources (same ones used in training)
# -----------------------------
for resource in ["punkt", "punkt_tab", "stopwords", "wordnet"]:
    nltk.download(resource, quiet=True)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))
stop_words.remove("not")  # MUST match training (keeps negation)

# -----------------------------
# Load the trained model, label encoder and embedding model
# -----------------------------
model = joblib.load("mood_model_best.pkl")
label_encoder = joblib.load("label_encoder.pkl")
semantic_model = SentenceTransformer("all-MiniLM-L6-v2")


# -----------------------------
# Text preprocessing (EXACT match to main.py)
# -----------------------------
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)  # remove URLs
    text = re.sub(r"[^a-zA-Z\s]", "", text)              # keep letters and spaces only
    words = word_tokenize(text)
    words = [
        lemmatizer.lemmatize(w)
        for w in words
        if w not in stop_words and w.isalpha()
    ]
    return " ".join(words)


# -----------------------------
# Routes
# -----------------------------
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    sentence = str(data.get("sentence", "")).strip()

    if not sentence:
        return jsonify({"error": "Please type a sentence."}), 400

    clean_text = preprocess_text(sentence)
    if not clean_text:
        return jsonify({"error": "Could not find meaningful words in that sentence."}), 400

    # Semantic embedding -> prediction
    embedding = semantic_model.encode([clean_text], convert_to_tensor=False)
    prediction = model.predict(embedding)[0]
    mood = label_encoder.inverse_transform([prediction])[0]

    return jsonify({"mood": mood})


# -----------------------------
# Run app
# -----------------------------
if __name__ == "__main__":
    # debug=True is for development only. use_reloader=False avoids loading the models twice.
    app.run(debug=True, use_reloader=False)