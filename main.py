# =============================
# Mood-Aware Chatbot Project - training script
# =============================
# Trains emotion classifiers on sentence-transformer embeddings,
# evaluates them on a held-out test set, and saves the models for app.py.
#
# Run:  python main.py
# Output files: label_encoder.pkl, mood_model_semantic.pkl (Logistic Regression),
#               mood_model_svm.pkl (Linear SVM), mood_model_best.pkl (better of the two),
#               class_distribution.png, confusion_matrix.png

import os
import re

import joblib
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC

# =============================
# Settings
# =============================
DATA_FILE = "combined_emotion.csv"
SAMPLE_PER_CLASS = None      # None = use ALL balanced data (about 15K per emotion); or set e.g. 1500 for a quick test
RANDOM_STATE = 42
SHOW_PLOTS = False           # True = open chart windows (the script waits until you close them)
EMBEDDINGS_FILE = f"X_semantic_{SAMPLE_PER_CLASS or 'all'}.npy"  # cache; delete it if you change the data or preprocessing

# =============================
# NLTK resources
# =============================
for resource in ["punkt", "punkt_tab", "stopwords", "wordnet"]:
    nltk.download(resource, quiet=True)

# =============================
# Load data
# =============================
df = pd.read_csv(DATA_FILE)

print("First 5 rows:\n", df.head())
print("\nMissing values:\n", df.isnull().sum())
print("\nClass distribution (raw):\n", df["emotion"].value_counts())

# Clean the data
df = df.dropna()
df = df[df["emotion"] != "empty"]
df = df.drop_duplicates(subset=["sentence", "emotion"])

# Fix the spelling used in the dataset's labels ("suprise" -> "surprise")
df["emotion"] = df["emotion"].replace({"suprise": "surprise"})

original_counts = df["emotion"].value_counts()
print("\nClass distribution (after cleaning):\n", original_counts)
print("Total sentences after cleaning:", len(df))

# =============================
# Balance classes by undersampling
# =============================
min_count = original_counts.min()
balanced_df = (
    df.groupby("emotion", group_keys=False)
    .sample(n=min_count, random_state=RANDOM_STATE)
)
balanced_counts = balanced_df["emotion"].value_counts()
print("\nBalanced counts:\n", balanced_counts)

# Chart: class distribution before and after balancing
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
original_counts.plot(kind="bar", color="skyblue")
plt.title("Before Balancing")
plt.xlabel("Emotion")
plt.ylabel("Count")

plt.subplot(1, 2, 2)
balanced_counts.plot(kind="bar", color="lightgreen")
plt.title("After Balancing")
plt.xlabel("Emotion")
plt.ylabel("Count")

plt.suptitle("Class Distribution Before and After Balancing")
plt.tight_layout()
plt.savefig("class_distribution.png", dpi=150)
if SHOW_PLOTS:
    plt.show()
plt.close()

# =============================
# Choose the training data (all balanced data, or a smaller sample)
# =============================
samples_per_class = min(SAMPLE_PER_CLASS, min_count) if SAMPLE_PER_CLASS else min_count
df = (
    balanced_df.groupby("emotion", group_keys=False)
    .sample(n=samples_per_class, random_state=RANDOM_STATE)
    .sample(frac=1, random_state=RANDOM_STATE)   # shuffle
    .reset_index(drop=True)
)
print("\nData used for training and testing:\n", df["emotion"].value_counts())
print("Total:", len(df), "sentences")

# =============================
# Text preprocessing
# (app.py must use EXACTLY the same steps)
# =============================
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))
stop_words.remove("not")  # keep "not" for negation


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


# Preprocess only the sample (much faster than preprocessing the whole dataset)
df["clean_text"] = df["sentence"].apply(preprocess_text)
df = df[df["clean_text"].str.strip() != ""].reset_index(drop=True)  # drop sentences that became empty

# =============================
# Sentence-Transformer embeddings
# =============================
# Each sentence becomes a dense vector that represents its meaning.
semantic_model = SentenceTransformer("all-MiniLM-L6-v2")

X_semantic = None
if os.path.exists(EMBEDDINGS_FILE):
    X_semantic = np.load(EMBEDDINGS_FILE)
    if len(X_semantic) == len(df):
        print("\nLoaded saved embeddings from", EMBEDDINGS_FILE)
    else:
        print("\nSaved embeddings do not match the data. Encoding again...")
        X_semantic = None

if X_semantic is None:
    print("\nEncoding text (first run only). With all the data this takes about 10-15 minutes...")
    X_semantic = semantic_model.encode(
        df["clean_text"].tolist(),
        batch_size=32,
        show_progress_bar=True,
        convert_to_tensor=False,
    )
    np.save(EMBEDDINGS_FILE, X_semantic)
    print("Embeddings saved to", EMBEDDINGS_FILE)

# =============================
# Encode labels
# =============================
le = LabelEncoder()
y = le.fit_transform(df["emotion"])
joblib.dump(le, "label_encoder.pkl")
print("\nClasses:", list(le.classes_), f"({len(le.classes_)} emotions)")

# =============================
# Train / test split
# =============================
X_train, X_test, y_train, y_test = train_test_split(
    X_semantic, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print(f"Training examples: {len(X_train)} | Test examples: {len(X_test)}")

# =============================
# Train models
# =============================
lr_model = LogisticRegression(max_iter=2000, class_weight="balanced")
# LinearSVC is a linear SVM that trains much faster than SVC(kernel="linear") on large data
svm_model = LinearSVC(class_weight="balanced", max_iter=5000, random_state=RANDOM_STATE)

print("\nTraining Logistic Regression...")
lr_model.fit(X_train, y_train)
print("Training Linear SVM...")
svm_model.fit(X_train, y_train)

# =============================
# Evaluate on the held-out test set
# =============================
results = {}
for name, model in [("Logistic Regression", lr_model), ("Linear SVM", svm_model)]:
    predictions = model.predict(X_test)
    results[name] = {
        "model": model,
        "predictions": predictions,
        "accuracy": accuracy_score(y_test, predictions),
        "macro_f1": f1_score(y_test, predictions, average="macro"),
    }

print("\n=============================")
print("Evaluation (held-out test set)")
print("=============================")
for name, r in results.items():
    print(f"{name}: accuracy = {r['accuracy']:.4f} | macro F1 = {r['macro_f1']:.4f}")

best_name = max(results, key=lambda n: results[n]["accuracy"])
best = results[best_name]
best_model = best["model"]
print(f"\nBest model: {best_name}")
print("\nClassification report:\n")
print(classification_report(y_test, best["predictions"], target_names=le.classes_))

# Confusion matrix of the best model
ConfusionMatrixDisplay.from_predictions(
    y_test,
    best["predictions"],
    display_labels=le.classes_,
    xticks_rotation=45,
    cmap="Blues",
)
plt.title(f"Confusion Matrix - {best_name}")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
if SHOW_PLOTS:
    plt.show()
plt.close()

# =============================
# Save models
# =============================
joblib.dump(lr_model, "mood_model_semantic.pkl")   # Logistic Regression (loaded by app.py by default)
joblib.dump(svm_model, "mood_model_svm.pkl")       # Linear SVM
joblib.dump(best_model, "mood_model_best.pkl")     # whichever scored higher
print("\nModels saved: mood_model_semantic.pkl, mood_model_svm.pkl, mood_model_best.pkl")

# =============================
# Command-line demo
# =============================
print(f"\nDemo using: {best_name}")
while True:
    user_input = input("Type a sentence (or 'quit' to exit): ")
    if user_input.strip().lower() == "quit":
        break
    if not user_input.strip():
        print("Please type a sentence.")
        continue

    processed_input = preprocess_text(user_input)
    if not processed_input:
        print("Could not find meaningful words in that sentence. Try another one.")
        continue

    embedding = semantic_model.encode([processed_input], convert_to_tensor=False)
    prediction = best_model.predict(embedding)[0]
    print("Detected mood:", le.inverse_transform([prediction])[0])