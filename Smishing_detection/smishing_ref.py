"""
smishing_ref.py
Main script to train two models:
1. Logistic Regression (TF-IDF)
2. LSTM (Embedding + LSTM)

Outputs:
- models/smishing_model.pkl
- models/smishing_lstm.h5
- utils/vectorizer.pkl
- utils/tokenizer.pkl
"""
import os
import pickle
import pandas as pd
from utils.text_cleaner import clean_text

# Logistic Regression Modules
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

# LSTM Modules
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np

# ======================================================
# 1. LOAD DATA
# ======================================================
df = pd.read_csv(
    "data/SMSSmishCollection.txt",
    sep="\t",
    header=None,
    names=["label", "message"],
    quoting=3,
    engine="python",
    on_bad_lines="skip"
)

print(df.head(5))
print(df.shape)

# ======================================================
# 2. CLEAN TEXT
# ======================================================
df["clean_text"] = df["message"].astype(str).apply(clean_text)
X = df["clean_text"].values

# For LR, keep string labels
y_lr = df["label"].values

# For LSTM, map labels to numeric
label_mapping = {"ham": 0, "smish": 1}
y_lstm = df["label"].map(label_mapping).values

# ======================================================
# 3. TRAIN-TEST SPLIT
# ======================================================
X_train_text, X_test_text, y_train_lr, y_test_lr, y_train_lstm, y_test_lstm = train_test_split(
    X, y_lr, y_lstm, test_size=0.2, stratify=y_lr, random_state=42
)

# ======================================================
# 4. LOGISTIC REGRESSION (TF-IDF)
# ======================================================
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))

logistic_pipeline = Pipeline([
    ("tfidf", tfidf),
    ("clf", LogisticRegression(solver="saga", max_iter=1000, class_weight="balanced"))
])

param_grid = {
    "clf__C": [0.1, 1.0],
    "clf__penalty": ["l2"]
}

print("Training Logistic Regression...")
grid = GridSearchCV(
    logistic_pipeline, param_grid,
    cv=3, scoring="accuracy", n_jobs=1
)
grid.fit(X_train_text, y_train_lr)

best_log_model = grid.best_estimator_

print("Best params:", grid.best_params_)

# Save TF-IDF vectorizer + logistic model
os.makedirs("models", exist_ok=True)
os.makedirs("utils", exist_ok=True)

with open("utils/vectorizer.pkl", "wb") as f:
    pickle.dump(best_log_model.named_steps["tfidf"], f)

with open("models/smishing_model.pkl", "wb") as f:
    pickle.dump(best_log_model, f)

# Evaluate LR model
y_pred = best_log_model.predict(X_test_text)
print("\n=== Logistic Regression Performance ===")
print("Accuracy:", accuracy_score(y_test_lr, y_pred))
print(classification_report(y_test_lr, y_pred))

# ======================================================
# 5. LSTM CLASSIFIER
# ======================================================
print("\nTraining LSTM model...")

tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train_text)

# Save tokenizer
with open("utils/tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

# Convert text → sequences
X_train_seq = tokenizer.texts_to_sequences(X_train_text)
X_test_seq = tokenizer.texts_to_sequences(X_test_text)

max_len = 60  # slightly longer to capture more text
X_train_pad = pad_sequences(X_train_seq, maxlen=max_len, padding="post")
X_test_pad = pad_sequences(X_test_seq, maxlen=max_len, padding="post")

vocab_size = min(5000, len(tokenizer.word_index) + 1)

# Build LSTM model
lstm_model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=64, input_length=max_len),
    LSTM(64),
    Dropout(0.3),
    Dense(32, activation="relu"),
    Dropout(0.3),
    Dense(1, activation="sigmoid")
])

lstm_model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
print(lstm_model.summary())

# EarlyStopping
es = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)

# Train
lstm_model.fit(
    X_train_pad, y_train_lstm,
    validation_split=0.1,
    epochs=10,
    batch_size=32,
    callbacks=[es]
)

# Evaluate
loss, acc = lstm_model.evaluate(X_test_pad, y_test_lstm, verbose=0)
print("\n=== LSTM Model Performance ===")
print("Accuracy:", acc)

# Save LSTM model
lstm_model.save("models/smishing_lstm.h5")
print("\nAll models saved successfully.")

# ======================================================
# 6. USER INPUT FOR PREDICTION
# ======================================================
# Load models
with open("models/smishing_model.pkl", "rb") as f:
    log_model = pickle.load(f)

with open("utils/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

lstm_model = load_model("models/smishing_lstm.h5")

def predict_message(msg):
    # Clean text
    clean_msg = clean_text(msg)
    
    # Logistic Regression Prediction
    lr_pred = log_model.predict([clean_msg])[0]
    
    # LSTM Prediction
    seq = tokenizer.texts_to_sequences([clean_msg])
    pad_seq = pad_sequences(seq, maxlen=max_len, padding="post")
    lstm_prob = lstm_model.predict(pad_seq)[0][0]
    lstm_pred = "smish" if lstm_prob >= 0.5 else "ham"
    
    # Combined: smish if either predicts smish
    combined = "smish" if (lr_pred == "smish" or lstm_pred == "smish") else "ham"
    
    print("\nPredictions:")
    print(f" - Logistic Regression: {lr_pred}")
    print(f" - LSTM: {lstm_pred}")
    print(f" - Combined Result: {combined}")

# Interactive loop
while True:
    user_input = input("\nEnter a message (or type 'exit' to quit): ")
    if user_input.lower() == "exit":
        break
    predict_message(user_input)
