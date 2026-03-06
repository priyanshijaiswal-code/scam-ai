import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# -----------------------------
# Load Dataset
# -----------------------------
# Your dataset should have columns: text , label


# Load dataset
data = pd.read_csv("spam.csv", encoding="latin-1")

# Keep only needed columns
data = data[['v2','v1']]

# Rename columns
data.columns = ["text", "label"]

# Convert labels
data['label'] = data['label'].map({'ham':0,'spam':1})
# -----------------------------
# Features and Labels
# -----------------------------
X = data["text"]
y = data["label"]

# -----------------------------
# Train Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -----------------------------
# TF-IDF Vectorization
# -----------------------------
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=5000
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# -----------------------------
# Train Model
# -----------------------------
model = LogisticRegression()

model.fit(X_train_vec, y_train)

# -----------------------------
# Predictions
# -----------------------------
y_pred = model.predict(X_test_vec)

# -----------------------------
# Evaluation Metrics
# -----------------------------
print("\nModel Evaluation\n")

print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

# -----------------------------
# Save Model
# -----------------------------
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("\nModel and vectorizer saved successfully.")