"""
Real-Time Multi-Layer AI Cyber Risk Intelligence Engine
Hybrid NLP + Structural + Behavioral Scam Detection System
Production-Stable Version
"""

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
import joblib
import re
import csv
import os
from datetime import datetime

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# -----------------------------
# Load ML Model
# -----------------------------
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# -----------------------------
# Structured Behavioral Keywords
# -----------------------------
# Structured Behavioral Keywords

urgency_words = [
    "urgent",
    "immediately",
    "now",
    "limited",
    "verify",
    "confirm",
    "notice"
]

financial_words = [
    "bank",
    "account",
    "password",
    "suspended",
    "update",
    "login",
    "secure",
    "service"
]

reward_words = [
    "winner",
    "lottery",
    "free",
    "reward",
    "prize",
    "bonus",
    "congratulations",
    "offer"
]

# -----------------------------
# Logging System
# -----------------------------
def log_scan(text, final_score, risk_level):
    try:
        with open("scan_logs.csv", mode="a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow([datetime.now(), text, final_score, risk_level])
    except:
        pass


# -----------------------------
# NLP Explainability
# -----------------------------
def nlp_explanation(text):
    try:
        feature_names = vectorizer.get_feature_names_out()
        coefficients = model.coef_[0]

        text_vector = vectorizer.transform([text])
        indices = text_vector.nonzero()[1]

        word_scores = []

        for index in indices:
            word = feature_names[index]
            weight = coefficients[index]
            word_scores.append((word, weight))

        word_scores = sorted(word_scores, key=lambda x: x[1], reverse=True)
        return [word for word, weight in word_scores[:5] if weight > 0]

    except:
        return []


# -----------------------------
# URL Risk Analysis
# -----------------------------
def url_risk_score(text):
    score = 0
    urls = re.findall(r'(https?://\S+)', text)

    for url in urls:
        score += 30

        if len(url) > 60:
            score += 20

        if "@" in url:
            score += 40

        if re.search(r'\d+\.\d+\.\d+\.\d+', url):
            score += 50

        for word in urgency_words + financial_words + reward_words:
            if word in url.lower():
                score += 30

    return min(score, 100)


# -----------------------------
# Behavioral Risk Analysis
# -----------------------------
def behavior_risk_score(text):

    text = text.lower()
    score = 0

    # count keyword occurrences
    urgency_hits = sum(word in text for word in urgency_words)
    financial_hits = sum(word in text for word in financial_words)
    reward_hits = sum(word in text for word in reward_words)

    # stronger weights
    score += urgency_hits * 12
    score += financial_hits * 18
    score += reward_hits * 14

    # extra boost when multiple categories appear
    categories_triggered = sum([
        urgency_hits > 0,
        financial_hits > 0,
        reward_hits > 0
    ])

    if categories_triggered >= 2:
        score += 15

    return min(score, 100)
# -----------------------------
# Main Risk Engine
# -----------------------------
def analyze_text(text):

    # ---- Input Validation ----
    if not text or not isinstance(text, str):
        return {
            "final_score": 0,
            "risk_level": "LOW RISK",
            "nlp_score": 0,
            "url_score": 0,
            "behavior_score": 0,
            "nlp_flags": []
        }

    text = text.strip()

    if len(text) > 5000:
        text = text[:5000]

    # ---- NLP Layer ----
    text_vector = vectorizer.transform([text])
    nlp_prob = model.predict_proba(text_vector)[0][1] * 100
    nlp_words = nlp_explanation(text)

    # ---- URL Layer ----
    url_score = url_risk_score(text)

    # ---- Behavioral Layer ----
    behavior_score = behavior_risk_score(text)

    # ---- Weighted Fusion ----
    final_score = (
        0.4 * nlp_prob +
        0.4 * url_score +
        0.2 * behavior_score
    )

    # Escalation Rule
    if url_score >= 70 and behavior_score >= 40:
        final_score = max(final_score, 75)

    # ---- Risk Classification ----
    if final_score >= 70:
        risk_level = "HIGH RISK"
    elif final_score >= 25:
        risk_level = "MEDIUM RISK"
    else:
        risk_level = "LOW RISK"

    return {
        "final_score": round(final_score, 2),
        "risk_level": risk_level,
        "nlp_score": round(nlp_prob, 2),
        "url_score": url_score,
        "behavior_score": behavior_score,
        "nlp_flags": nlp_words
    }


# -----------------------------
# Routes
# -----------------------------
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/api/predict")
def api_predict(text: str = Form(...)):

    result = analyze_text(text)

    log_scan(text, result["final_score"], result["risk_level"])

    return JSONResponse(result)


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard(request: Request):

    total = high = medium = low = 0

    if os.path.exists("scan_logs.csv"):
        try:
            import pandas as pd
            df = pd.read_csv("scan_logs.csv", header=None)
            df.columns = ["timestamp", "text", "score", "risk"]

            total = len(df)
            high = (df["risk"] == "HIGH RISK").sum()
            medium = (df["risk"] == "MEDIUM RISK").sum()
            low = (df["risk"] == "LOW RISK").sum()
        except:
            pass

    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "total": total,
            "high": high,
            "medium": medium,
            "low": low
        }
    )