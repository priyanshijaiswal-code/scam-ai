# AI Multi-Layer Scam Detection System

This project detects scam messages using machine learning and multi-layer risk analysis.

Features:
- Real-time scam detection
- NLP classification (TF-IDF + Logistic Regression)
- URL risk detection
- Behavioral keyword analysis
- Risk scoring engine
- Logging system
- Analytics dashboard

Tech Stack:
Python, FastAPI, Scikit-learn, TailwindCSS

Run the project:

1. Install dependencies
pip install -r requirements.txt

2. Start server
uvicorn app:app --reload

3. Open browser
http://127.0.0.1:8000

Dashboard
http://127.0.0.1:8000/dashboard