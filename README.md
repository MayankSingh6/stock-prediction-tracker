# 📈 Stock Prediction Tracker

An AI-powered stock prediction dashboard that forecasts next-day returns for SPY, QQQ, AAPL, MSFT, and TSLA.

## Features
- Daily predictions using Random Forest ML model
- Performance tracking and accuracy metrics
- Clean Streamlit dashboard

## Setup
```bash
pip install -r requirements.txt
python pipeline.py  # Run daily to get predictions
streamlit run app.py  # View dashboard
```

## Tech Stack
- Python, scikit-learn, yfinance
- Streamlit for UI
- SQLite database