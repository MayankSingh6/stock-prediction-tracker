import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib
from database import init_db, save_predictions, save_actuals

# Configuration
TICKERS = ['SPY', 'QQQ', 'AAPL', 'MSFT', 'TSLA']
MODEL_PATH = 'model.pkl'
MODEL_VERSION = 'v1.0'
LOOKBACK_DAYS = 730  # 2 years of training data

def fetch_data(tickers, period='2y'):
    """Fetch historical data for all tickers"""
    print(f"📥 Fetching data for {tickers}...")
    data = {}
    
    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)  # ~2 years
    
    for ticker in tickers:
        try:
            print(f"  Fetching {ticker}...")
            # Use date range instead of period
            df = yf.download(
                ticker, 
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d'),
                progress=False
            )
            
            if len(df) > 0:
                data[ticker] = df
                print(f"  ✅ {ticker}: {len(df)} days")
            else:
                print(f"  ❌ {ticker}: No data")
                
        except Exception as e:
            print(f"  ❌ {ticker}: Error - {e}")
    
    return data

def calculate_features(df):
    """Calculate technical features for a single ticker"""
    df = df.copy()
    
    # Flatten multi-level columns if they exist
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # Make sure we have the right column names
    if 'Close' not in df.columns and 'close' in df.columns:
        df = df.rename(columns={'close': 'Close', 'volume': 'Volume', 
                                'open': 'Open', 'high': 'High', 'low': 'Low'})
    
    # Returns
    df['return_1d'] = df['Close'].pct_change(1)
    df['return_5d'] = df['Close'].pct_change(5)
    df['return_10d'] = df['Close'].pct_change(10)
    
    # Volatility
    df['volatility_20d'] = df['return_1d'].rolling(20).std()
    
    # Volume - handle potential issues
    if 'Volume' in df.columns:
        df['volume_5d_avg'] = df['Volume'].rolling(5).mean()
        df['volume_change'] = (df['Volume'] / df['volume_5d_avg'] - 1).fillna(0)
    else:
        # If no volume data, use dummy values
        df['volume_change'] = 0
    
    # Moving average
    df['ma_20'] = df['Close'].rolling(20).mean()
    df['price_to_ma'] = df['Close'] / df['ma_20'] - 1
    
    # Target: next day return
    df['target'] = df['Close'].shift(-1) / df['Close'] - 1
    
    # Drop NaN rows
    df = df.dropna()
    
    return df

def prepare_training_data(data_dict):
    """Combine all tickers into one training dataset"""
    print("🔧 Building features...")
    
    all_data = []
    
    for ticker, df in data_dict.items():
        df_features = calculate_features(df)
        df_features['ticker'] = ticker
        all_data.append(df_features)
    
    # Combine all tickers
    combined = pd.concat(all_data, ignore_index=True)
    
    # Feature columns
    feature_cols = [
        'return_1d', 'return_5d', 'return_10d',
        'volatility_20d', 'volume_change', 'price_to_ma'
    ]
    
    X = combined[feature_cols]
    y = combined['target']
    
    print(f"  ✅ Training data: {len(X)} samples, {len(feature_cols)} features")
    
    return X, y, feature_cols, combined

def train_model(X, y):
    """Train Random Forest model"""
    print("🎯 Training model...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False
    )
    
    # Train model
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=20,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    print(f"  ✅ Train R²: {train_score:.4f}")
    print(f"  ✅ Test R²: {test_score:.4f}")
    
    # Save model
    joblib.dump(model, MODEL_PATH)
    print(f"  ✅ Model saved to {MODEL_PATH}")
    
    return model

def make_predictions(model, data_dict, feature_cols):
    """Make predictions for next trading day"""
    print("🔮 Making predictions...")
    
    predictions = []
    
    for ticker, df in data_dict.items():
        # Get latest data point
        df_features = calculate_features(df)
        
        if len(df_features) == 0:
            print(f"  ⚠️  {ticker}: Not enough data")
            continue
        
        latest = df_features.iloc[-1]
        
        # Prepare features
        X_pred = latest[feature_cols].values.reshape(1, -1)
        
        # Make prediction
        pred_return = model.predict(X_pred)[0]
        
        # Get next trading day (tomorrow)
        last_date = df.index[-1]
        next_date = last_date + timedelta(days=1)
        
        # Skip weekends
        while next_date.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
            next_date += timedelta(days=1)
        
        predictions.append({
            'ticker': ticker,
            'date_for': next_date.strftime('%Y-%m-%d'),
            'predicted_return': pred_return,
            'last_close': latest['Close']
        })
        
        print(f"  ✅ {ticker}: {pred_return*100:+.2f}% (for {next_date.strftime('%Y-%m-%d')})")
    
    return pd.DataFrame(predictions)

def record_actuals(data_dict):
    """Record actual returns from yesterday's predictions"""
    print("📊 Recording actual returns...")
    
    actuals = []
    
    for ticker, df in data_dict.items():
        # Get last 5 days of actual data
        recent = df.tail(5).copy()
        recent['actual_return'] = recent['Close'].pct_change(1)
        recent = recent.dropna()
        
        for date, row in recent.iterrows():
            actuals.append({
                'ticker': ticker,
                'date': date.strftime('%Y-%m-%d'),
                'actual_return': row['actual_return'].item(),  # Use .item() instead
                'close_price': row['Close'].item()  # Use .item() instead
            })
    
    actuals_df = pd.DataFrame(actuals)
    
    if len(actuals_df) > 0:
        save_actuals(actuals_df)
    
    return actuals_df

def run_full_pipeline():
    """Run the complete pipeline"""
    print("\n" + "="*50)
    print("🚀 STOCK PREDICTION PIPELINE")
    print("="*50 + "\n")
    
    # Initialize database
    init_db()
    
    # Fetch data
    data = fetch_data(TICKERS)
    
    if len(data) == 0:
        print("❌ No data fetched. Exiting.")
        return
    
    # Prepare training data
    X, y, feature_cols, combined = prepare_training_data(data)
    
    # Train model
    model = train_model(X, y)
    
    # Make predictions for tomorrow
    predictions_df = make_predictions(model, data, feature_cols)
    save_predictions(predictions_df, MODEL_VERSION)
    
    # Record actual returns
    actuals_df = record_actuals(data)
    
    print("\n" + "="*50)
    print("✅ PIPELINE COMPLETE")
    print("="*50)
    print(f"\nPredictions for tomorrow:")
    print(predictions_df[['ticker', 'predicted_return', 'date_for']].to_string(index=False))
    print("\n")

if __name__ == "__main__":
    run_full_pipeline()