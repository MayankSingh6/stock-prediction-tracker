import sqlite3
from datetime import datetime
import pandas as pd

DB_PATH = "predictions.db"

def init_db():
    """Create tables if they don't exist"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Predictions table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            date_for TEXT NOT NULL,
            predicted_return REAL NOT NULL,
            model_version TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    ''')
    
    # Actuals table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS actuals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            date TEXT NOT NULL,
            actual_return REAL NOT NULL,
            close_price REAL NOT NULL,
            created_at TEXT NOT NULL,
            UNIQUE(ticker, date)
        )
    ''')
    
    conn.commit()
    conn.close()
    print("✅ Database initialized")

def save_predictions(predictions_df, model_version):
    """Save predictions to database"""
    conn = sqlite3.connect(DB_PATH)
    
    for _, row in predictions_df.iterrows():
        conn.execute('''
            INSERT INTO predictions (ticker, date_for, predicted_return, model_version, created_at)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            row['ticker'],
            row['date_for'],
            row['predicted_return'],
            model_version,
            datetime.now().isoformat()
        ))
    
    conn.commit()
    conn.close()
    print(f"✅ Saved {len(predictions_df)} predictions")

def save_actuals(actuals_df):
    """Save actual returns to database"""
    conn = sqlite3.connect(DB_PATH)
    
    for _, row in actuals_df.iterrows():
        try:
            conn.execute('''
                INSERT INTO actuals (ticker, date, actual_return, close_price, created_at)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                row['ticker'],
                row['date'],
                row['actual_return'],
                row['close_price'],
                datetime.now().isoformat()
            ))
        except sqlite3.IntegrityError:
            # Already exists, skip
            pass
    
    conn.commit()
    conn.close()
    print(f"✅ Saved {len(actuals_df)} actuals")

def get_predictions(days=30):
    """Get recent predictions"""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(f'''
        SELECT * FROM predictions 
        ORDER BY date_for DESC 
        LIMIT {days * 5}
    ''', conn)
    conn.close()
    return df

def get_actuals(days=30):
    """Get recent actuals"""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(f'''
        SELECT * FROM actuals 
        ORDER BY date DESC 
        LIMIT {days * 5}
    ''', conn)
    conn.close()
    return df

if __name__ == "__main__":
    # Test the database
    init_db()
    print("Database test successful!")