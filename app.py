import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from database import get_predictions, get_actuals, init_db
from datetime import datetime, timedelta

# Page config
st.set_page_config(
    page_title="Stock Prediction Tracker",
    page_icon="📈",
    layout="wide"
)

# Initialize database
init_db()

# Title
st.title("📈 Stock Prediction Tracker")
st.markdown("*AI-powered next-day return predictions*")

# Sidebar
st.sidebar.header("Settings")
days_to_show = st.sidebar.slider("Days to display", 7, 60, 30)

# Create tabs
tab1, tab2, tab3 = st.tabs(["📊 Today's Predictions", "📈 Performance", "ℹ️ Model Info"])

# ============================================
# TAB 1: Today's Predictions
# ============================================
with tab1:
    st.header("Latest Predictions")
    
    # Get predictions
    predictions_df = get_predictions(days=5)
    
    if len(predictions_df) > 0:
        # Get most recent predictions
        latest_date = predictions_df['date_for'].max()
        latest_predictions = predictions_df[predictions_df['date_for'] == latest_date].copy()
        
        st.subheader(f"Predictions for {latest_date}")
        
        # Format the data
        latest_predictions['predicted_return_pct'] = latest_predictions['predicted_return'] * 100
        latest_predictions['direction'] = latest_predictions['predicted_return'].apply(
            lambda x: '🟢 UP' if x > 0 else '🔴 DOWN'
        )
        
        # Sort by predicted return
        latest_predictions = latest_predictions.sort_values('predicted_return', ascending=False)
        
        # Display table
        display_df = latest_predictions[['ticker', 'predicted_return_pct', 'direction']].copy()
        display_df.columns = ['Ticker', 'Predicted Return (%)', 'Direction']
        display_df['Predicted Return (%)'] = display_df['Predicted Return (%)'].apply(lambda x: f"{x:+.2f}%")
        
        st.dataframe(
            display_df,
            hide_index=True,
            use_container_width=True,
            height=250
        )
        
        # Highlight top picks
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                "🟢 Most Bullish",
                latest_predictions.iloc[0]['ticker'],
                f"{latest_predictions.iloc[0]['predicted_return']*100:+.2f}%"
            )
        
        with col2:
            st.metric(
                "🔴 Most Bearish",
                latest_predictions.iloc[-1]['ticker'],
                f"{latest_predictions.iloc[-1]['predicted_return']*100:+.2f}%"
            )
    else:
        st.info("No predictions yet. Run `python pipeline.py` to generate predictions.")

# ============================================
# TAB 2: Performance
# ============================================
with tab2:
    st.header("Model Performance")
    
    # Get data
    predictions_df = get_predictions(days=days_to_show)
    actuals_df = get_actuals(days=days_to_show)
    
    if len(predictions_df) > 0 and len(actuals_df) > 0:
        # Merge predictions with actuals
        merged = predictions_df.merge(
            actuals_df,
            left_on=['ticker', 'date_for'],
            right_on=['ticker', 'date'],
            how='inner'
        )
        
        if len(merged) > 0:
            # Calculate metrics
            merged['error'] = abs(merged['predicted_return'] - merged['actual_return'])
            merged['direction_correct'] = (
                (merged['predicted_return'] > 0) == (merged['actual_return'] > 0)
            )
            
            # Overall metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                mae = merged['error'].mean() * 100
                st.metric("Mean Absolute Error", f"{mae:.2f}%")
            
            with col2:
                direction_acc = merged['direction_correct'].mean() * 100
                st.metric("Direction Accuracy", f"{direction_acc:.1f}%")
            
            with col3:
                st.metric("Predictions Made", len(merged))
            
            # Prediction vs Actual scatter plot
            st.subheader("Predicted vs Actual Returns")
            
            fig = px.scatter(
                merged,
                x='predicted_return',
                y='actual_return',
                color='ticker',
                title='Prediction Accuracy by Ticker',
                labels={
                    'predicted_return': 'Predicted Return',
                    'actual_return': 'Actual Return'
                },
                hover_data=['date_for']
            )
            
            # Add diagonal line (perfect prediction)
            min_val = min(merged['predicted_return'].min(), merged['actual_return'].min())
            max_val = max(merged['predicted_return'].max(), merged['actual_return'].max())
            fig.add_trace(
                go.Scatter(
                    x=[min_val, max_val],
                    y=[min_val, max_val],
                    mode='lines',
                    name='Perfect Prediction',
                    line=dict(color='gray', dash='dash')
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Accuracy over time
            st.subheader("Accuracy Over Time")
            
            daily_metrics = merged.groupby('date_for').agg({
                'error': 'mean',
                'direction_correct': 'mean'
            }).reset_index()
            
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=daily_metrics['date_for'],
                y=daily_metrics['direction_correct'] * 100,
                mode='lines+markers',
                name='Direction Accuracy (%)',
                line=dict(color='green')
            ))
            
            fig2.update_layout(
                title='Direction Accuracy Over Time',
                xaxis_title='Date',
                yaxis_title='Accuracy (%)',
                hovermode='x unified'
            )
            
            st.plotly_chart(fig2, use_container_width=True)
            
        else:
            st.info("No matching predictions and actuals yet. Predictions need at least one day to be evaluated.")
    else:
        st.info("Not enough data to show performance. Run the pipeline for a few days.")

# ============================================
# TAB 3: Model Info
# ============================================
with tab3:
    st.header("Model Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Configuration")
        st.write("**Model Type:** Random Forest Regressor")
        st.write("**Tickers:** SPY, QQQ, AAPL, MSFT, TSLA")
        st.write("**Training Window:** 2 years")
        st.write("**Model Version:** v1.0")
    
    with col2:
        st.subheader("Features Used")
        features = [
            "1-day return",
            "5-day return",
            "10-day return",
            "20-day volatility",
            "Volume change",
            "Price vs 20-day MA"
        ]
        for feature in features:
            st.write(f"• {feature}")
    
    st.subheader("How it Works")
    st.markdown("""
    1. **Data Collection**: Downloads 2 years of historical OHLCV data
    2. **Feature Engineering**: Calculates technical indicators
    3. **Model Training**: Trains on all tickers combined
    4. **Prediction**: Predicts next-day return for each ticker
    5. **Evaluation**: Compares predictions with actual outcomes
    
    **Note:** This is an educational project. Not financial advice.
    """)
    
    # Show raw data option
    if st.checkbox("Show Raw Predictions Data"):
        predictions_df = get_predictions(days=30)
        st.dataframe(predictions_df, use_container_width=True)
    
    if st.checkbox("Show Raw Actuals Data"):
        actuals_df = get_actuals(days=30)
        st.dataframe(actuals_df, use_container_width=True)