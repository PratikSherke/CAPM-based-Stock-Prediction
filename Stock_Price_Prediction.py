import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from datetime import datetime, timedelta

# Set page configuration
st.set_page_config(page_title="Stock Price Prediction Dashboard", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for enhanced styling
st.markdown("""
    <style>
        .main-title {
            font-size: 3em;
            color: #062025;
            text-align: center;
            padding: 20px 0;
            font-weight: bold;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        }
        .section-header {
            font-size: 2em;
            color: #FFFFFF;
            cursor: pointer;
            padding: 15px;
            border-radius: 8px;
            background: linear-gradient(45deg, #0057B8, #0077C0);
            margin: 15px 0;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .subheader {
            font-size: 1.8em;
            color: #0077C0;
            padding: 10px 0;
            font-weight: bold;
            display: flex;
            align-items: center;
        }
        .big-subheader {
            font-size: 2.2em;
            color: #FF5733;
            padding: 12px 0;
            font-weight: bold;
            display: flex;
            align-items: center;
            background: linear-gradient(45deg, #FFF5F0, #FFECE6);
            border-radius: 8px;
            padding-left: 10px;
            margin: 10px 0;
        }
        .description-box {
            background-color: #C4C7D0;
            border: 2px solid #0077C0;
            border-radius: 8px;
            padding: 15px;
            margin: 10px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .stButton>button {
            background: linear-gradient(45deg, #02272B, #FF8C66);
            color: white;
            border-radius: 8px;
            padding: 12px 24px;
            font-size: 1.8em;
            font-weight: bold;
            border: none;
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            background: linear-gradient(45deg, #063F05, #160101);
            box-shadow: 0 4px 6px rgba(0,0,0,0.2);
        }
        .stSidebar {
            background: linear-gradient(180deg, #E6F0FA, #F0F8FF);
        }
        .stDataFrame, .stTable {
            background-color: #FFFFFF;
            border-radius: 8px;
            padding: 10px;
            border: 1px solid #0077C0;
        }
        .stPlotlyChart {
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="main-title">📈 Stock Price Prediction Dashboard 📈</div>', unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center; font-size: 18px; font-weight: bold; color: #4e73df;'>
    🌟 Advanced stock price predictions with GRU, LSTM, SVM, and Random Forest 🌟
</div>
""", unsafe_allow_html=True)

# Sidebar for parameters
st.sidebar.header("⚙️ Configuration")
stock_options = [
    'MSFT', 'NVDA', 'AAPL', 'AMZN', 'GOOG', 'GOOGL', 'META', 'AVGO', 'BRK-B', 'TSLA',
    'WMT', 'JPM', 'V', 'LLY', 'MA', 'NFLX', 'ORCL', 'COST', 'XOM', 'PG',
    'JNJ', 'HD', 'BAC', 'ABBV', 'KO', 'PLTR', 'PM', 'TMUS', 'UNH', 'GE',
    'CRM', 'CSCO', 'IBM', 'WFC', 'CVX', 'ABT', 'LIN', 'MCD', 'INTU', 'NOW',
    'AXP', 'MS', 'DIS', 'T', 'ISRG', 'ACN', 'MRK', 'GS', 'AMD', 'RTX'
]
stock_symbols = st.sidebar.multiselect("Select Stocks", stock_options, default=['AMZN'])
lookback_days = st.sidebar.slider("Lookback Days", 30, 120, 60)
prediction_days = st.sidebar.slider("Prediction Days", 5, 30, 15)

# Fixed model parameters
TEST_SPLIT = 0.2
BATCH_SIZE = 32
EPOCHS = 40
RNN_UNITS = [50, 100, 200]
ACTIVATIONS = ['relu', 'tanh']
LEARNING_RATES = [0.001, 0.01, 0.1]

# Functions
def download_stock_data(symbols, start_date='2010-01-01', end_date=None):
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    data = yf.download(symbols, start=start_date, end=end_date, group_by='ticker')
    return data

def get_company_info(symbol):
    ticker = yf.Ticker(symbol)
    info = ticker.info
    name = info.get('longName', 'N/A')
    description = info.get('longBusinessSummary', 'N/A')
    market_cap = info.get('marketCap', 'N/A')
    current_price = info.get('regularMarketPrice', 'N/A')
    previous_close = info.get('previousClose', 'N/A')
    
    price_change = 'N/A'
    if isinstance(current_price, (int, float)) and isinstance(previous_close, (int, float)):
        price_change_value = current_price - previous_close
        price_change = f"{'📈 Up' if price_change_value > 0 else '📉 Down'} ${abs(price_change_value):.2f} ({(price_change_value/previous_close*100):.2f}%)"
    
    if isinstance(market_cap, (int, float)):
        market_cap = f"${market_cap / 1e9:.2f} Billion"
    
    return {
        'name': name,
        'description': description,
        'market_cap': market_cap,
        'current_price': f"${current_price:.2f}" if isinstance(current_price, (int, float)) else 'N/A',
        'price_change': price_change
    }

def preprocess_data(stock_data, symbol):
    df = stock_data[symbol].copy()
    df = df[['Close']]
    # Handle NaN values
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)  # Additional backfill for any remaining NaNs
    if df.isna().any().any():
        st.error(f"Data for {symbol} contains NaN values after preprocessing. Please select another stock or check the data source.")
        return None, None, None
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df)
    return scaled_data, scaler, df

def create_sequences(data, lookback_days, prediction_days):
    X, y = [], []
    for i in range(lookback_days, len(data) - prediction_days):
        X.append(data[i-lookback_days:i, 0])
        y.append(data[i:i+prediction_days, 0])
    return np.array(X), np.array(y)

def build_gru_model(input_shape):
    model = Sequential([
        GRU(RNN_UNITS[0], return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        GRU(RNN_UNITS[0]),
        Dropout(0.2),
        Dense(prediction_days)
    ])
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model

def build_lstm_model(units, activation, learning_rate, input_shape):
    model = Sequential([
        LSTM(units=units, activation=activation, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(units=units, activation=activation),
        Dropout(0.2),
        Dense(prediction_days)
    ])
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model

def train_svm_model(data, lookback_days, prediction_days):
    X = np.arange(len(data)).reshape(-1, 1)
    y = data.flatten()
    svm_params = {'C': [0.1, 1, 10], 'gamma': [0.01, 0.1, 1]}
    svm_model = SVR()
    grid_search = GridSearchCV(svm_model, svm_params, scoring='neg_mean_squared_error')
    grid_search.fit(X, y)
    return grid_search.best_estimator_, grid_search.best_params_

def train_rf_model(data, lookback_days, prediction_days):
    X = np.arange(len(data)).reshape(-1, 1)
    y = data.flatten()
    rf_params = {'n_estimators': [50, 100, 200], 'max_depth': [None, 5, 10]}
    rf_model = RandomForestRegressor()
    grid_search = GridSearchCV(rf_model, rf_params, scoring='neg_mean_squared_error')
    grid_search.fit(X, y)
    return grid_search.best_estimator_, grid_search.best_params_

def predict_future_ml(model, data_length, num_predictions):
    future_X = np.arange(data_length, data_length + num_predictions).reshape(-1, 1)
    return model.predict(future_X)

def calculate_metrics(actual, predicted):
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual, predicted)
    r2 = r2_score(actual, predicted)
    return {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R2': r2}

def plot_predictions(symbol, model_name, dates, predictions, actual_prices):
    fig = go.Figure()
    
    actual_dates = actual_prices.index[-30:]
    fig.add_trace(go.Scatter(
        x=actual_dates, y=actual_prices[-30:],
        name='Actual Prices', line=dict(color='#0057B8', width=3), mode='lines+markers',
        marker=dict(size=10, symbol='circle', line=dict(width=1, color='#003087'))
    ))
    
    connecting_date = pd.date_range(start=actual_prices.index[-1], periods=2, freq='D')
    connecting_prices = [actual_prices[-1], predictions[0]]
    
    fig.add_trace(go.Scatter(
        x=connecting_date, y=connecting_prices,
        name='Connection', line=dict(color='#0057B8', dash='dash', width=1.5), mode='lines'
    ))
    
    fig.add_trace(go.Scatter(
        x=dates, y=predictions,
        name='Predicted Prices', line=dict(color='#FF5733', width=3), mode='lines+markers',
        marker=dict(size=10, symbol='star', line=dict(width=1, color='#003087'))
    ))
    
    fig.update_layout(
        title=f'{symbol} {model_name} Prediction',
        title_font=dict(size=20, color='#0077C0'),
        xaxis_title='Date',
        yaxis_title='Price ($)',
        template='plotly_white',
        xaxis_tickangle=45,
        hovermode='x unified',
        showlegend=True,
        margin=dict(l=50, r=50, t=50, b=50),
        paper_bgcolor='#F8FAFC',
        plot_bgcolor='#FFFFFF',
        font=dict(family="Arial", size=14)
    )
    return fig

def train_and_evaluate(stock_data, symbol, model_type):
    scaled_data, scaler, actual_prices = preprocess_data(stock_data, symbol)
    if scaled_data is None:
        return None, None, None, None, None
    
    X, y = create_sequences(scaled_data, lookback_days, prediction_days)
    
    split_idx = int((1 - TEST_SPLIT) * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    
    best_rmse = float('inf')
    best_model = None
    best_params = None
    
    if model_type == 'LSTM':
        for units in RNN_UNITS:
            for activation in ACTIVATIONS:
                for lr in LEARNING_RATES:
                    model = build_lstm_model(units, activation, lr, (X_train.shape[1], X_train.shape[2]))
                    model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(X_test, y_test), verbose=0)
                    test_pred = model.predict(X_test)
                    test_pred = scaler.inverse_transform(test_pred.reshape(-1, 1))
                    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
                    # Handle NaN in predictions or actual values
                    if np.any(np.isnan(test_pred)) or np.any(np.isnan(y_test_actual)):
                        continue
                    rmse = np.sqrt(mean_squared_error(y_test_actual, test_pred))
                    if rmse < best_rmse:
                        best_rmse = rmse
                        best_model = model
                        best_params = {'units': units, 'activation': activation, 'learning_rate': lr}
    else:
        model = build_gru_model((X_train.shape[1], X_train.shape[2]))
        model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(X_test, y_test), verbose=0)
        best_model = model
        best_params = {'units': RNN_UNITS[0], 'learning_rate': 0.001}
    
    if best_model is None:
        st.error(f"Failed to train {model_type} model for {symbol} due to data issues.")
        return None, None, None, None, None
    
    test_predictions = best_model.predict(X_test)
    test_predictions = scaler.inverse_transform(test_predictions.reshape(-1, 1))
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
    
    # Handle NaN in final metrics calculation
    if np.any(np.isnan(test_predictions)) or np.any(np.isnan(y_test_actual)):
        st.error(f"NaN values detected in {model_type} predictions for {symbol}.")
        return None, None, None, None, None
    
    metrics = calculate_metrics(y_test_actual, test_predictions)
    
    last_sequence = scaled_data[-lookback_days:]
    last_sequence = np.reshape(last_sequence, (1, lookback_days, 1))
    predicted_scaled = best_model.predict(last_sequence)
    predicted_prices = scaler.inverse_transform(predicted_scaled.reshape(-1, 1)).flatten()
    
    last_date = stock_data[symbol].index[-1]
    prediction_dates = [last_date + timedelta(days=i) for i in range(1, prediction_days+1)]
    
    return prediction_dates, predicted_prices, actual_prices['Close'], metrics, best_params

# Main execution
if stock_symbols:
    if st.button("🚀 Run Analysis", key="run_analysis"):
        stock_data = download_stock_data(stock_symbols)
        
        for symbol in stock_symbols:
            with st.expander(f"📊 Analysis for {symbol}", expanded=True):
                st.markdown(f'<div class="section-header">🌐 {symbol} Overview</div>', unsafe_allow_html=True)
                
                # Company Information
                st.markdown('<div class="subheader">🏢 Company Information</div>', unsafe_allow_html=True)
                company_info = get_company_info(symbol)
                col1, col2 = st.columns([1, 1])
                with col1:
                    st.write(f"**Name**: {company_info['name']}")
                    st.markdown(f'<div class="big-subheader">💰 Market Capitalization: {company_info['market_cap']}</div>', unsafe_allow_html=True)
                with col2:
                    st.markdown(f'<div class="big-subheader">📈 Current Price: {company_info['current_price']}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="big-subheader">{company_info['price_change']}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="description-box">📝 <b>Description</b>: {company_info['description']}</div>', unsafe_allow_html=True)
                
                # Last 10 Days Stock Prices
                st.markdown('<div class="subheader">📅 Last 10 Days Stock Prices</div>', unsafe_allow_html=True)
                recent_data = stock_data[symbol].tail(10)[['Close']].copy()
                recent_data['Date'] = recent_data.index.strftime('%Y-%m-%d')
                recent_data = recent_data[['Date', 'Close']]
                recent_data['Close'] = recent_data['Close'].round(2)
                st.dataframe(recent_data.reset_index(drop=True).style.set_properties(**{
                    'background-color': "#080505",
                    'border-color': "#5F707A",
                    'color': "#CCDED4",
                    'font-weight': 'bold'
                }), use_container_width=True)
                
                # Model Training and Predictions
                st.markdown('<div class="subheader">🔮 Price Predictions</div>', unsafe_allow_html=True)
                
                scaled_data, scaler, actual_prices = preprocess_data(stock_data, symbol)
                if scaled_data is None:
                    continue
                
                # Train ML models
                svm_model, svm_params = train_svm_model(actual_prices['Close'].values, lookback_days, prediction_days)
                rf_model, rf_params = train_rf_model(actual_prices['Close'].values, lookback_days, prediction_days)
                
                # Predict future prices
                last_date = stock_data[symbol].index[-1]
                prediction_dates = [last_date + timedelta(days=i) for i in range(1, prediction_days+1)]
                
                svm_predictions = predict_future_ml(svm_model, len(actual_prices), prediction_days)
                rf_predictions = predict_future_ml(rf_model, len(actual_prices), prediction_days)
                
                # GRU and LSTM
                col1, col2 = st.columns([1, 1])
                with col1:
                    gru_dates, gru_preds, actual_prices, gru_metrics, gru_params = train_and_evaluate(stock_data, symbol, 'GRU')
                    if gru_dates is not None:
                        st.plotly_chart(plot_predictions(symbol, 'GRU', gru_dates, gru_preds, actual_prices), use_container_width=True)
                
                with col2:
                    lstm_dates, lstm_preds, actual_prices, lstm_metrics, lstm_params = train_and_evaluate(stock_data, symbol, 'LSTM')
                    if lstm_dates is not None:
                        st.plotly_chart(plot_predictions(symbol, 'LSTM', lstm_dates, lstm_preds, actual_prices), use_container_width=True)
                
                # Additional ML Models
                col3, col4 = st.columns([1, 1])
                with col3:
                    st.plotly_chart(plot_predictions(symbol, 'SVM', prediction_dates, svm_predictions, actual_prices), use_container_width=True)
                with col4:
                    st.plotly_chart(plot_predictions(symbol, 'Random Forest', prediction_dates, rf_predictions, actual_prices), use_container_width=True)
                
                # Model Performance Metrics
                st.markdown('<div class="subheader">📊 Model Performance Metrics</div>', unsafe_allow_html=True)
                metrics_df = pd.DataFrame({
                    'Metric': ['MSE', 'RMSE', 'MAE', 'R2'],
                    'GRU': [gru_metrics['MSE'] if gru_metrics else 'N/A', gru_metrics['RMSE'] if gru_metrics else 'N/A', 
                            gru_metrics['MAE'] if gru_metrics else 'N/A', gru_metrics['R2'] if gru_metrics else 'N/A'],
                    'LSTM': [lstm_metrics['MSE'] if lstm_metrics else 'N/A', lstm_metrics['RMSE'] if lstm_metrics else 'N/A', 
                             lstm_metrics['MAE'] if lstm_metrics else 'N/A', lstm_metrics['R2'] if lstm_metrics else 'N/A'],
                    'SVM': calculate_metrics(actual_prices[-prediction_days:], svm_predictions)['MSE', 'RMSE', 'MAE', 'R2'],
                    'Random Forest': calculate_metrics(actual_prices[-prediction_days:], rf_predictions)['MSE', 'RMSE', 'MAE', 'R2']
                })
                st.dataframe(metrics_df.style.format("{:.4f}", subset=['GRU', 'LSTM', 'SVM', 'Random Forest']).set_properties(**{
                    'background-color': "#749AC7",
                    'border-color': '#FF5733',
                    'color': '#003087',
                    'font-weight': 'bold'
                }), use_container_width=True)
                
                # Predicted Prices Table
                st.markdown('<div class="subheader">📈 Predicted Stock Prices</div>', unsafe_allow_html=True)
                pred_df = pd.DataFrame({
                    'Date': [d.strftime('%Y-%m-%d') for d in prediction_dates],
                    'GRU': [f"${p:.2f}" if gru_preds is not None else 'N/A' for p in (gru_preds if gru_preds is not None else [0]*prediction_days)],
                    'LSTM': [f"${p:.2f}" if lstm_preds is not None else 'N/A' for p in (lstm_preds if lstm_preds is not None else [0]*prediction_days)],
                    'SVM': [f"${p:.2f}" for p in svm_predictions],
                    'Random Forest': [f"${p:.2f}" for p in rf_predictions]
                })
                st.dataframe(pred_df.style.set_properties(**{
                    'background-color': "#85B0D6",
                    'border-color': '#0077C0',
                    'color': '#003087',
                    'font-weight': 'bold'
                }), use_container_width=True)
                
                # Best Hyperparameters
                st.markdown('<div class="subheader">⚙️ Best Hyperparameters</div>', unsafe_allow_html=True)
                params_df = pd.DataFrame({
                    'Model': ['LSTM', 'SVM', 'Random Forest'],
                    'Parameters': [
                        str(lstm_params) if lstm_params else 'N/A',
                        str(svm_params),
                        str(rf_params)
                    ]
                })
                st.dataframe(params_df.style.set_properties(**{
                    'background-color': "#85B0D6",
                    'border-color': '#0077C0',
                    'color': '#003087',
                    'font-weight': 'bold'
                }), use_container_width=True)

else:
    st.warning("⚠️ Please select at least one stock to analyze.")