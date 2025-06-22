# 📦 Install Dependencies (run in terminal if not installed)
# pip install pandas numpy matplotlib seaborn plotly scikit-learn statsmodels prophet tensorflow streamlit

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import streamlit as st
from math import sqrt
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disables oneDNN optimizations

# Function to calculate RSI
def compute_rsi(series, window=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    # Calculate simple moving averages
    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()
    
    # Calculate relative strength
    rs = avg_gain / avg_loss
    
    # Calculate RSI
    rsi = 100 - (100 / (1 + rs))
    return rsi

# 📥 Load Dataset
data = pd.read_csv('stock_data.csv')
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Calculate additional technical indicators
data['Daily_Return'] = data['Close'].pct_change()
data['Cumulative_Return'] = (1 + data['Daily_Return']).cumprod()
data['SMA_50'] = data['Close'].rolling(window=50).mean()
data['SMA_200'] = data['Close'].rolling(window=200).mean()
data['EMA_20'] = data['Close'].ewm(span=20, adjust=False).mean()
data['RSI'] = compute_rsi(data['Close'])

# 📊 Streamlit Dashboard
st.title("📈 Advanced Stock Market Analysis Dashboard")

# Original charts
st.header("1. Basic Price Analysis")
col1, col2 = st.columns(2)
with col1:
    st.subheader("Historical Closing Price")
    st.line_chart(data['Close'])
with col2:
    st.subheader("Trading Volume")
    st.bar_chart(data['Volume'])

# New Chart 1: Moving Averages
st.header("2. Moving Averages Analysis")
st.subheader("Price with 50-day and 200-day Moving Averages")
fig_ma, ax = plt.subplots(figsize=(12, 6))
ax.plot(data.index, data['Close'], label='Closing Price', alpha=0.5)
ax.plot(data.index, data['SMA_50'], label='50-day SMA', linestyle='--')
ax.plot(data.index, data['SMA_200'], label='200-day SMA', linestyle='--')
ax.legend()
st.pyplot(fig_ma)

# New Chart 2: Daily Returns Distribution
st.header("3. Returns Analysis")
col1, col2 = st.columns(2)
with col1:
    st.subheader("Daily Returns Distribution")
    fig_returns = plt.figure(figsize=(10, 6))
    sns.histplot(data['Daily_Return'].dropna(), bins=50, kde=True)
    st.pyplot(fig_returns)
with col2:
    st.subheader("Cumulative Returns Over Time")
    st.line_chart(data['Cumulative_Return'])

# New Chart 3: Volatility Analysis
st.header("4. Volatility Analysis")
rolling_volatility = data['Daily_Return'].rolling(window=20).std() * np.sqrt(252)  # Annualized
st.subheader("20-day Rolling Volatility (Annualized)")
st.line_chart(rolling_volatility)

# New Chart 4: Price vs Volume
st.header("5. Price-Volume Relationship")
fig_vol = px.scatter(data.reset_index(), x='Volume', y='Close', trendline="lowess",
                     title="Closing Price vs Trading Volume")
st.plotly_chart(fig_vol)

# New Chart 5: Relative Strength Index (RSI)
st.header("6. Momentum Analysis (RSI)")
st.subheader("14-day Relative Strength Index")
st.line_chart(data['RSI'])
st.markdown("""
**RSI Interpretation:**
- Above 70: Overbought (potential sell signal)
- Below 30: Oversold (potential buy signal)
""")

# Original forecast models
st.header("7. Forecasting Models")

# 📏 ARIMA Forecast
model_arima = ARIMA(data['Close'], order=(5, 1, 0))
result_arima = model_arima.fit()
forecast_arima = result_arima.forecast(steps=30)

# 📏 SARIMA Forecast
model_sarima = SARIMAX(data['Close'], order=(1, 1, 1), seasonal_order=(1, 1, 0, 12))
result_sarima = model_sarima.fit()
forecast_sarima = result_sarima.forecast(steps=30)

# 📏 Prophet Forecast
prophet_df = data.reset_index()[['Date', 'Close']]
prophet_df.columns = ['ds', 'y']
model_prophet = Prophet()
model_prophet.fit(prophet_df)
future = model_prophet.make_future_dataframe(periods=30)
forecast_prophet = model_prophet.predict(future)

# 📏 LSTM Forecast
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data[['Close']])

X, y = [], []
for i in range(60, len(data_scaled)):
    X.append(data_scaled[i-60:i, 0])
    y.append(data_scaled[i, 0])

X, y = np.array(X), np.array(y)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

model_lstm = Sequential()
model_lstm.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
model_lstm.add(LSTM(units=50))
model_lstm.add(Dense(1))

model_lstm.compile(loss='mean_squared_error', optimizer='adam')
model_lstm.fit(X, y, epochs=10, batch_size=32, verbose=0)

# 🔮 Predict Future with LSTM
pred_input = data_scaled[-60:]
pred_input = pred_input.reshape(1, 60, 1)

lstm_predictions = []

for _ in range(30):
    next_pred = model_lstm.predict(pred_input, verbose=0)[0][0]
    lstm_predictions.append(next_pred)
    pred_input = np.append(pred_input[:, 1:, :], [[[next_pred]]], axis=1)

lstm_predictions_actual = scaler.inverse_transform(np.array(lstm_predictions).reshape(-1, 1))

# 📅 Create date index for forecast
last_date = data.index[-1]
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=30)
lstm_forecast_df = pd.DataFrame({'Date': future_dates, 'LSTM_Predicted_Close': lstm_predictions_actual.flatten()})
lstm_forecast_df.set_index('Date', inplace=True)

# Display forecast charts
st.subheader("ARIMA Forecast")
st.line_chart(forecast_arima)

st.subheader("SARIMA Forecast")
st.line_chart(forecast_sarima)

st.subheader("Prophet Forecast")
st.line_chart(forecast_prophet[['ds', 'yhat']].set_index('ds').tail(30))

st.subheader("LSTM Forecast")
st.line_chart(lstm_forecast_df)

# Correlation Heatmap (Bonus Chart)
st.header("8. Feature Correlations")
corr_matrix = data[['Open', 'High', 'Low', 'Close', 'Volume', 'Daily_Return']].corr()
fig_corr = plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
st.pyplot(fig_corr)

st.success("✅ Enhanced analysis completed successfully!")