import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from io import BytesIO
from datetime import timedelta, datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error
from streamlit_autorefresh import st_autorefresh
import requests
from bs4 import BeautifulSoup
import warnings
import nltk
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

# --- Conditional Imports for Optional Libraries ---
try:
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
except ImportError:
    SentimentIntensityAnalyzer = None
    st.warning("VADER Sentiment Analyzer not available. Run: pip install nltk")

try:
    from prophet import Prophet
except ImportError:
    Prophet = None

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, LSTM, Dropout
    from sklearn.preprocessing import MinMaxScaler
except ImportError:
    tf, Sequential, LSTM, MinMaxScaler = None, None, None, None
    
warnings.filterwarnings('ignore')

# --- Streamlit Setup ---
st.set_page_config(page_title="Indian Stock Dashboard", layout="wide", initial_sidebar_state="expanded", page_icon="📊")
st.markdown("""
<style>
.main > div {padding-top: 2rem;}
.metric-card {background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #1f77b4;}
.stTabs [data-baseweb="tab-list"] {gap: 2px;}
.stTabs [data-baseweb="tab"] {height: 50px; background-color: #f0f2f6; border-radius: 4px 4px 0 0; padding: 0 20px;}
.stTabs [aria-selected="true"] {background-color: #1f77b4; color: white;}
</style>
""", unsafe_allow_html=True)
st.markdown("# 📊 Indian Stock Market Dashboard\n### Advanced Analysis • Forecasting • Sentiment")

# --- NLTK Data Check (Fix: More robust check) ---
if SentimentIntensityAnalyzer:
    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        with st.spinner("Downloading sentiment data..."):
            try:
                nltk.download("vader_lexicon", quiet=True)
            except Exception as e:
                st.error(f"Failed to download NLTK data: {e}")
                SentimentIntensityAnalyzer = None

# --- Sidebar Configuration ---
with st.sidebar:
    st.markdown("## 🔧 Configuration")
    st.markdown("### 📈 Stock Selection")
    tickers_input = st.text_input("Enter NSE stock symbols (comma separated)", "RELIANCE,TCS,INFY,HDFC")
    popular_stocks = {
        "FAANG of India": "RELIANCE,TCS,INFY,HDFCBANK,ITC",
        "Banking Stocks": "HDFCBANK,ICICIBANK,SBIN,KOTAKBANK,AXISBANK",
        "IT Stocks": "TCS,INFY,WIPRO,HCLTECH,TECHM",
        "Auto Stocks": "MARUTI,TATAMOTORS,M&M,BAJAJ-AUTO,HEROMOTOCO"
    }
    preset = st.selectbox("Or choose a preset:", ["Custom"] + list(popular_stocks.keys()))
    if preset != "Custom":
        tickers_input = popular_stocks[preset]
        st.info(f"Selected: {preset}")
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
    symbols = [f"{t}.NS" for t in tickers]
    st.markdown("### 📊 Analysis Mode")
    mode = st.radio("Select mode", ["Single Stock Analysis", "Multi-Stock Comparison", "Portfolio Analysis"], index=1 if len(symbols) > 1 else 0) # Enhancement: Default to comparison if multiple stocks are entered
    
    st.markdown("### 🔮 Forecasting")
    # Enhancement: Disable/Info for missing libraries
    forecast_options = ["None", "ARIMA", "SARIMA"]
    if Prophet: forecast_options.append("Prophet")
    if LSTM: forecast_options.append("LSTM")
    
    forecast_method = st.selectbox("Forecast method", forecast_options)
    if forecast_method not in forecast_options and forecast_method != "None":
        st.warning(f"'{forecast_method}' requires external libraries which failed to import.")
        forecast_method = "None"
        
    forecast_days = st.slider("Forecast horizon (days)", 5, 365, 30, step=5)
    
    st.markdown("### ⚙️ Settings")
    period_options = {"1 Month": "1mo", "3 Months": "3mo", "6 Months": "6mo", "1 Year": "1y", "2 Years": "2y", "5 Years": "5y", "Max": "max"} # Enhancement: Added "Max" period
    period = period_options[st.selectbox("Data period", list(period_options.keys()), index=4)]
    show_volume = st.checkbox("Show volume data", value=True)
    show_news = st.checkbox("Show news & sentiment", value=True)
    refresh_rate = st.slider("Auto-refresh (seconds, 0 = off)", 0, 300, 0)
    if refresh_rate > 0:
        st_autorefresh(interval=refresh_rate * 1000, key="refresh")
        
    st.markdown("---")
    st.markdown("### 📋 Current Selection")
    for i, ticker in enumerate(tickers, 1):
        st.markdown(f"{i}. **{ticker}**")

# --- Core Data Functions ---
@st.cache_data(ttl=300)
def load_data(ticker, period):
    """Loads stock data from yfinance with caching."""
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period, auto_adjust=True) # Enhancement: auto_adjust for better historical data
        if df.empty:
            return None, None
        df.reset_index(inplace=True)
        # Fix: Robustly handle datetime with or without timezone
        if 'Datetime' in df.columns:
            df.rename(columns={'Datetime': 'Date'}, inplace=True)
        df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
        
        # Enhancement: Fetching key info only once
        info = stock.info
        
        return df, info
    except Exception as e:
        st.error(f"Error loading {ticker}: {str(e)}")
        return None, None

@st.cache_data(ttl=1800)
def get_news(ticker, max_articles=10):
    """Scrapes news headlines from Yahoo Finance."""
    url = f"https://in.finance.yahoo.com/quote/{ticker}/news?p={ticker}"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/91.0.4472.124"}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            return []
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Enhancement: Simplified, more robust selection for Yahoo Finance news titles
        articles = soup.select('div.Ov(h) > div > div > div > div > div > div > ul > li') or \
                   soup.select('li[data-test-locator="stream-item"]') or \
                   soup.select('.js-stream-content article')
        
        news = []
        for article in articles:
            try:
                a_tag = article.find('a')
                if a_tag and a_tag.text:
                    title = a_tag.text.strip()
                    # Fix: Robustly extract link, handling relative and absolute URLs
                    link = a_tag.get('href', '#')
                    if link.startswith('/quote/'):
                        link = "https://in.finance.yahoo.com" + link
                    elif not link.startswith('http'): # Fallback for other relative links
                         link = "https://in.finance.yahoo.com" + link
                    
                    if link != '#':
                        news.append((title, link))
                        if len(news) >= max_articles:
                            break
            except:
                continue
        return news
    except Exception as e:
        # st.warning(f"News fetch failed: {str(e)}") # Removed as it can be too noisy
        return []

@st.cache_data(ttl=3600) # Enhancement: Cache this function too
def add_technical_indicators(df):
    """Adds a comprehensive set of technical indicators."""
    df = df.copy()
    
    # Moving Averages (SMA & EMA)
    df["SMA20"] = df["Close"].rolling(window=20).mean()
    df["SMA50"] = df["Close"].rolling(window=50).mean()
    df["SMA200"] = df["Close"].rolling(window=200).mean()
    df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()
    
    # Bollinger Bands (BB)
    rolling_mean = df["Close"].rolling(window=20).mean()
    rolling_std = df["Close"].rolling(window=20).std()
    df["BB_Upper"] = rolling_mean + (rolling_std * 2)
    df["BB_Lower"] = rolling_mean - (rolling_std * 2)
    df["BB_Middle"] = rolling_mean
    
    # Relative Strength Index (RSI)
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    # Fix: Handle division by zero/NaN for stable RSI calculation
    rs = gain.divide(loss.replace(0, np.nan)) # Replace 0 with NaN for division
    df["RSI"] = 100 - (100 / (1 + rs))
    
    # Moving Average Convergence Divergence (MACD)
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Histogram"] = df["MACD"] - df["Signal"]
    
    # Stochastic Oscillator (Stoch)
    low_min = df["Low"].rolling(window=14).min()
    high_max = df["High"].rolling(window=14).max()
    # Fix: Handle division by zero/NaN for stable Stochastic calculation
    denominator = (high_max - low_min)
    df["Stoch_K"] = 100 * ((df["Close"] - low_min) / denominator.replace(0, np.nan))
    df["Stoch_D"] = df["Stoch_K"].rolling(window=3).mean()
    
    # Average True Range (ATR)
    high_low = df["High"] - df["Low"]
    high_close = np.abs(df["High"] - df["Close"].shift())
    low_close = np.abs(df["Low"] - df["Close"].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["ATR"] = true_range.ewm(span=14, adjust=False).mean() # Enhancement: Use EMA for ATR for better responsiveness
    
    return df

def compute_metrics(actual, predicted):
    """Computes standard regression and directional accuracy metrics."""
    # ... (existing compute_metrics function is fine) ...
    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    actual_arr, pred_arr = np.array(actual), np.array(predicted)
    mask = actual_arr != 0
    # Fix: Use np.mean for MAPE calculation to handle masked array correctly
    mape = (np.abs((actual_arr[mask] - pred_arr[mask]) / actual_arr[mask]).mean() * 100) if mask.sum() else float("nan")
    ss_res = np.sum((actual_arr - pred_arr) ** 2)
    ss_tot = np.sum((actual_arr - np.mean(actual_arr)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    actual_direction = np.diff(actual_arr) > 0
    pred_direction = np.diff(pred_arr) > 0
    direction_accuracy = np.mean(actual_direction == pred_direction) * 100 if len(actual_direction) > 0 else 0
    return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape, 'R²': r2, 'Direction Accuracy (%)': direction_accuracy}

def create_download_buttons(df, filename_prefix):
    """Generates and displays download buttons for CSV and Excel."""
    # ... (existing create_download_buttons function is fine) ...
    col1, col2 = st.columns(2)
    with col1:
        csv = df.to_csv(index=True).encode("utf-8") # Fix: ensure index is included if it's the date
        st.download_button("📥 Download as CSV", csv, f"{filename_prefix}_{datetime.now().strftime('%Y%m%d')}.csv", mime="text/csv")
    with col2:
        excel_buffer = BytesIO()
        df_export = df.copy()
        # Fix: Robustly handle timezone awareness before exporting
        if isinstance(df_export.index, pd.DatetimeIndex) and df_export.index.tz is not None:
            df_export.index = df_export.index.tz_localize(None)
        for col in df_export.select_dtypes(include=["datetimetz"]).columns:
            df_export[col] = df_export[col].dt.tz_localize(None)
        with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
            df_export.to_excel(writer, index=True, sheet_name="Data")
        st.download_button("📥 Download as Excel", excel_buffer.getvalue(), f"{filename_prefix}_{datetime.now().strftime('%Y%m%d')}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# --- Forecasting Functions ---
def forecast_prophet(df, periods):
    if not Prophet:
        st.error("Prophet is not installed or failed to import.")
        return None
    try:
        fdf = df[["ds", "y"]].dropna().copy()
        fdf["ds"] = pd.to_datetime(fdf["ds"]).dt.tz_localize(None)
        if fdf.empty or len(fdf) < 10:
            st.warning("Insufficient data for Prophet")
            return None
        # Enhancement: Added a simple growth model parameter
        m = Prophet(daily_seasonality=False, yearly_seasonality=True, weekly_seasonality=True, growth='linear')
        m.fit(fdf)
        future = m.make_future_dataframe(periods=periods)
        forecast = m.predict(future)
        return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].rename(columns={"ds": "Date", "yhat": "Forecast", "yhat_lower": "Lower_CI", "yhat_upper": "Upper_CI"}).set_index("Date")
    except Exception as e:
        st.error(f"Prophet forecast error: {str(e)}")
        return None

def forecast_lstm(df, periods, epochs=50, batch_size=32):
    if not LSTM:
        st.error("TensorFlow/Keras is not installed or failed to import.")
        return None
    try:
        # Enhancement: Use Close price for time series modeling, index by date for consistency
        series = df.set_index("Date")["Close"].dropna().values.reshape(-1, 1)
        
        # 1. Improved Sequence Length Guardrail
        min_data_points = 60 
        if len(series) < min_data_points:
            # Fix: Make the warning more precise and return early
            st.warning(f"Insufficient data for LSTM. Need at least {min_data_points} data points, but only have {len(series)}.")
            return None
            
        scaler = MinMaxScaler((0, 1))
        scaled = scaler.fit_transform(series)
        
        # Sequence length calculation should be safe here
        seq_len = min(60, len(scaled) // 4)
        
        X, y = [], []
        for i in range(seq_len, len(scaled)):
            X.append(scaled[i-seq_len:i, 0])
            y.append(scaled[i, 0])
            
        # 2. FIX: Check if the feature array X is empty before converting to NumPy
        if not X:
            st.warning(f"LSTM sequence creation failed. Check data length and sequence length ({seq_len}).")
            return None
            
        X, y = np.array(X), np.array(y)
        X = X.reshape((X.shape[0], X.shape[1], 1))
        
        # ... (rest of the model definition and training logic) ...

        # Check if the training set is still too small after creating sequences
        if X.shape[0] < 10:
             st.warning(f"Insufficient sequences created for LSTM training. Only {X.shape[0]} sequences generated.")
             return None

        # Existing training logic (simplified for multi-stock)
        epochs_to_run = 10 if len(st.session_state.get("current_tickers", [])) > 1 else epochs
        
        # Model definition and training...
        model = Sequential([
            LSTM(30, return_sequences=False, input_shape=(X.shape[1], 1)),
            Dropout(0.2),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        
        # Progress bar logic for single stock
        if len(st.session_state.get("current_tickers", [])) == 1:
             progress_bar = st.progress(0)
             class ProgressCallback(tf.keras.callbacks.Callback):
                 def on_epoch_end(self, epoch, logs=None):
                     progress_bar.progress((epoch + 1) / epochs_to_run)
             model.fit(X, y, epochs=epochs_to_run, batch_size=batch_size, verbose=0, callbacks=[ProgressCallback()])
             progress_bar.empty()
        else:
            model.fit(X, y, epochs=epochs_to_run, batch_size=batch_size, verbose=0)
            
        # Prediction logic
        last_seq = scaled[-seq_len:].reshape(1, seq_len, 1)
        preds = []
        cur_seq = last_seq.copy()
        for _ in range(periods):
            # FIX: The original error was often caused by `cur_seq` being empty, but the `if not X` check above handles the core issue. 
            # This loop is generally safe if X was successfully created.
            pred = model.predict(cur_seq, verbose=0)[0, 0]
            preds.append(pred)
            cur_seq = np.roll(cur_seq, -1, axis=1)
            cur_seq[0, -1, 0] = pred
            
        preds = scaler.inverse_transform(np.array(preds).reshape(-1, 1))
        last_date = df["Date"].max()
        dates = pd.to_datetime([last_date + timedelta(days=i+1) for i in range(periods)])
        return pd.DataFrame(preds.flatten(), index=dates, columns=["Forecast"])
        
    except Exception as e:
        # Re-raise the error for better debugging, or provide a generic message
        st.error(f"LSTM forecast error: {str(e)}")
        return None

def forecast_arima(df, periods, order=(5,1,0)):
    # ... (existing ARIMA function is fine, ensure index is datetime) ...
    try:
        series = df["Close"].dropna()
        series.index = pd.to_datetime(series.index) # Fix: Ensure index is datetime
        if series.empty or len(series) < 50:
            st.warning("Insufficient data for ARIMA")
            return None
        
        # Enhancement: Wrap ARIMA fit to avoid common warnings/errors
        model = ARIMA(series, order=order)
        fit = model.fit(low_memory=True)
        
        forecast = fit.get_forecast(steps=periods)
        forecast_values = forecast.predicted_mean
        conf_int = forecast.conf_int()
        
        # Fix: Generate dates for non-contiguous index (daily basis from last known date)
        last_date = series.index.max()
        dates = pd.to_datetime([last_date + timedelta(days=i+1) for i in range(periods)])
        
        return pd.DataFrame({'Forecast': forecast_values.values, 'Lower_CI': conf_int.iloc[:, 0].values, 'Upper_CI': conf_int.iloc[:, 1].values}, index=dates)
    except Exception as e:
        st.error(f"ARIMA forecast error: {str(e)}")
        return None

def forecast_sarima(df, periods, order=(2,1,2), seasonal_order=(1,1,1,12)):
    # ... (existing SARIMA function is fine, ensure index is datetime) ...
    try:
        series = df["Close"].dropna()
        series.index = pd.to_datetime(series.index) # Fix: Ensure index is datetime
        if series.empty or len(series) < 100:
            st.warning("Insufficient data for SARIMA")
            return None
        
        # Enhancement: Wrap SARIMA fit to avoid common warnings/errors
        model = SARIMAX(series, order=order, seasonal_order=seasonal_order)
        fit = model.fit(disp=False, low_memory=True)
        
        forecast = fit.get_forecast(steps=periods)
        forecast_values = forecast.predicted_mean
        conf_int = forecast.conf_int()
        
        # Fix: Generate dates for non-contiguous index (daily basis from last known date)
        last_date = series.index.max()
        dates = pd.to_datetime([last_date + timedelta(days=i+1) for i in range(periods)])
        
        return pd.DataFrame({'Forecast': forecast_values.values, 'Lower_CI': conf_int.iloc[:, 0].values, 'Upper_CI': conf_int.iloc[:, 1].values}, index=dates)
    except Exception as e:
        st.error(f"SARIMA forecast error: {str(e)}")
        return None
    
# --- Main Logic ---

# Store current tickers in session state for cross-functionality (like in LSTM optimization)
st.session_state["current_tickers"] = tickers 

if not tickers:
    st.info("👉 Enter stock symbols in the sidebar to begin.")
    st.stop()
    
# --- Single Stock Analysis Mode ---
if mode == "Single Stock Analysis" and len(symbols) == 1:
    ticker, symbol = tickers[0], symbols[0]
    with st.spinner(f"Loading **{ticker}**..."):
        df, info = load_data(symbol, period)
    
    if df is None:
        st.error(f"Could not load **{ticker}**. Check symbol or date range.")
        st.stop()
        
    df = add_technical_indicators(df)
    
    st.markdown(f"## 📈 {ticker} - Detailed Analysis")
    
    # --- Price Metrics (Existing logic is fine) ---
    col1, col2, col3, col4, col5 = st.columns(5)
    if info:
        current_price = info.get("currentPrice", df["Close"].iloc[-1] if not df.empty else "N/A")
        prev_close = info.get("previousClose", df["Close"].iloc[-2] if len(df) >= 2 else "N/A")
        market_cap = info.get("marketCap", "N/A")
        pe_ratio = info.get("trailingPE", "N/A")
        div_yield = info.get("dividendYield", "N/A")
        
        with col1:
            if current_price != "N/A" and prev_close != "N/A" and prev_close != 0: # Fix: check for division by zero
                change_pct = ((current_price - prev_close) / prev_close) * 100
                st.metric("💰 Current Price", f"₹{current_price:.2f}", f"{change_pct:+.2f}%")
            else:
                st.metric("💰 Current Price", f"₹{current_price:.2f}" if isinstance(current_price, (int, float)) else current_price)
        with col2:
            st.metric("📊 Previous Close", f"₹{prev_close:.2f}" if isinstance(prev_close, (int, float)) else prev_close)
        with col3:
            st.metric("🏢 Market Cap", f"₹{market_cap / 10000000:.0f} Cr" if isinstance(market_cap, (int, float)) and market_cap > 10000000 else market_cap)
        with col4:
            st.metric("📈 P/E Ratio", f"{pe_ratio:.2f}" if isinstance(pe_ratio, (int, float)) else pe_ratio)
        with col5:
            st.metric("💸 Dividend Yield", f"{div_yield*100:.2f}%" if isinstance(div_yield, (int, float)) and div_yield is not None else div_yield)
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Price Chart", "📈 Technical Indicators", "🔮 Forecasting", "📰 News & Sentiment", "📋 Data Export"])
    
    # --- Price Chart (Existing logic is fine) ---
    with tab1:
        st.subheader("Price Chart")
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, subplot_titles=('Price Chart', 'Volume'), row_width=[0.7, 0.3])
        fig.add_trace(go.Candlestick(x=df["Date"], open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Price"), row=1, col=1)
        for ma, color in [("SMA20", "orange"), ("SMA50", "blue"), ("EMA20", "red")]:
            if ma in df.columns and df[ma].dropna().shape[0] > 0: # Enhancement: check if indicator has data
                fig.add_trace(go.Scatter(x=df["Date"], y=df[ma], name=ma, line=dict(color=color)), row=1, col=1)
        if all(col in df.columns for col in ["BB_Upper", "BB_Lower"]):
            fig.add_trace(go.Scatter(x=df["Date"], y=df["BB_Upper"], name="BB Upper", line=dict(color="gray", dash="dash")), row=1, col=1)
            fig.add_trace(go.Scatter(x=df["Date"], y=df["BB_Lower"], name="BB Lower", line=dict(color="gray", dash="dash"), fill='tonexty', fillcolor='rgba(128,128,128,0.1)'), row=1, col=1) # Enhancement: BB fill color
        if show_volume and "Volume" in df.columns:
            fig.add_trace(go.Bar(x=df["Date"], y=df["Volume"], name="Volume", marker_color="lightblue"), row=2, col=1)
        fig.update_layout(height=700, title_text=f"**{ticker}** - Price and Volume", template="plotly_white") # Enhancement: Template
        fig.update_xaxes(rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)
    
    # --- Technical Indicators (Existing logic is fine) ---
    with tab2:
        st.subheader("Technical Indicators")
        indicator_tabs = st.tabs(["RSI", "MACD", "Stochastic", "ATR"])
        # ... (RSI, MACD, Stoch, ATR chart generation logic) ...
        with indicator_tabs[0]:
            if "RSI" in df.columns and df["RSI"].dropna().shape[0] > 0:
                fig_rsi = go.Figure()
                fig_rsi.add_trace(go.Scatter(x=df["Date"], y=df["RSI"], name="RSI"))
                fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought (70)")
                fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold (30)")
                fig_rsi.update_layout(title="RSI", yaxis_title="RSI", template="plotly_white")
                st.plotly_chart(fig_rsi, use_container_width=True)
            else:
                st.info("RSI data not available (Need 14+ data points)")
        with indicator_tabs[1]:
            if all(col in df.columns for col in ["MACD", "Signal", "MACD_Histogram"]) and df["MACD"].dropna().shape[0] > 0:
                fig_macd = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1)
                fig_macd.add_trace(go.Scatter(x=df["Date"], y=df["MACD"], name="MACD", line=dict(color="blue")), row=1, col=1)
                fig_macd.add_trace(go.Scatter(x=df["Date"], y=df["Signal"], name="Signal", line=dict(color="red")), row=1, col=1)
                fig_macd.add_trace(go.Bar(x=df["Date"], y=df["MACD_Histogram"], name="Histogram", marker_color=np.where(df["MACD_Histogram"] >= 0, 'green', 'red')), row=2, col=1)
                fig_macd.update_layout(title="MACD Analysis", template="plotly_white")
                st.plotly_chart(fig_macd, use_container_width=True)
            else:
                st.info("MACD data not available (Need 26+ data points)")
        with indicator_tabs[2]:
            if all(col in df.columns for col in ["Stoch_K", "Stoch_D"]) and df["Stoch_K"].dropna().shape[0] > 0:
                fig_stoch = go.Figure()
                fig_stoch.add_trace(go.Scatter(x=df["Date"], y=df["Stoch_K"], name="%K", line=dict(color="blue")), showlegend=True)
                fig_stoch.add_trace(go.Scatter(x=df["Date"], y=df["Stoch_D"], name="%D", line=dict(color="red")), showlegend=True)
                fig_stoch.add_hline(y=80, line_dash="dash", line_color="red", annotation_text="Overbought (80)")
                fig_stoch.add_hline(y=20, line_dash="dash", line_color="green", annotation_text="Oversold (20)")
                fig_stoch.update_layout(title="Stochastic Oscillator", yaxis_title="Stochastic", template="plotly_white")
                st.plotly_chart(fig_stoch, use_container_width=True)
            else:
                st.info("Stochastic data not available (Need 14+ data points)")
        with indicator_tabs[3]:
            if "ATR" in df.columns and df["ATR"].dropna().shape[0] > 0:
                fig_atr = go.Figure()
                fig_atr.add_trace(go.Scatter(x=df["Date"], y=df["ATR"], name="ATR"))
                fig_atr.update_layout(title="Average True Range (ATR)", yaxis_title="ATR", template="plotly_white")
                st.plotly_chart(fig_atr, use_container_width=True)
            else:
                st.info("ATR data not available (Need 14+ data points)")
    
    # --- Forecasting (Existing logic is fine, with added robustness) ---
    with tab3:
        if forecast_method != "None":
            st.subheader(f"🔮 {forecast_method} Forecast ({forecast_days} days)")
            with st.spinner(f"Generating {forecast_method} forecast..."):
                result = None
                df2 = df.set_index("Date").copy() # Base for ARIMA/SARIMA
                
                if forecast_method == "Prophet":
                    dfp = df.rename(columns={"Date": "ds", "Close": "y"})
                    result = forecast_prophet(dfp, forecast_days)
                elif forecast_method == "LSTM":
                    result = forecast_lstm(df, forecast_days)
                elif forecast_method == "ARIMA":
                    result = forecast_arima(df2, forecast_days)
                elif forecast_method == "SARIMA":
                    result = forecast_sarima(df2, forecast_days)
                    
            if result is not None:
                # Plot
                fig_forecast = go.Figure()
                fig_forecast.add_trace(go.Scatter(x=df["Date"], y=df["Close"], name="Historical Price", line=dict(color="blue")))
                fig_forecast.add_trace(go.Scatter(x=result.index, y=result["Forecast"], name=f"{forecast_method} Forecast", line=dict(color="red", dash="dash")))
                if "Lower_CI" in result.columns and "Upper_CI" in result.columns:
                    # Fix: Ensure indices are sorted for fill area
                    sorted_dates = result.index.tolist()
                    fill_x = sorted_dates + sorted_dates[::-1]
                    fill_y = result["Upper_CI"].tolist() + result["Lower_CI"].tolist()[::-1]
                    fig_forecast.add_trace(go.Scatter(x=fill_x, y=fill_y, fill='toself', fillcolor='rgba(255, 0, 0, 0.1)', line=dict(color='rgba(255,255,255,0)'), name='Confidence Interval'))
                
                fig_forecast.update_layout(title=f"**{ticker}** - {forecast_method} Forecast", xaxis_title="Date", yaxis_title="Price (₹)", height=500, template="plotly_white")
                st.plotly_chart(fig_forecast, use_container_width=True)
                
                # Metrics
                st.subheader("📊 Forecast Statistics")
                current_price = df["Close"].iloc[-1]
                forecast_price = result["Forecast"].iloc[-1]
                price_change_pct = ((forecast_price - current_price) / current_price) * 100
                max_forecast = result["Forecast"].max()
                max_change_pct = ((max_forecast - current_price) / current_price) * 100
                min_forecast = result["Forecast"].min()
                min_change_pct = ((min_forecast - current_price) / current_price) * 100
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(f"Price in {forecast_days} days", f"₹{forecast_price:.2f}", f"{price_change_pct:+.2f}%")
                with col2:
                    st.metric("Forecast High", f"₹{max_forecast:.2f}", f"{max_change_pct:+.2f}%")
                with col3:
                    st.metric("Forecast Low", f"₹{min_forecast:.2f}", f"{min_change_pct:+.2f}%")
                    
                overlap = result.join(df2[["Close"]], how="inner")
                if not overlap.empty and len(overlap) > 1:
                    st.subheader("🎯 Model Accuracy (Historical Overlap)")
                    metrics = compute_metrics(overlap["Close"], overlap["Forecast"])
                    metric_cols = st.columns(len(metrics))
                    for i, (name, value) in enumerate(metrics.items()):
                        with metric_cols[i]:
                            if isinstance(value, float) and not np.isnan(value):
                                st.metric(name, f"{value:.4f}" if name == 'R²' else f"{value:.2f}%" if '%' in name else f"{value:.2f}")
                            else:
                                st.metric(name, "N/A")

                st.subheader("📥 Export Forecast Data")
                create_download_buttons(result, f"{ticker}_forecast_{forecast_method.lower()}")
            else:
                st.error(f"Could not generate {forecast_method} forecast.")
        else:
            st.info("Select a forecasting method in the sidebar.")
            
    # --- News & Sentiment (Existing logic is fine, with added check for VADER) ---
    with tab4:
        if show_news and SentimentIntensityAnalyzer:
            st.subheader("📰 News & Sentiment")
            with st.spinner("Fetching news..."):
                news_items = get_news(symbol)
            if news_items:
                try:
                    sid = SentimentIntensityAnalyzer()
                    sentiment_data = []
                    for title, link in news_items:
                        score = sid.polarity_scores(title)
                        compound = score["compound"]
                        sentiment = "🟢 Positive" if compound > 0.05 else "🔴 Negative" if compound < -0.05 else "⚪ Neutral"
                        sentiment_data.append({"Title": title, "Sentiment": sentiment, "Score": compound, "Link": link})
                        
                    for item in sentiment_data:
                        with st.container():
                            col1, col2 = st.columns([4, 1])
                            with col1:
                                st.markdown(f"**[{item['Title']}]({item['Link']})**")
                            with col2:
                                st.markdown(f"**{item['Sentiment']}** ({item['Score']:.2f})")
                            st.markdown("---")
                            
                    # Sentiment Summary (existing logic is fine)
                    if sentiment_data:
                        scores = [item["Score"] for item in sentiment_data]
                        avg_sentiment = np.mean(scores)
                        st.subheader("📊 Sentiment Summary")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("🟢 Positive News", sum(1 for s in scores if s > 0.05))
                        with col2:
                            st.metric("⚪ Neutral News", sum(1 for s in scores if -0.05 <= s <= 0.05))
                        with col3:
                            st.metric("🔴 Negative News", sum(1 for s in scores if s < -0.05))
                        
                        fig_sentiment = go.Figure(go.Bar(x=["Positive", "Neutral", "Negative"], y=[sum(1 for s in scores if s > 0.05), sum(1 for s in scores if -0.05 <= s <= 0.05), sum(1 for s in scores if s < -0.05)], marker_color=["green", "gray", "red"]))
                        fig_sentiment.update_layout(title="News Sentiment Distribution", xaxis_title="Sentiment", yaxis_title="Articles", template="plotly_white")
                        st.plotly_chart(fig_sentiment, use_container_width=True)
                        
                        sentiment_interpretation = "Very Positive" if avg_sentiment > 0.3 else "Positive" if avg_sentiment > 0.05 else "Neutral" if avg_sentiment >= -0.05 else "Negative" if avg_sentiment > -0.3 else "Very Negative"
                        st.info(f"**Average Sentiment**: {avg_sentiment:.3f} ({sentiment_interpretation})")
                except Exception as e:
                    st.error(f"Sentiment analysis error: {str(e)}")
            else:
                st.info("No recent news found.")
        elif not show_news:
            st.info("Enable 'Show news & sentiment' in sidebar.")
        elif not SentimentIntensityAnalyzer:
            st.error("Sentiment Analyzer (NLTK VADER) is not available.")

    # --- Data Export (Existing logic is fine) ---
    with tab5:
        st.subheader("📋 Data Export")
        st.markdown("### Data Preview")
        st.dataframe(df.tail(10), use_container_width=True) # Enhancement: use_container_width
        export_options = st.multiselect("Select data to export:", ["Price Data", "Technical Indicators", "News Sentiment"], default=["Price Data"])
        
        if "Price Data" in export_options:
            st.markdown("#### 📊 Price Data Export")
            create_download_buttons(df[["Date", "Open", "High", "Low", "Close", "Volume"]].set_index("Date"), f"{ticker}_price_data") # Fix: Set Date as index for better export
        
        if "Technical Indicators" in export_options:
            st.markdown("#### 📈 Technical Indicators Export")
            tech_columns = [col for col in df.columns if col not in ["Date", "Open", "High", "Low", "Close", "Volume", "Adj Close"]] # Fix: Exclude 'Adj Close'
            create_download_buttons(df.set_index("Date")[tech_columns], f"{ticker}_technical_indicators") # Fix: Set Date as index
            
        if "News Sentiment" in export_options and show_news and 'sentiment_data' in locals():
            st.markdown("#### 📰 News Sentiment Export")
            create_download_buttons(pd.DataFrame(sentiment_data), f"{ticker}_news_sentiment")
        
# --- Multi-Stock Comparison Mode ---
elif mode == "Multi-Stock Comparison" and len(symbols) > 1:
    st.markdown(f"## 📈 Multi-Stock Comparison\n**Analyzing**: {', '.join(tickers)}")
    
    with st.spinner("Loading and processing multiple stocks..."):
        all_data = {}
        stock_infos = {}
        for ticker, symbol in zip(tickers, symbols):
            df, info = load_data(symbol, period)
            if df is not None:
                # Store the data and compute indicators for performance metrics
                df = add_technical_indicators(df)
                all_data[ticker] = df
                stock_infos[ticker] = info
                
    if not all_data:
        st.error("Could not load any stocks.")
        st.stop()
        
    comp_tab1, comp_tab2, comp_tab3, comp_tab4 = st.tabs(["📊 Price Comparison", "📈 Performance Metrics", "🔮 Forecast Comparison", "📋 Export Data"])
    
    # --- Price Comparison (Existing logic is fine) ---
    with comp_tab1:
        st.subheader("Price Comparison")
        normalize_prices = st.checkbox("Normalize prices (% change from start)", value=True)
        fig_comp = go.Figure()
        for ticker, df in all_data.items():
            if not df.empty:
                y_data = (df["Close"] / df["Close"].iloc[0] - 1) * 100 if normalize_prices else df["Close"]
                fig_comp.add_trace(go.Scatter(x=df["Date"], y=y_data, mode="lines", name=ticker))
                
        y_title = "Percentage Change (%)" if normalize_prices else "Price (₹)"
        fig_comp.update_layout(title="Stock Price Comparison", xaxis_title="Date", yaxis_title=y_title, height=500, template="plotly_white")
        st.plotly_chart(fig_comp, use_container_width=True)
        
        st.subheader("Volume Comparison")
        fig_vol = go.Figure()
        for ticker, df in all_data.items():
             if not df.empty and "Volume" in df.columns:
                fig_vol.add_trace(go.Scatter(x=df["Date"], y=df["Volume"], mode="lines", name=f"{ticker} Volume"))
                
        fig_vol.update_layout(title="Volume Comparison", xaxis_title="Date", yaxis_title="Volume", height=400, template="plotly_white")
        st.plotly_chart(fig_vol, use_container_width=True)
        
    # --- Performance Metrics (Existing logic is fine) ---
    with comp_tab2:
        st.subheader("Performance Metrics")
        performance_data = []
        for ticker, df in all_data.items():
            if len(df) > 1:
                # ... (metric calculation is fine)
                current_price = df["Close"].iloc[-1]
                first_price = df["Close"].iloc[0]
                total_return = ((current_price - first_price) / first_price) * 100
                volatility = df["Close"].pct_change().std() * np.sqrt(252) * 100
                max_price = df["Close"].max()
                min_price = df["Close"].min()
                avg_volume = df["Volume"].mean()
                risk_adjusted_return = total_return / volatility if volatility != 0 else 0
                performance_data.append({
                    "Stock": ticker, "Current Price (₹)": f"{current_price:.2f}", "Total Return (%)": f"{total_return:.2f}",
                    "Volatility (%)": f"{volatility:.2f}", "Max Price (₹)": f"{max_price:.2f}", "Min Price (₹)": f"{min_price:.2f}",
                    "Avg Volume": f"{avg_volume:,.0f}", "Risk-Adj Return": f"{risk_adjusted_return:.2f}"
                })
        
        if performance_data:
            perf_df = pd.DataFrame(performance_data)
            st.dataframe(perf_df, use_container_width=True)
            
            # Subplots visualization (existing logic is fine)
            fig_perf = make_subplots(rows=2, cols=2, subplot_titles=('Total Return (%)', 'Volatility (%)', 'Risk-Adjusted Return', 'Average Volume'))
            stocks = [item["Stock"] for item in performance_data]
            returns = [float(item["Total Return (%)"].replace('%', '')) for item in performance_data]
            volatilities = [float(item["Volatility (%)"].replace('%', '')) for item in performance_data]
            risk_adj = [float(item["Risk-Adj Return"]) for item in performance_data]
            volumes = [float(item["Avg Volume"].replace(',', '')) for item in performance_data]
            
            fig_perf.add_trace(go.Bar(x=stocks, y=returns, name="Return", marker_color="green"), row=1, col=1)
            fig_perf.add_trace(go.Bar(x=stocks, y=volatilities, name="Volatility", marker_color="red"), row=1, col=2)
            fig_perf.add_trace(go.Bar(x=stocks, y=risk_adj, name="Risk-Adj Return", marker_color="blue"), row=2, col=1)
            fig_perf.add_trace(go.Bar(x=stocks, y=volumes, name="Avg Volume", marker_color="orange"), row=2, col=2)
            
            fig_perf.update_layout(height=600, showlegend=False, title_text="Performance Metrics Comparison", template="plotly_white")
            st.plotly_chart(fig_perf, use_container_width=True)
    
    # --- Forecast Comparison (Completion of the existing unfinished block) ---
    with comp_tab3:
        if forecast_method != "None":
            st.subheader(f"🔮 Forecast Comparison: {forecast_method} ({forecast_days} days)")
            forecast_results = {}
            forecast_performance = []
            
            with st.spinner(f"Generating {forecast_method} forecasts for all stocks... (This may take a moment for complex models like LSTM)"):
                for ticker, df in all_data.items():
                    result = None
                    df2 = df.set_index("Date").copy()
                    
                    if forecast_method == "Prophet":
                        dfp = df.rename(columns={"Date": "ds", "Close": "y"})
                        result = forecast_prophet(dfp, forecast_days)
                    elif forecast_method == "LSTM":
                        # Enhancement: Use fewer epochs for speed in multi-stock mode
                        result = forecast_lstm(df, forecast_days, epochs=10) 
                    elif forecast_method == "ARIMA":
                        result = forecast_arima(df2, forecast_days)
                    elif forecast_method == "SARIMA":
                        result = forecast_sarima(df2, forecast_days)
                        
                    if result is not None and not result.empty:
                        forecast_results[ticker] = result
                        
                        # Calculate Forecast Metrics for Comparison Table
                        current_price = df["Close"].iloc[-1]
                        forecast_price = result["Forecast"].iloc[-1]
                        price_change_pct = ((forecast_price - current_price) / current_price) * 100
                        forecast_performance.append({
                            "Stock": ticker,
                            "Last Close (₹)": f"{current_price:.2f}",
                            f"{forecast_days} Day Forecast (₹)": f"{forecast_price:.2f}",
                            "Price Change (%)": f"{price_change_pct:+.2f}",
                            "Forecast High (₹)": f"{result['Forecast'].max():.2f}",
                            "Forecast Low (₹)": f"{result['Forecast'].min():.2f}"
                        })

            if forecast_results:
                # Comparison Table
                st.subheader("Summary Table")
                perf_df = pd.DataFrame(forecast_performance).set_index("Stock")
                st.dataframe(perf_df, use_container_width=True)
                
                # Comparison Chart
                st.subheader("Comparative Price Trend and Forecast")
                fig_comp_forecast = go.Figure()
                
                # Plot Historical and Forecast in one chart
                max_hist_date = max(df["Date"].max() for df in all_data.values())
                
                for i, (ticker, result) in enumerate(forecast_results.items()):
                    hist_df = all_data[ticker]
                    # Only show historical data up to the last max date for cleaner look
                    hist_df_filtered = hist_df[hist_df["Date"] <= max_hist_date] 
                    
                    # Historical Trace
                    fig_comp_forecast.add_trace(go.Scatter(
                        x=hist_df_filtered["Date"], y=hist_df_filtered["Close"], 
                        mode="lines", name=f"{ticker} (Historical)", 
                        line=dict(dash='solid'), opacity=0.8
                    ))
                    
                    # Forecast Trace
                    fig_comp_forecast.add_trace(go.Scatter(
                        x=result.index, y=result["Forecast"], 
                        mode="lines", name=f"{ticker} (Forecast)", 
                        line=dict(dash='dash'), opacity=0.9
                    ))

                fig_comp_forecast.update_layout(
                    title=f"Multi-Stock {forecast_method} Forecast Trend", 
                    xaxis_title="Date", 
                    yaxis_title="Price (₹)", 
                    height=600, 
                    template="plotly_white"
                )
                st.plotly_chart(fig_comp_forecast, use_container_width=True)

            else:
                st.info("Could not generate a forecast for any of the selected stocks. Try adjusting the period or using a simpler method like ARIMA.")

        else:
            st.info("Select a forecasting method in the sidebar to enable comparison.")
            
    # --- Data Export (New section for multi-stock) ---
    with comp_tab4:
        st.subheader("📋 Data Export - Multi-Stock")
        
        # Merge all data for export
        merge_key = "Date"
        all_df = all_data[tickers[0]].set_index(merge_key)[["Close", "Volume"]].rename(columns={"Close": f"{tickers[0]}_Close", "Volume": f"{tickers[0]}_Volume"})
        
        for i, ticker in enumerate(tickers[1:]):
            df = all_data[ticker]
            if not df.empty:
                df_to_merge = df.set_index(merge_key)[["Close", "Volume"]].rename(columns={"Close": f"{ticker}_Close", "Volume": f"{ticker}_Volume"})
                all_df = all_df.join(df_to_merge, how="outer")
        
        st.markdown("### Combined Historical Data Preview")
        st.dataframe(all_df.tail(10), use_container_width=True)
        
        st.markdown("#### 📥 Combined Data Export")
        create_download_buttons(all_df, "multi_stock_comparison_data")
        
        if forecast_results:
            st.markdown("#### 📥 Combined Forecast Export")
            forecast_df = pd.DataFrame()
            for ticker, result in forecast_results.items():
                temp_df = result[["Forecast"]].rename(columns={"Forecast": f"{ticker}_Forecast"})
                if forecast_df.empty:
                    forecast_df = temp_df
                else:
                    forecast_df = forecast_df.join(temp_df, how="outer")
            
            st.dataframe(forecast_df.tail(10), use_container_width=True)
            create_download_buttons(forecast_df, f"multi_stock_forecast_{forecast_method.lower()}")

# --- Portfolio Analysis Mode (Placeholder Enhancement) ---
elif mode == "Portfolio Analysis":
    st.markdown("## 💰 Portfolio Analysis (Coming Soon!)")
    if len(symbols) > 1:
        st.info("This mode will feature **Portfolio Optimization** (e.g., Markowitz Portfolio Theory), **Risk & Return Metrics** (e.g., Sharpe Ratio), and **Beta Calculation** for the selected stocks.")
    else:
        st.warning("Please select at least two stocks in the sidebar to enable a meaningful Portfolio Analysis.")

# --- Error/Info for Single Stock Mode with Multiple Tickers ---
elif mode == "Single Stock Analysis" and len(symbols) > 1:
    st.warning("You have selected **Multi-Stock Comparison** mode but need to switch the 'Analysis Mode' to **Multi-Stock Comparison** in the sidebar, or reduce your selection to one stock.")

# --- End of Script ---
