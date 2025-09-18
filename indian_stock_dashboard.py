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
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
warnings.filterwarnings('ignore')

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

try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    with st.spinner("Downloading sentiment data..."):
        nltk.download("vader_lexicon")

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
    mode = st.radio("Select mode", ["Single Stock Analysis", "Multi-Stock Comparison", "Portfolio Analysis"])
    st.markdown("### 🔮 Forecasting")
    forecast_method = st.selectbox("Forecast method", ["None", "Prophet", "LSTM", "ARIMA", "SARIMA"])
    forecast_days = st.slider("Forecast horizon (days)", 5, 365, 30, step=5)
    st.markdown("### ⚙️ Settings")
    period_options = {"1 Month": "1mo", "3 Months": "3mo", "6 Months": "6mo", "1 Year": "1y", "2 Years": "2y", "5 Years": "5y"}
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

@st.cache_data(ttl=300)
def load_data(ticker, period):
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)
        if df.empty:
            return None, None
        df.reset_index(inplace=True)
        df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
        return df, stock.info
    except Exception as e:
        st.error(f"Error loading {ticker}: {str(e)}")
        return None, None

@st.cache_data(ttl=1800)
def get_news(ticker, max_articles=10):
    url = f"https://in.finance.yahoo.com/quote/{ticker}/news?p={ticker}"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/91.0.4472.124"}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            return []
        soup = BeautifulSoup(response.text, 'html.parser')
        selectors = ['h3[class*="Mb(5px)"]', 'h3', '[data-test-locator="mega-item-header"]', '.js-content-viewer h3']
        articles = []
        for selector in selectors:
            articles = soup.select(selector)
            if articles:
                break
        news = []
        for article in articles[:max_articles]:
            try:
                a_tag = article.find('a')
                if a_tag and a_tag.text and a_tag.has_attr('href'):
                    title = a_tag.text.strip()
                    link = a_tag['href']
                    if not link.startswith('http'):
                        link = "https://in.finance.yahoo.com" + link
                    news.append((title, link))
            except:
                continue
        return news
    except Exception as e:
        st.warning(f"News fetch failed: {str(e)}")
        return []

def add_technical_indicators(df):
    df = df.copy()
    df["SMA20"] = df["Close"].rolling(window=20).mean()
    df["SMA50"] = df["Close"].rolling(window=50).mean()
    df["SMA200"] = df["Close"].rolling(window=200).mean()
    df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()
    rolling_mean = df["Close"].rolling(window=20).mean()
    rolling_std = df["Close"].rolling(window=20).std()
    df["BB_Upper"] = rolling_mean + (rolling_std * 2)
    df["BB_Lower"] = rolling_mean - (rolling_std * 2)
    df["BB_Middle"] = rolling_mean
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Histogram"] = df["MACD"] - df["Signal"]
    low_min = df["Low"].rolling(window=14).min()
    high_max = df["High"].rolling(window=14).max()
    df["Stoch_K"] = 100 * ((df["Close"] - low_min) / (high_max - low_min))
    df["Stoch_D"] = df["Stoch_K"].rolling(window=3).mean()
    high_low = df["High"] - df["Low"]
    high_close = np.abs(df["High"] - df["Close"].shift())
    low_close = np.abs(df["Low"] - df["Close"].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["ATR"] = true_range.rolling(window=14).mean()
    return df

def compute_metrics(actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    actual_arr, pred_arr = np.array(actual), np.array(predicted)
    mask = actual_arr != 0
    mape = (np.abs((actual_arr[mask] - pred_arr[mask]) / actual_arr[mask]).mean() * 100) if mask.sum() else float("nan")
    ss_res = np.sum((actual_arr - pred_arr) ** 2)
    ss_tot = np.sum((actual_arr - np.mean(actual_arr)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    actual_direction = np.diff(actual_arr) > 0
    pred_direction = np.diff(pred_arr) > 0
    direction_accuracy = np.mean(actual_direction == pred_direction) * 100 if len(actual_direction) > 0 else 0
    return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape, 'R²': r2, 'Direction Accuracy (%)': direction_accuracy}

def create_download_buttons(df, filename_prefix):
    col1, col2 = st.columns(2)
    with col1:
        csv = df.to_csv().encode("utf-8")
        st.download_button("📥 Download as CSV", csv, f"{filename_prefix}_{datetime.now().strftime('%Y%m%d')}.csv", mime="text/csv")
    with col2:
        excel_buffer = BytesIO()
        df_export = df.copy()
        if isinstance(df_export.index, pd.DatetimeIndex) and df_export.index.tz is not None:
            df_export.index = df_export.index.tz_localize(None)
        for col in df_export.select_dtypes(include=["datetimetz"]).columns:
            df_export[col] = df_export[col].dt.tz_localize(None)
        with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
            df_export.to_excel(writer, index=True, sheet_name="Data")
        st.download_button("📥 Download as Excel", excel_buffer.getvalue(), f"{filename_prefix}_{datetime.now().strftime('%Y%m%d')}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

def forecast_prophet(df, periods):
    try:
        from prophet import Prophet
        fdf = df[["ds", "y"]].dropna().copy()
        fdf["ds"] = pd.to_datetime(fdf["ds"]).dt.tz_localize(None)
        if fdf.empty or len(fdf) < 10:
            return None
        m = Prophet(daily_seasonality=False, yearly_seasonality=True, weekly_seasonality=True)
        m.fit(fdf)
        future = m.make_future_dataframe(periods=periods)
        forecast = m.predict(future)
        return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].rename(columns={"ds": "Date", "yhat": "Forecast", "yhat_lower": "Lower_CI", "yhat_upper": "Upper_CI"}).set_index("Date")
    except ImportError:
        st.error("Prophet not installed. Install with: pip install prophet")
        return None
    except Exception as e:
        st.error(f"Prophet forecast error: {str(e)}")
        return None

def forecast_lstm(df, periods, epochs=50, batch_size=32):
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, LSTM, Dropout
        from sklearn.preprocessing import MinMaxScaler
        series = df["Close"].dropna().values.reshape(-1, 1)
        if len(series) < 60:
            st.warning("Insufficient data for LSTM (minimum 60 data points)")
            return None
        scaler = MinMaxScaler((0, 1))
        scaled = scaler.fit_transform(series)
        seq_len = min(60, len(scaled) // 4)
        X, y = [], []
        for i in range(seq_len, len(scaled)):
            X.append(scaled[i-seq_len:i, 0])
            y.append(scaled[i, 0])
        if not X:
            return None
        X, y = np.array(X), np.array(y)
        X = X.reshape((X.shape[0], X.shape[1], 1))
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        progress_bar = st.progress(0)
        class ProgressCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                progress_bar.progress((epoch + 1) / epochs)
        model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0, callbacks=[ProgressCallback()])
        progress_bar.empty()
        last_seq = scaled[-seq_len:].reshape(1, seq_len, 1)
        preds = []
        cur_seq = last_seq.copy()
        for _ in range(periods):
            pred = model.predict(cur_seq, verbose=0)[0, 0]
            preds.append(pred)
            cur_seq = np.roll(cur_seq, -1, axis=1)
            cur_seq[0, -1, 0] = pred
        preds = scaler.inverse_transform(np.array(preds).reshape(-1, 1))
        last_date = df["Date"].max()
        dates = [last_date + timedelta(days=i+1) for i in range(periods)]
        return pd.DataFrame(preds.flatten(), index=dates, columns=["Forecast"])
    except ImportError:
        st.error("TensorFlow not installed. Install with: pip install tensorflow")
        return None
    except Exception as e:
        st.error(f"LSTM forecast error: {str(e)}")
        return None

def forecast_arima(df, periods, order=(5,1,0)):
    try:
        series = df["Close"].dropna()
        if series.empty or len(series) < 50:
            st.warning("Insufficient data for ARIMA")
            return None
        model = ARIMA(series, order=order)
        fit = model.fit()
        forecast = fit.get_forecast(steps=periods)
        forecast_values = forecast.predicted_mean
        conf_int = forecast.conf_int()
        dates = [series.index.max() + timedelta(days=i+1) for i in range(periods)]
        return pd.DataFrame({'Forecast': forecast_values.values, 'Lower_CI': conf_int.iloc[:, 0].values, 'Upper_CI': conf_int.iloc[:, 1].values}, index=dates)
    except Exception as e:
        st.error(f"ARIMA forecast error: {str(e)}")
        return None

def forecast_sarima(df, periods, order=(2,1,2), seasonal_order=(1,1,1,12)):
    try:
        series = df["Close"].dropna()
        if series.empty or len(series) < 100:
            st.warning("Insufficient data for SARIMA")
            return None
        model = SARIMAX(series, order=order, seasonal_order=seasonal_order)
        fit = model.fit(disp=False)
        forecast = fit.get_forecast(steps=periods)
        forecast_values = forecast.predicted_mean
        conf_int = forecast.conf_int()
        dates = [series.index.max() + timedelta(days=i+1) for i in range(periods)]
        return pd.DataFrame({'Forecast': forecast_values.values, 'Lower_CI': conf_int.iloc[:, 0].values, 'Upper_CI': conf_int.iloc[:, 1].values}, index=dates)
    except Exception as e:
        st.error(f"SARIMA forecast error: {str(e)}")
        return None

if not tickers:
    st.info("👉 Enter stock symbols in sidebar.")
    st.stop()

if mode == "Single Stock Analysis" and len(symbols) == 1:
    ticker, symbol = tickers[0], symbols[0]
    with st.spinner(f"Loading {ticker}..."):
        df, info = load_data(symbol, period)
    if df is None:
        st.error(f"Could not load {ticker}. Check symbol.")
        st.stop()
    df = add_technical_indicators(df)
    st.markdown(f"## 📈 {ticker} - Detailed Analysis")
    col1, col2, col3, col4, col5 = st.columns(5)
    if info:
        current_price = info.get("currentPrice", "N/A")
        prev_close = info.get("previousClose", "N/A")
        market_cap = info.get("marketCap", "N/A")
        pe_ratio = info.get("trailingPE", "N/A")
        div_yield = info.get("dividendYield", "N/A")
        with col1:
            if current_price != "N/A" and prev_close != "N/A":
                change_pct = ((current_price - prev_close) / prev_close) * 100
                st.metric("💰 Current Price", f"₹{current_price:.2f}", f"{change_pct:+.2f}%")
            else:
                st.metric("💰 Current Price", current_price)
        with col2:
            st.metric("📊 Previous Close", f"₹{prev_close:.2f}" if prev_close != "N/A" else prev_close)
        with col3:
            st.metric("🏢 Market Cap", f"₹{market_cap / 10000000:.0f} Cr" if market_cap != "N/A" else market_cap)
        with col4:
            st.metric("📈 P/E Ratio", f"{pe_ratio:.2f}" if pe_ratio != "N/A" else pe_ratio)
        with col5:
            st.metric("💸 Dividend Yield", f"{div_yield*100:.2f}%" if div_yield != "N/A" else div_yield)
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Price Chart", "📈 Technical Indicators", "🔮 Forecasting", "📰 News & Sentiment", "📋 Data Export"])
    with tab1:
        st.subheader("Price Chart")
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, subplot_titles=('Price Chart', 'Volume'), row_width=[0.7, 0.3])
        fig.add_trace(go.Candlestick(x=df["Date"], open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Price"), row=1, col=1)
        for ma, color in [("SMA20", "orange"), ("SMA50", "blue"), ("EMA20", "red")]:
            if ma in df.columns:
                fig.add_trace(go.Scatter(x=df["Date"], y=df[ma], name=ma, line=dict(color=color)), row=1, col=1)
        if all(col in df.columns for col in ["BB_Upper", "BB_Lower"]):
            fig.add_trace(go.Scatter(x=df["Date"], y=df["BB_Upper"], name="BB Upper", line=dict(color="gray", dash="dash")), row=1, col=1)
            fig.add_trace(go.Scatter(x=df["Date"], y=df["BB_Lower"], name="BB Lower", line=dict(color="gray", dash="dash"), fill='tonexty'), row=1, col=1)
        if show_volume and "Volume" in df.columns:
            fig.add_trace(go.Bar(x=df["Date"], y=df["Volume"], name="Volume", marker_color="lightblue"), row=2, col=1)
        fig.update_layout(height=700, title_text=f"{ticker} - Price and Volume")
        fig.update_xaxes(rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)
    with tab2:
        st.subheader("Technical Indicators")
        indicator_tabs = st.tabs(["RSI", "MACD", "Stochastic", "ATR"])
        with indicator_tabs[0]:
            if "RSI" in df.columns:
                fig_rsi = go.Figure()
                fig_rsi.add_trace(go.Scatter(x=df["Date"], y=df["RSI"], name="RSI"))
                fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought (70)")
                fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold (30)")
                fig_rsi.update_layout(title="RSI", yaxis_title="RSI")
                st.plotly_chart(fig_rsi, use_container_width=True)
        with indicator_tabs[1]:
            if all(col in df.columns for col in ["MACD", "Signal", "MACD_Histogram"]):
                fig_macd = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1)
                fig_macd.add_trace(go.Scatter(x=df["Date"], y=df["MACD"], name="MACD"), row=1, col=1)
                fig_macd.add_trace(go.Scatter(x=df["Date"], y=df["Signal"], name="Signal"), row=1, col=1)
                fig_macd.add_trace(go.Bar(x=df["Date"], y=df["MACD_Histogram"], name="Histogram"), row=2, col=1)
                fig_macd.update_layout(title="MACD Analysis")
                st.plotly_chart(fig_macd, use_container_width=True)
        with indicator_tabs[2]:
            if all(col in df.columns for col in ["Stoch_K", "Stoch_D"]):
                fig_stoch = go.Figure()
                fig_stoch.add_trace(go.Scatter(x=df["Date"], y=df["Stoch_K"], name="%K"))
                fig_stoch.add_trace(go.Scatter(x=df["Date"], y=df["Stoch_D"], name="%D"))
                fig_stoch.add_hline(y=80, line_dash="dash", line_color="red", annotation_text="Overbought (80)")
                fig_stoch.add_hline(y=20, line_dash="dash", line_color="green", annotation_text="Oversold (20)")
                fig_stoch.update_layout(title="Stochastic Oscillator", yaxis_title="Stochastic")
                st.plotly_chart(fig_stoch, use_container_width=True)
        with indicator_tabs[3]:
            if "ATR" in df.columns:
                fig_atr = go.Figure()
                fig_atr.add_trace(go.Scatter(x=df["Date"], y=df["ATR"], name="ATR"))
                fig_atr.update_layout(title="ATR", yaxis_title="ATR")
                st.plotly_chart(fig_atr, use_container_width=True)
    with tab3:
        if forecast_method != "None":
            st.subheader(f"🔮 {forecast_method} Forecast ({forecast_days} days)")
            with st.spinner(f"Generating {forecast_method} forecast..."):
                result = None
                if forecast_method == "Prophet":
                    dfp = df.rename(columns={"Date": "ds", "Close": "y"})
                    result = forecast_prophet(dfp, forecast_days)
                elif forecast_method == "LSTM":
                    result = forecast_lstm(df, forecast_days)
                elif forecast_method == "ARIMA":
                    df2 = df.set_index("Date")
                    result = forecast_arima(df2, forecast_days)
                elif forecast_method == "SARIMA":
                    df2 = df.set_index("Date")
                    result = forecast_sarima(df2, forecast_days)
            if result is not None:
                fig_forecast = go.Figure()
                fig_forecast.add_trace(go.Scatter(x=df["Date"], y=df["Close"], name="Historical Price", line=dict(color="blue")))
                fig_forecast.add_trace(go.Scatter(x=result.index, y=result["Forecast"], name=f"{forecast_method} Forecast", line=dict(color="red", dash="dash")))
                if "Lower_CI" in result.columns and "Upper_CI" in result.columns:
                    fig_forecast.add_trace(go.Scatter(x=result.index, y=result["Upper_CI"], fill=None, mode='lines', line_color='rgba(0,100,80,0)', showlegend=False))
                    fig_forecast.add_trace(go.Scatter(x=result.index, y=result["Lower_CI"], fill='tonexty', mode='lines', line_color='rgba(0,100,80,0)', name='Confidence Interval'))
                fig_forecast.update_layout(title=f"{ticker} - {forecast_method} Forecast", xaxis_title="Date", yaxis_title="Price (₹)", height=500)
                st.plotly_chart(fig_forecast, use_container_width=True)
                st.subheader("📊 Forecast Statistics")
                col1, col2, col3 = st.columns(3)
                with col1:
                    current_price = df["Close"].iloc[-1]
                    forecast_price = result["Forecast"].iloc[-1]
                    price_change_pct = ((forecast_price - current_price) / current_price) * 100
                    st.metric(f"Price in {forecast_days} days", f"₹{forecast_price:.2f}", f"{price_change_pct:+.2f}%")
                with col2:
                    max_forecast = result["Forecast"].max()
                    max_change_pct = ((max_forecast - current_price) / current_price) * 100
                    st.metric("Forecast High", f"₹{max_forecast:.2f}", f"{max_change_pct:+.2f}%")
                with col3:
                    min_forecast = result["Forecast"].min()
                    min_change_pct = ((min_forecast - current_price) / current_price) * 100
                    st.metric("Forecast Low", f"₹{min_forecast:.2f}", f"{min_change_pct:+.2f}%")
                overlap = result.join(df.set_index("Date")[["Close"]], how="inner")
                if not overlap.empty and len(overlap) > 1:
                    st.subheader("🎯 Model Accuracy")
                    metrics = compute_metrics(overlap["Close"], overlap["Forecast"])
                    metric_cols = st.columns(len(metrics))
                    for i, (name, value) in enumerate(metrics.items()):
                        with metric_cols[i]:
                            if isinstance(value, float):
                                st.metric(name, f"{value:.4f}" if name == 'R²' else f"{value:.2f}%")
                            else:
                                st.metric(name, str(value))
                st.subheader("📥 Export Forecast Data")
                create_download_buttons(result, f"{ticker}_forecast_{forecast_method.lower()}")
            else:
                st.error("Could not generate forecast. Try another method.")
        else:
            st.info("Select a forecasting method.")
    with tab4:
        if show_news:
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
                        fig_sentiment.update_layout(title="News Sentiment Distribution", xaxis_title="Sentiment", yaxis_title="Articles")
                        st.plotly_chart(fig_sentiment, use_container_width=True)
                        sentiment_interpretation = "Very Positive" if avg_sentiment > 0.3 else "Positive" if avg_sentiment > 0.05 else "Neutral" if avg_sentiment >= -0.05 else "Negative" if avg_sentiment > -0.3 else "Very Negative"
                        st.info(f"**Average Sentiment**: {avg_sentiment:.3f} ({sentiment_interpretation})")
                except Exception as e:
                    st.error(f"Sentiment analysis error: {str(e)}")
            else:
                st.info("No recent news found.")
        else:
            st.info("Enable 'Show news & sentiment' in sidebar.")
    with tab5:
        st.subheader("📋 Data Export")
        st.markdown("### Data Preview")
        st.dataframe(df.tail(10))
        export_options = st.multiselect("Select data to export:", ["Price Data", "Technical Indicators", "News Sentiment"], default=["Price Data"])
        if "Price Data" in export_options:
            st.markdown("#### 📊 Price Data Export")
            create_download_buttons(df[["Date", "Open", "High", "Low", "Close", "Volume"]], f"{ticker}_price_data")
        if "Technical Indicators" in export_options:
            st.markdown("#### 📈 Technical Indicators Export")
            tech_columns = [col for col in df.columns if col not in ["Open", "High", "Low", "Close", "Volume"]]
            create_download_buttons(df[tech_columns], f"{ticker}_technical_indicators")
        if "News Sentiment" in export_options and show_news and 'sentiment_data' in locals():
            st.markdown("#### 📰 News Sentiment Export")
            create_download_buttons(pd.DataFrame(sentiment_data), f"{ticker}_news_sentiment")

elif mode == "Multi-Stock Comparison" and len(symbols) > 1:
    st.markdown(f"## 📈 Multi-Stock Comparison\n**Analyzing**: {', '.join(tickers)}")
    with st.spinner("Loading multiple stocks..."):
        all_data = {}
        stock_infos = {}
        for ticker, symbol in zip(tickers, symbols):
            df, info = load_data(symbol, period)
            if df is not None:
                all_data[ticker] = df
                stock_infos[ticker] = info
    if not all_data:
        st.error("Could not load any stocks.")
        st.stop()
    comp_tab1, comp_tab2, comp_tab3, comp_tab4 = st.tabs(["📊 Price Comparison", "📈 Performance Metrics", "🔮 Forecast Comparison", "📋 Export Data"])
    with comp_tab1:
        st.subheader("Price Comparison")
        normalize_prices = st.checkbox("Normalize prices (% change from start)", value=True)
        fig_comp = go.Figure()
        for ticker, df in all_data.items():
            y_data = (df["Close"] / df["Close"].iloc[0] - 1) * 100 if normalize_prices else df["Close"]
            y_title = "Percentage Change (%)" if normalize_prices else "Price (₹)"
            fig_comp.add_trace(go.Scatter(x=df["Date"], y=y_data, mode="lines", name=ticker))
        fig_comp.update_layout(title="Stock Price Comparison", xaxis_title="Date", yaxis_title=y_title, height=500)
        st.plotly_chart(fig_comp, use_container_width=True)
        st.subheader("Volume Comparison")
        fig_vol = go.Figure()
        for ticker, df in all_data.items():
            fig_vol.add_trace(go.Scatter(x=df["Date"], y=df["Volume"], mode="lines", name=f"{ticker} Volume"))
        fig_vol.update_layout(title="Volume Comparison", xaxis_title="Date", yaxis_title="Volume", height=400)
        st.plotly_chart(fig_vol, use_container_width=True)
    with comp_tab2:
        st.subheader("Performance Metrics")
        performance_data = []
        for ticker, df in all_data.items():
            if len(df) > 1:
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
            fig_perf = make_subplots(rows=2, cols=2, subplot_titles=('Total Return (%)', 'Volatility (%)', 'Risk-Adjusted Return', 'Average Volume'), specs=[[{"secondary_y": False}, {"secondary_y": False}], [{"secondary_y": False}, {"secondary_y": False}]])
            stocks = [item["Stock"] for item in performance_data]
            returns = [float(item["Total Return (%)"].replace('%', '')) for item in performance_data]
            volatilities = [float(item["Volatility (%)"].replace('%', '')) for item in performance_data]
            risk_adj = [float(item["Risk-Adj Return"]) for item in performance_data]
            volumes = [float(item["Avg Volume"].replace(',', '')) for item in performance_data]
            fig_perf.add_trace(go.Bar(x=stocks, y=returns, name="Return", marker_color="green"), row=1, col=1)
            fig_perf.add_trace(go.Bar(x=stocks, y=volatilities, name="Volatility", marker_color="red"), row=1, col=2)
            fig_perf.add_trace(go.Bar(x=stocks, y=risk_adj, name="Risk-Adj Return", marker_color="blue"), row=2, col=1)
            fig_perf.add_trace(go.Bar(x=stocks, y=volumes, name="Avg Volume", marker_color="orange"), row=2, col=2)
            fig_perf.update_layout(height=600, showlegend=False, title_text="Performance Metrics Comparison")
            st.plotly_chart(fig_perf, use_container_width=True)
    with comp_tab3:
        if forecast_method != "None":
            st.subheader(f"🔮 Forecast Comparison: {forecast_method}")
            forecast_results = {}
            forecast_performance = []
            with st.spinner("Generating forecasts..."):
                for ticker, df in all_data.items():
                    try:
                        result = None
                        if forecast_method == "Prophet":
                            dfp = df.rename(columns={"Date": "ds", "Close": "y"})
                            result = forecast_prophet(dfp, forecast_days)
                        elif forecast_method == "LSTM":
                            result = forecast_lstm(df, forecast_days, epochs=30)
                        elif forecast_method == "ARIMA":
                            df2 = df.set_index("Date")
                            result = forecast_arima(df2, forecast_days)
                        elif forecast_method == "SARIMA":
                            df2 = df.set_index("Date")
                            result = forecast_sarima(df2, forecast_days)
                        if result is not None:
                            forecast_results[ticker] = result
                            current_price = df["Close"].iloc[-1]
                            forecast_price = result["Forecast"].iloc[-1]
                            forecast_return = ((forecast_price - current_price) / current_price) * 100
                            forecast_performance.append({
                                "Stock": ticker, "Current Price": f"₹{current_price:.2f}",
                                f"Forecast Price ({forecast_days}d)": f"₹{forecast_price:.2f}", "Expected Return (%)": f"{forecast_return:.2f}%"
                            })
                    except Exception as e:
                        st.warning(f"Forecast failed for {ticker}: {str(e)}")
            if forecast_results:
                fig_forecast_comp = go.Figure()
                for ticker, result in forecast_results.items():
                    if ticker in all_data:
                        historical_df = all_data[ticker]
                        fig_forecast_comp.add_trace(go.Scatter(x=historical_df["Date"], y=historical_df["Close"], name=f"{ticker} Historical", line=dict(width=2)))
                        fig_forecast_comp.add_trace(go.Scatter(x=result.index, y=result["Forecast"], name=f"{ticker} Forecast", line=dict(width=2, dash="dash")))
                fig_forecast_comp.update_layout(title=f"Forecast Comparison - {forecast_method}", xaxis_title="Date", yaxis_title="Price (₹)", height=600)
                st.plotly_chart(fig_forecast_comp, use_container_width=True)
                if forecast_performance:
                    st.subheader("📊 Forecast Summary")
                    perf_df = pd.DataFrame(forecast_performance)
                    st.dataframe(perf_df, use_container_width=True)
                    fig_return = go.Figure(data=[go.Bar(x=[item["Stock"] for item in forecast_performance], y=[float(item["Expected Return (%)"].replace('%', '')) for item in forecast_performance], marker_color=['green' if float(item["Expected Return (%)"].replace('%', '')) > 0 else 'red' for item in forecast_performance])])
                    fig_return.update_layout(title=f"Expected Returns ({forecast_days} days)", xaxis_title="Stock", yaxis_title="Expected Return (%)")
                    st.plotly_chart(fig_return, use_container_width=True)
            else:
                st.warning("Could not generate forecasts.")
        else:
            st.info("Select a forecasting method.")
    with comp_tab4:
        st.subheader("📋 Export Comparison Data")
        combined_data = pd.DataFrame()
        for ticker, df in all_data.items():
            price_data = df[["Date", "Close"]].copy().rename(columns={"Close": f"{ticker}_Close"})
            combined_data = price_data if combined_data.empty else combined_data.merge(price_data, on="Date", how="outer")
        if not combined_data.empty:
            st.markdown("### Combined Price Data Preview")
            st.dataframe(combined_data.tail(10))
            create_download_buttons(combined_data, "multi_stock_comparison")
        st.markdown("### Individual Stock Data")
        for ticker, df in all_data.items():
            with st.expander(f"📊 {ticker} Data"):
                st.dataframe(df.tail(5))
                create_download_buttons(df, f"{ticker}_individual_data")
else:
    if mode == "Portfolio Analysis" and len(symbols) > 1:
        st.markdown(f"## 💼 Portfolio Analysis\n**Portfolio Stocks**: {', '.join(tickers)}")
        st.subheader("📊 Portfolio Configuration")
        total_investment = st.number_input("Total Investment Amount (₹)", min_value=1000, value=100000, step=1000)
        weights = {}
        col_count = min(len(tickers), 4)
        cols = st.columns(col_count)
        for i, ticker in enumerate(tickers):
            with cols[i % col_count]:
                weights[ticker] = st.slider(f"{ticker} Weight (%)", 0, 100, 100//len(tickers), key=f"weight_{ticker}") / 100
        total_weight = sum(weights.values())
        if abs(total_weight - 1.0) > 0.01:
            st.warning(f"⚠️ Total weight is {total_weight*100:.1f}%. Normalizing to 100%.")
            weights = {k: v/total_weight for k, v in weights.items()}
        try:
            with st.spinner("Loading portfolio data..."):
                portfolio_data = {}
                for ticker, symbol in zip(tickers, symbols):
                    df, info = load_data(symbol, period)
                    if df is not None:
                        portfolio_data[ticker] = df
            if portfolio_data:
                portfolio_tabs = st.tabs(["📈 Portfolio Performance", "📊 Risk Analysis", "📋 Holdings Summary"])
                with portfolio_tabs[0]:
                    st.subheader("Portfolio Performance")
                    portfolio_dates, portfolio_values = None, None
                    for ticker, df in portfolio_data.items():
                        weight = weights.get(ticker, 0)
                        if weight > 0:
                            if portfolio_dates is None:
                                portfolio_dates = df["Date"]
                                portfolio_values = df["Close"] * weight
                            else:
                                temp_df = pd.DataFrame({"Date": portfolio_dates, "Value": portfolio_values})
                                stock_df = pd.DataFrame({"Date": df["Date"], "Stock_Value": df["Close"] * weight})
                                merged = temp_df.merge(stock_df, on="Date", how="inner")
                                portfolio_dates = merged["Date"]
                                portfolio_values = merged["Value"] + merged["Stock_Value"]
                    if portfolio_values is not None:
                        initial_value = portfolio_values.iloc[0]
                        portfolio_values = (portfolio_values / initial_value) * total_investment
                        fig_portfolio = go.Figure()
                        fig_portfolio.add_trace(go.Scatter(x=portfolio_dates, y=portfolio_values, name="Portfolio Value", line=dict(color="blue", width=3)))
                        for ticker, df in portfolio_data.items():
                            if weights.get(ticker, 0) > 0:
                                stock_investment = total_investment * weights[ticker]
                                normalized_stock = (df["Close"] / df["Close"].iloc[0]) * stock_investment
                                fig_portfolio.add_trace(go.Scatter(x=df["Date"], y=normalized_stock, name=f"{ticker} (Individual)", line=dict(dash="dot")))
                        fig_portfolio.update_layout(title="Portfolio vs Individual Stock Performance", xaxis_title="Date", yaxis_title="Value (₹)", height=500)
                        st.plotly_chart(fig_portfolio, use_container_width=True)
                        current_value = portfolio_values.iloc[-1]
                        total_return = current_value - total_investment
                        return_pct = (total_return / total_investment) * 100
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("💰 Current Value", f"₹{current_value:,.2f}")
                        with col2:
                            st.metric("📈 Total Return", f"₹{total_return:,.2f}", f"{return_pct:+.2f}%")
                        with col3:
                            returns = pd.Series(portfolio_values).pct_change().dropna()
                            volatility = returns.std() * np.sqrt(252) * 100
                            st.metric("📊 Volatility (Annual)", f"{volatility:.2f}%")
                        with col4:
                            sharpe = return_pct / volatility if volatility != 0 else 0
                            st.metric("⚡ Risk-Adj Return", f"{sharpe:.2f}")
                with portfolio_tabs[1]:
                    st.subheader("Portfolio Risk Analysis")
                    returns_data = {ticker: df["Close"].pct_change().dropna() for ticker, df in portfolio_data.items()}
                    if len(returns_data) > 1:
                        returns_df = pd.DataFrame(returns_data).dropna()
                        correlation_matrix = returns_df.corr()
                        fig_corr = px.imshow(correlation_matrix, text_auto=True, aspect="auto", title="Stock Correlation Matrix", color_continuous_scale="RdBu_r")
                        st.plotly_chart(fig_corr, use_container_width=True)
                        st.subheader("Risk Contribution")
                        risk_contributions = []
                        for ticker in tickers:
                            if ticker in returns_data and weights.get(ticker, 0) > 0:
                                stock_volatility = returns_data[ticker].std() * np.sqrt(252) * 100
                                risk_contribution = weights[ticker] * stock_volatility
                                risk_contributions.append({"Stock": ticker, "Weight (%)": f"{weights[ticker]*100:.1f}%", "Volatility (%)": f"{stock_volatility:.2f}%", "Risk Contribution": f"{risk_contribution:.2f}%"})
                        if risk_contributions:
                            risk_df = pd.DataFrame(risk_contributions)
                            st.dataframe(risk_df, use_container_width=True)
                            fig_risk = go.Figure(data=[go.Bar(x=[item["Stock"] for item in risk_contributions], y=[float(item["Risk Contribution"].replace('%', '')) for item in risk_contributions], marker_color="orange")])
                            fig_risk.update_layout(title="Risk Contribution by Stock", xaxis_title="Stock", yaxis_title="Risk Contribution (%)")
                            st.plotly_chart(fig_risk, use_container_width=True)
                with portfolio_tabs[2]:
                    st.subheader("Holdings Summary")
                    holdings_data = []
                    total_portfolio_value = 0
                    for ticker, symbol in zip(tickers, symbols):
                        if ticker in portfolio_data and weights.get(ticker, 0) > 0:
                            df = portfolio_data[ticker]
                            current_price = df["Close"].iloc[-1]
                            investment = total_investment * weights[ticker]
                            shares = investment / df["Close"].iloc[0]
                            current_value = shares * current_price
                            pnl = current_value - investment
                            pnl_pct = (pnl / investment) * 100
                            holdings_data.append({
                                "Stock": ticker, "Shares": f"{shares:.2f}", "Avg Cost (₹)": f"{df['Close'].iloc[0]:.2f}",
                                "Current Price (₹)": f"{current_price:.2f}", "Investment (₹)": f"{investment:,.2f}",
                                "Current Value (₹)": f"{current_value:,.2f}", "P&L (₹)": f"{pnl:,.2f}", "P&L (%)": f"{pnl_pct:+.2f}%"
                            })
                            total_portfolio_value += current_value
                    if holdings_data:
                        holdings_df = pd.DataFrame(holdings_data)
                        st.dataframe(holdings_df, use_container_width=True)
                        fig_pie = go.Figure(data=[go.Pie(labels=[item["Stock"] for item in holdings_data], values=[float(item["Current Value (₹)"].replace(',', '').replace('₹', '')) for item in holdings_data], hole=.3)])
                        fig_pie.update_layout(title="Current Portfolio Allocation")
                        st.plotly_chart(fig_pie, use_container_width=True)
                        st.subheader("📥 Export Portfolio Data")
                        create_download_buttons(holdings_df, "portfolio_holdings")
        except Exception as e:
            st.error(f"Portfolio analysis error: {str(e)}")
    else:
        st.info("👉 Select stocks and analysis mode from sidebar.")

st.markdown("---")
st.markdown("### Insights and Recommendations")