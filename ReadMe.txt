
# 📊 Stock Market Forecasting and Analysis Dashboard

## 📌 Introduction
This project is an **interactive dashboard** built with Streamlit for **real-time stock market analysis and forecasting**.  
It integrates **yFinance API** for data, **Plotly** for visualization, and **machine learning models** (ARIMA, SARIMA, Prophet, LSTM) for forecasting.

---

## 🎯 Objectives
- Collect live financial data using yFinance.  
- Provide interactive visualization of stock trends and technical indicators.  
- Forecast stock prices using ARIMA, SARIMA, Prophet, and LSTM.  
- Compare models using performance metrics (MAE, RMSE, MAPE, R², Directional Accuracy).  
- Support multi-stock comparison and portfolio analysis.  
- Deploy on cloud (EC2/Streamlit Cloud).  

---

## 🏗️ System Architecture
1. **Data Source:** Yahoo Finance via yFinance  
2. **Preprocessing:** Cleaning, handling missing values, splitting train-test data  
3. **Visualization:** Plotly charts, Moving Averages, RSI, Bollinger Bands  
4. **Forecasting Models:** ARIMA, SARIMA, Prophet, LSTM  
5. **Evaluation:** MAE, RMSE, MAPE, R², Directional Accuracy  
6. **Interface:** Streamlit Dashboard  

---

## ⚙️ Methodology
1. **Data Collection** → Yahoo Finance API  
2. **EDA** → Price charts, technical indicators  
3. **Forecasting** → Apply ARIMA, SARIMA, Prophet, LSTM  
4. **Evaluation** → Compare metrics  
5. **Visualization** → Plotly + Streamlit UI  

---

## 📊 Features
- 📈 Real-time stock data retrieval  
- 🔍 Technical indicators (MA, RSI, Bollinger Bands)  
- 🤖 Forecasting with ARIMA, SARIMA, Prophet, LSTM  
- 📊 Multi-stock comparison  
- 💼 Portfolio analysis (returns, correlation heatmaps)  
- 🌐 Streamlit-based deployment  

---

## ✅ Results
- Prophet & LSTM perform better for long-term predictions.  
- ARIMA/SARIMA work well for short-term forecasts.  
- Directional accuracy achieved between 55–70%.  

---

## 🚀 Future Enhancements
- Add news sentiment analysis for market movement prediction.  
- Integrate reinforcement learning for trading strategies.  
- Expand to cryptocurrency and commodities forecasting.  
- Deploy on AWS/GCP with background auto-refresh jobs.  

---

## 🛠️ Tech Stack
- **Python**
- **Libraries:** yFinance, Pandas, NumPy, Scikit-learn, Statsmodels, Prophet, TensorFlow/Keras, Plotly  
- **Framework:** Streamlit  
- **Deployment:** AWS EC2 / Streamlit Cloud  

---

## 📚 References
- Yahoo Finance API  
- Facebook Prophet Documentation  
- Time Series Forecasting (ARIMA, LSTM) Literature  
- Streamlit Documentation  
