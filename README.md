
# 📈 Stock Price Forecasting Project

This project demonstrates the implementation and comparison of multiple time series forecasting models to predict stock prices. The models used are **ARIMA**, **SARIMA**, **Prophet**, and **LSTM**. Each model has been trained on historical stock data, and their performances have been evaluated using common forecasting error metrics.

---

## 📌 Project Objectives

- Preprocess and clean stock market data.
- Visualize historical stock trends.
- Implement and train multiple forecasting models.
- Evaluate models using MAE, MSE, and RMSE..
- Compare models and draw meaningful insights.

---

## 🧠 Models Implemented

### 1. ARIMA (AutoRegressive Integrated Moving Average)
- Best suited for univariate time series data.
- Captures linear trend and noise.
- Requires stationarity, achieved via differencing.

### 2. SARIMA (Seasonal ARIMA)
- Extends ARIMA by modeling seasonality.
- Suitable for data with cyclic trends.
- Hyperparameters: (p, d, q)(P, D, Q, s)

### 3. Prophet (by Meta)
- Handles seasonality, trends, and holidays well.
- Easy to tune and visualize components.
- Robust to missing data and outliers.

### 4. LSTM (Long Short-Term Memory)
- Deep learning model for sequential data.
- Captures long-term dependencies.
- Requires normalization and reshaping of input.

---

## 📊 Evaluation Metrics

| Model   | MAE    | MSE    | RMSE   |
|---------|--------|--------|--------|
| ARIMA   | 3.2721 | 27.5875 | 5.2521 |
| SARIMA  | 3.3189 | 27.6871 | 5.2619 |
| Prophet | 5.9030 | 56.2981 | 7.5032 |
| LSTM    | 7.8815 | 87.3722 | 9.3473 |

**Conclusion**: ARIMA and SARIMA outperformed the others with the lowest errors, while Prophet offered useful visual insights. LSTM underperformed, possibly due to limited data size or tuning.

---

## 📁 Project Structure

```
📦 stock-forecasting-project
├── stock_forecasting.ipynb       # Main Jupyter notebook
├── cleaned_stock_data.csv        # Cleaned input data
├── model_outputs/                # Model outputs and visualizations
├── README.md                     # Project documentation
└── requirements.txt              # Python dependencies
```

---

## 🚀 How to Run

1. Clone this repository:
```bash
git clone https://github.com/your-username/stock-forecasting-project.git
cd stock-forecasting-project
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Open the notebook:
```bash
jupyter notebook stock_forecasting.ipynb
```

---

## 📬 Author

Created by **Vijeth Amin**  
Feel free to reach out for collaborations or questions.

📅 Last updated: July 24, 2025
