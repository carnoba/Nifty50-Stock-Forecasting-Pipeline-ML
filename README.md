# 📈 NIFTY 50 Time-Series Forecasting & Analytics

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Status](https://img.shields.io/badge/Status-Project_Complete-brightgreen.svg)]()
[![ML](https://img.shields.io/badge/Machine_Learning-Prophet_%7C_ARIMA-orange.svg)]()

A professional-grade financial analytics pipeline designed to analyze and forecast the **NIFTY 50 Index**. This project leverages advanced statistical models (ARIMA) and additive models (Facebook Prophet) to predict market trends with high precision, incorporating technical indicators for enhanced accuracy.

---

## 🌟 Key Features

- **📊 Advanced Data Processing**: Automatic handling of market holidays, stationarity checks (ADF Test), and seasonal decomposition.
- **🤖 Dual-Model Forecasting**:
  - **Auto-ARIMA**: Intelligent parameter selection for optimal time-series fitting.
  - **Facebook Prophet**: Multi-regressor model utilizing technical indicators (RSI, MACD, SMA).
- **📈 Professional Dashboarding**: Automated generation of comprehensive analytics dashboards.
- **⚖️ Comparative Analysis**: Side-by-side performance evaluation using RMSE, MAE, and MAPE.
- **🔬 Statistical Rigor**: Residual analysis, normalcy checks, and white-noise verification.

---

## 📸 Visualization Preview

### 1. Forecasting Dashboard

The consolidated dashboard provides a deep dive into model predictions versus actual market movements, along with residual error distributions.

![Forecasting Dashboard](nifty_forecast_dashboard.png)

### 2. Seasonal Decomposition

Understanding the underlying Trend, Seasonality, and Residual components of the NIFTY 50 index.

![Decomposition Plot](decomposition_plot.png)

---

## 🚀 Tech Stack

- **Language:** Python 3.x
- **Time-Series Models:** `statsmodels` (ARIMA), `prophet` (Facebook), `pmdarima` (Auto-ARIMA)
- **Data Engineering:** `pandas`, `numpy`
- **Visualization:** `matplotlib`, `seaborn`
- **Evaluation:** `scikit-learn`

---

## 🛠️ Installation & Setup

1. **Clone the Repository**

   ```bash
   git clone https://github.com/your-username/nifty-forecasting.git
   cd nifty-forecasting
   ```

2. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Pipeline**
   ```bash
   python main.py
   ```

---

## 📂 Project Structure

```text
├── data_processor.py      # Data cleaning, ADF test, & decomposition logic
├── forecasting_models.py  # ARIMA and Prophet model implementations
├── main.py                # Entry point & dashboard generation
├── requirements.txt       # Project dependencies
└── Nifty_50_with_indicators_.csv # Dataset (Input)
```

---

## 📈 Methodology

1. **Preprocessing**: The `DataProcessor` cleans the raw CSV, handles gaps via forward-filling (ffill), and slices the last 5 years of data.
2. **Stationarity**: Augmented Dickey-Fuller (ADF) test is performed to determine the integration order (d).
3. **ARIMA Model**: `auto_arima` optimizes (p, d, q) parameters automatically based on AIC/BIC.
4. **Prophet Model**: A powerful forecasting engine that treats the time-series as a curve-fitting problem, enhanced by **SMA20**, **RSI14**, and **MACD** as external regressors.
5. **Validation**: Time-Series Cross-Validation and error metrics (RMSE/MAPE) ensure model reliability.

---

## 📝 SEO & Search Optimization

`NIFTY 50 Forecast`, `Stock Market Prediction Python`, `Time Series Analysis`, `Prophet vs ARIMA`, `Financial Data Science`, `Technical Analysis Machine Learning`, `NIFTY Index Prediction`, `Quantitative Finance Python`.

---

## 🤝 Contributing

Contributions are welcome! If you have ideas for improving the models or adding new indicators, feel free to fork and submit a PR.

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

_Created with ❤️ for Financial Data Science enthusiasts._
