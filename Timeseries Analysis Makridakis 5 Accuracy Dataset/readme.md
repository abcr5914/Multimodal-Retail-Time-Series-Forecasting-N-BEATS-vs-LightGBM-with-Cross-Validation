
# 🧠 M5 Forecasting with N-BEATS & LightGBM

This project addresses the **M5 Forecasting Accuracy** challenge using deep learning and gradient boosting methods on multivariate retail time series data. The approach combines careful feature engineering, model tuning, and time-aware validation, optimized for use in Google Colab environments.

---

## 📦 Dataset

- **Source**: [M5 Forecasting – Accuracy (Kaggle)](https://www.kaggle.com/competitions/m5-forecasting-accuracy)
- **Data**: Daily sales of 3,000+ Walmart products across 10 stores and 3 states
- **Features used**:
  - Calendar events (e.g., holidays, SNAP days)
  - Price dynamics (`sell_price`)
  - Product/store metadata (category, department, store)
- **Multimodal preprocessing**:
  - One-hot encoding for categorical features
  - Time series lags and rolling window statistics
  - Combined into unified DataFrames for model input

---

## 🔍 Models Compared

- **N-BEATS** using the [`darts`](https://github.com/unit8co/darts) time series library
- **LightGBM** trained on reformatted tabular sequences
- **NaiveSeasonal** and **NaiveDrift** for baseline comparison

---

## 🛠️ Key Implementation Aspects

- **Feature Engineering**:
  - `extend_index_with_past()` to prepare future forecasting windows
  - Lags, rolling means, and price trends per series
- **Memory Efficiency**:
  - `reduce_memory()` for downcasting large DataFrames (Colab optimized)
- **Reusable Functions**:
  - `fit_predict_and_evaluate()` to unify model training and evaluation
  - Metrics computed: SMAPE, MASE
- **Hyperparameter Tuning**:
  - Conducted via **Optuna** for N-BEATS (block layers, dropout, learning rate)

---

## ⏱️ Time Series Cross-Validation

Special care is taken to prevent temporal leakage:

- **GroupTimeSeriesSplit**: Respects temporal order within product-store groups
- **Purged Split**: Prevents contamination by removing overlapping time windows

> While Combinatorial Purged CV is referenced, only GroupTimeSeries and Purged methods are implemented in this notebook.

---

## 📚 Requirements

Install core dependencies:

```bash
pip install darts optuna plotly lightgbm
```

Python 3.8+ recommended.
