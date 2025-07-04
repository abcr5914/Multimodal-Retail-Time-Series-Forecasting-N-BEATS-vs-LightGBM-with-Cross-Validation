# 📘 M4 Hourly Forecasting: N-BEATS vs LightGBM (Comparative Study)

This notebook presents a comparative study between two time series forecasting models — **N-BEATS** and **LightGBM** — on the **M4 Hourly Dataset**, a widely used benchmark. The study explores model performance across multiple folds using realistic time-based validation techniques. It also examines runtime optimization, especially for constrained environments like Google Colab.

---

## 🔧 What Was Done

- **Data preparation**: Cleaned, reshaped, and scaled M4 Hourly series using `MaxAbsScaler` and Darts’ `TimeSeries` API.
- **Feature engineering**: Created lag-based features, datetime encodings, and multivariate combinations (for LightGBM).
- **Model implementation**: Modular functions were defined to create, train, and evaluate both models.
- **Time-aware validation**: Cross-validation was adapted specifically for time series using group-aware, purged, and embargoed strategies.
- **Evaluation**: RMSE, MAPE, and SMAPE metrics were used to benchmark both models across folds, with plots for comparison.

---

## 🧠 Models Overview

- **N-BEATS**: Deep learning architecture tailored for time series. It learns interpretable trend and seasonality blocks from raw lagged values.
- **LightGBM**: A fast gradient boosting method applied here with lag-based features, treating forecasting as a supervised regression problem on past windows.

---

## 🔁 Time Series Cross-Validation (CV)

Traditional k-fold CV is not suitable for time series because it shuffles the data and breaks temporal dependencies. Instead, this notebook implements **three advanced CV strategies**, each respecting time order and designed to avoid leakage from train to test sets.

### ✅ CV Methods Implemented

1. **GroupTimeSeriesSplit**
   - Like expanding window CV, but grouped by logical blocks (e.g., product or SKU).
   - Ensures no group appears in both train and test sets within any fold.

2. **PurgedGroupTimeSeriesSplit**
   - Adds a **gap** between train and test splits to avoid information leakage.
   - Especially useful when using lag features or windowed inputs.

3. **CombinatorialPurgedGroupKFold**
   - Generates all valid combinations of train/test group splits while **purging overlapping data** and applying an **embargo period**.
   - Best suited for high-leakage-risk time series forecasting problems.

### 🧠 Why This Is Different

These methods:
- Handle **grouped time series** (e.g., multiple products/customers)
- Prevent leakage due to lag features or windowed data
- Simulate real-world evaluation using **group-purging** and **time-aware holdouts**

This makes the evaluation more robust and applicable for retail, finance, and other sequential domains.

---

## 🧩 Key Functions Created

- `create_nbeats_model()` — Sets up a tuned N-BEATS architecture for forecasting
- `create_lightgbm_model()` — Prepares LightGBM with lag and time-based features
- `fit_predict_and_evaluate()` — Manages model training, forecasting, and error evaluation
- `objective()` — Optuna-compatible function for LightGBM hyperparameter tuning
- `multi_ts_support()` & `multivariate_support()` — Format converters for batch/multivariate forecasting

---

## 📦 Required Packages

Make sure the following packages are installed:
```bash
pip install darts optuna lightgbm pandas numpy scikit-learn plotly matplotlib seaborn
```

Key Libraries:
- `darts` – time series data structure, modeling, and evaluation
- `optuna` – automated hyperparameter tuning
- `scikit-learn` – preprocessing and cross-validation
- `plotly`, `seaborn`, `matplotlib` – data visualization
