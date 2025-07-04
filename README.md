# 🧠 Retail Time Series Forecasting: Deep Learning vs. Machine Learning

This repository contains the complete code and experiments from my master’s thesis, benchmarking deep learning models against traditional machine learning approaches for large-scale retail time series forecasting.  
All work is organized into three parts—each focused on a different dataset and pipeline, but following a consistent workflow: **data preprocessing, feature engineering, model development, and robust time series cross-validation**.

---

## 1️⃣ Visuelle Retail Forecasting (Multimodal, Custom Deep Learning)

**Dataset:** Visuelle 2.0 (proprietary retail sales + product images)  
**Objective:** Compare a custom-built TensorFlow/Keras N-BEATS model and LightGBM on multimodal retail data, integrating image features.

- Data preprocessing merges sales, calendar, and image-based CNN features
- **Image Feature Extraction:** VGG16 CNN → tabular “image feature” per SKU
- Feature engineering: calendar, lags, rolling stats
- Model training:
  - **Custom N-BEATS** (TensorFlow/Keras, built from scratch)
  - **PyTorch N-BEATS** (reference)
  - **LightGBM** (baseline)
- **Cross-Validation:**
  - Group Time Series, Purged Group Time Series, Combinatorial Purged CV (all implemented)
- **Evaluation:** MAE, WAPE, MAPE

---

## 2️⃣ M5 Forecasting (Walmart Retail Sales)

**Dataset:** [M5 Forecasting – Accuracy (Kaggle)](https://www.kaggle.com/competitions/m5-forecasting-accuracy)  
**Objective:** Predict daily sales for 3,000+ Walmart products across 10 stores using tabular, calendar, and price features.

- Data preprocessing with calendar, price, and metadata features
- Lag and rolling statistics engineering
- Model training:
  - N-BEATS (via Darts)
  - LightGBM (gradient boosting baseline)
  - Naive methods (seasonal/drift)
- **Cross-Validation:**
  - GroupTimeSeriesSplit and Purged Split (time-aware, group-respecting)
- **Hyperparameter tuning:** Optuna for N-BEATS

---

## 3️⃣ M4 Hourly Forecasting (Benchmark Time Series Study)

**Dataset:** [M4 Hourly Dataset](https://github.com/Mcompetitions/M4-methods)  
**Objective:** Benchmark N-BEATS and LightGBM on standard hourly time series using advanced CV.

- Data cleaning, scaling, and lag feature creation
- Modular, reusable functions for modeling and evaluation
- Model training:
  - N-BEATS (deep neural network for time series)
  - LightGBM (lagged feature regression)
- **Cross-Validation:**
  - GroupTimeSeries, PurgedGroupTimeSeries, and CombinatorialPurgedGroupKFold (all implemented and explained)
- **Evaluation:** RMSE, MAPE, SMAPE across all folds

---

## 🛠️ Common Workflow

**Key Steps:**
- **Data Preprocessing:**  
  Start with the preprocessing notebook in each folder to generate a clean, feature-rich dataset.
- **Model Training & Evaluation:**  
  Choose and run the modeling notebook(s) for the approach you want to benchmark.
- **Cross-Validation:**  
  All parts use advanced, time-aware CV to avoid data leakage and ensure fair assessment.

---

## 📦 Dependencies

All parts require Python 3.8+ and the following core packages:

- `pandas`, `numpy`, `scikit-learn`
- `tensorflow`, `keras`, `torch` (PyTorch)
- `lightgbm`, `darts`, `optuna` (for some modules)
- `matplotlib`, `plotly`, `seaborn`

Install with:
```bash
pip install pandas numpy scikit-learn tensorflow keras torch lightgbm darts optuna matplotlib plotly seaborn
