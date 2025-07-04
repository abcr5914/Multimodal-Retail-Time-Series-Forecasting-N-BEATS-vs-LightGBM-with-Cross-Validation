# Visuelle Time Series Forecasting: Deep Learning vs. Traditional ML

This project benchmarks deep neural network approaches against traditional machine learning for retail sales forecasting on the Visuelle 2.0 dataset. It features a fully custom TensorFlow/Keras implementation of the N-BEATS architecture, as well as a LightGBM baseline, with all models rigorously evaluated using advanced time series cross-validation strategies.

---

## Project Overview

- **Goal:** Compare the performance of a modern deep learning model (N-BEATS) with a proven machine learning approach (LightGBM) for retail sales forecasting.
- **Dataset:** Visuelle 2.0 – a real-world retail dataset with sales history, product images, and associated metadata.
- **Pipeline:** End-to-end workflow from data preprocessing (including image feature extraction) to model building, evaluation, and visualization.

---

## Data Processing Pipeline

All experiments begin with thorough data preprocessing:
- **Raw Input:** Sales data per SKU (date, product ID, sales volume) and associated product images.
- **Image Feature Extraction:** Each product image is processed using a pretrained VGG16 CNN (Keras). The output activations are flattened and the mean is computed, resulting in a single numeric “image feature” per product.
- **Feature Engineering:** Additional tabular features are added:
    - Calendar features: day of week, week of year, month, etc.
    - Lagged sales and rolling statistics to capture recent trends and seasonality.
- **Merging:** The image feature is merged into the main dataset, so every row contains both tabular and image-derived information.
- **Missing Data:** Addressed using standard imputation.
- **Final Output:** A single processed dataset containing all engineered features, ready for model input.

---

## Modeling Approaches

### 1. Custom N-BEATS (TensorFlow/Keras)

- **What is N-BEATS?**  
  Neural Basis Expansion Analysis for Time Series is a state-of-the-art deep learning architecture designed specifically for univariate and multivariate forecasting.
- **Your Implementation:**  
    - Built entirely from scratch in TensorFlow/Keras (no wrappers or prebuilt libraries).
    - Modular structure: model is composed of stacks, each with multiple blocks.
    - Each block has 4 fully connected (Dense) layers (ReLU activations), followed by backcast (reconstruction) and forecast (prediction) outputs.
    - Residual connections and aggregation across stacks, as described in the original N-BEATS paper.
    - Highly configurable: block depth, stack count, hidden units, and input feature size can all be tuned.
    - Accepts both tabular features and the scalar image feature.
    - Training uses MSE loss and is tracked with WAPE, MAE, and MAPE metrics.

### 2. LightGBM

- Gradient boosting model (decision tree based), serving as a strong traditional ML baseline.
- Trained and evaluated on the **same processed dataset** as N-BEATS, ensuring a fair comparison.
- Uses all engineered features, including the image feature, as input.

### 3. PyTorch N-BEATS (Reference)

- Includes a PyTorch-based implementation of N-BEATS for additional benchmarking.
- Trained and evaluated using the same pipeline and cross-validation splits.

---

## Cross-Validation Strategies

Robust model evaluation is a major focus of this project. Standard K-Fold CV is **not** used due to leakage risk in time series.

**Implemented CV methods:**
- **Group Time Series Cross-Validation:** Ensures that groups (e.g., SKUs or time periods) do not leak between train and test splits.
- **Purged Group Time Series Cross-Validation:** Further purges data near split boundaries to prevent overlap and leakage.
- **Combinatorial Purged Cross-Validation:**  
  A sophisticated approach generating many train/test split combinations while purging overlapping groups, leading to a highly realistic assessment of model performance.

All CV approaches are implemented from scratch and explained in the notebooks.

---

## Repository Structure

- `DataPreprocessing.ipynb`  
    Complete data cleaning, feature engineering, image-to-feature extraction, and merging. Run this notebook first.
- `Custom_Nbeats_Forecast.ipynb`  
    End-to-end workflow for custom N-BEATS model (TensorFlow/Keras): model definition, training, advanced CV, and evaluation.
- `Light_GBM_Forecast.ipynb`  
    Baseline LightGBM modeling and evaluation using the same processed features and CV strategies.
- `Pytorch_NBeats_Forecast.ipynb`  
    PyTorch-based reference implementation of N-BEATS for further benchmarking.

---

## How to Run

1. **Preprocessing:**  
    Run `DataPreprocessing.ipynb` to generate the processed dataset.
2. **Model Training & Evaluation:**  
    Choose either `Custom_Nbeats_Forecast.ipynb`, `Pytorch_NBeats_Forecast.ipynb`, or `Light_GBM_Forecast.ipynb` to train and evaluate models on the preprocessed data.

---

## Dependencies

- `pandas`
- `numpy`
- `scikit-learn`
- `tensorflow` (for N-BEATS)
- `keras`
- `torch` (for PyTorch N-BEATS)
- `lightgbm`
- `matplotlib`

Install via:
```bash
pip install pandas numpy scikit-learn tensorflow keras torch lightgbm matplotlib
