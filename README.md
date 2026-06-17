# Assessment 2 — IoT Sensor Data Trend Prediction

## Dataset
**Air Quality UCI** — Hourly averaged sensor readings from a chemical multisensor device deployed at road level in an Italian city (March 2004 – February 2005).

- Source: [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/Air+Quality)
- Domain: Urban air quality monitoring (industrial IoT)
- Target Variable: `T` — Ambient Temperature (°C)

## Setup

```bash
pip install pandas numpy scikit-learn matplotlib seaborn openpyxl jupyter
```

Download the dataset:
```
https://archive.ics.uci.edu/ml/machine-learning-databases/00360/AirQualityUCI.zip
```
Unzip and place `AirQualityUCI.xlsx` in this directory.

## Run

```bash
jupyter notebook iot_trend_prediction.ipynb
```

Run all cells top to bottom.

## Pipeline Overview

| Phase | Description |
|-------|-------------|
| **Ingestion** | Load Excel, parse datetime index, sort chronologically |
| **Cleaning** | Replace -200 sentinel values, IQR outlier clipping, hourly reindex, time-interpolation (≤6h gaps) |
| **Feature Engineering** | Lag features (1h/3h/6h/12h/24h), rolling mean/std (6h/12h/24h windows), calendar features |
| **Modeling** | Linear Regression (baseline), Random Forest, Gradient Boosting |
| **Evaluation** | RMSE, MAE, R² — chronological test split, train vs test gap analysis |
| **Visualization** | 5 publication-quality plots |

## Output Plots

| File | Description |
|------|-------------|
| `plot_01_cleaned_series.png` | Full cleaned temperature series + 30-day zoom |
| `plot_02_prediction_vs_truth.png` | Ground truth vs model predictions (14-day window) |
| `plot_03_model_comparison.png` | RMSE / MAE / R² bar chart across models |
| `plot_04_feature_importance.png` | Top 20 features (Random Forest) |
| `plot_05_residual_analysis.png` | Residuals over time + distribution |

## Key Design Decisions

### Why Random Forest + Gradient Boosting?
Temperature follows non-linear diurnal cycles and weather-driven patterns. Tree-based ensembles capture these interactions naturally without requiring deep learning infrastructure.

### Overfitting Prevention
- Strict chronological train/test split (no shuffling)
- `min_samples_leaf=4` in Random Forest prevents memorizing noise
- `subsample=0.8` in Gradient Boosting adds stochastic regularization
- Train vs Test RMSE gap explicitly reported

### IoT Anomaly Handling
- **Sentinel values (-200)**: Device-documented error flag — replaced with NaN
- **Outlier spikes**: IQR × 3.0 clipping removes hardware noise while preserving real extremes
- **Missing timestamps**: Hourly reindex + time-aware interpolation for gaps ≤ 6 hours
- **Long dropouts**: Dropped entirely to avoid fabricating sensor context
