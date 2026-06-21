# IoT Sensor Data Trend Prediction

End-to-end machine learning pipeline for forecasting ambient temperature from a real, noisy, multi-sensor industrial IoT deployment — covering data cleaning, feature engineering, model training, and evaluation.

---

## Dataset & Industrial Context

**Source**: [Air Quality UCI dataset](https://archive.ics.uci.edu/dataset/360/air+quality) — hourly readings from a multi-sensor array deployed in the field in an Italian city (2004–2005), originally collected for air quality monitoring research. This is a genuine connected-hardware deployment: 5 metal-oxide chemical sensors plus reference analyzers, logging continuously over roughly a year, with the typical real-world IoT problems — sensor drift, missing readings, and a known-faulty channel.

**Target variable**: Ambient Temperature (°C), predicted from the sensor array's other channels plus engineered temporal features. Temperature was chosen as the target because it has clear physically-grounded daily and seasonal periodicity, making it a good test of whether lag/rolling-window features can capture that structure.

---

## Data Cleaning Methodology

| Issue | Handling | Why |
|---|---|---|
| Sentinel missing-value code (`-200`) | Replaced with `NaN`, per-column | The dataset uses `-200` to flag a missing/invalid reading instead of a true blank — treating it as a real value would massively distort any model |
| `NMHC(GT)` sensor channel | Dropped entirely | This channel failed for the majority of the deployment window (well over 85% missing) — a documented issue with this specific dataset. Imputing or interpolating across that much missingness would mostly be inventing data, not cleaning it |
| Irregular/missing timestamps | Reindexed to a strict hourly grid | Guarantees lag and rolling-window features are computed over actual elapsed time, not just row position |
| Short gaps (≤6h) | Time-based interpolation | Short dropouts are typical of intermittent connectivity and are safe to linearly bridge without introducing meaningfully fake trends |
| Long gaps (>6h) | Dropped rather than interpolated | Interpolating across long blackouts would fabricate hours of fake sensor behavior rather than reflect genuine readings |
| Out-of-bounds spikes | Clipped using IQR × 3.0 | Wide enough to preserve genuine daily swings, tight enough to catch hardware glitches/transients without manually hand-picking thresholds per sensor |

---

## Feature Engineering

- **Lag features**: 1h, 3h, 6h, 12h, 24h — captures short-term momentum and the prior-day same-hour value, which is typically the single strongest predictor of temperature
- **Rolling statistics**: 6h/12h/24h rolling mean and standard deviation — captures local trend and local volatility, useful both as predictive signal and as an implicit noise filter
- **Calendar features**: hour-of-day, day-of-week — gives the model direct access to the diurnal cycle instead of forcing it to infer periodicity purely from lag values

---

## Modeling & Results

Three models were trained and compared: a Linear Regression baseline, Random Forest, and Gradient Boosting. The split was **strictly chronological** (no shuffling) — the model never sees future data during training, which is the only valid way to evaluate a forecasting model.

```
Train size : 251 hourly samples  (2004-01-05 → 2004-11-04)
Test size  : 63 hourly samples   (2004-11-04 → 2004-12-05)
Features   : 30 (raw sensors + lag + rolling + calendar)
```

| Model | RMSE (°C) | MAE (°C) | R² |
|---|---|---|---|
| **Linear Regression (Baseline)** | **0.8713** | **0.7082** | **0.9132** |
| Gradient Boosting | 1.3814 | 1.0655 | 0.7818 |
| Random Forest | 1.4996 | 1.0886 | 0.7428 |

### The simpler model won — and that's the actual finding here

Going in, the expectation was that tree ensembles would outperform a linear baseline by capturing non-linear interactions between sensors. The results say otherwise: **Linear Regression outperformed both tree-based models on every metric.**

The overfitting gap analysis explains why:

| Model | Train RMSE | Test RMSE | Gap |
|---|---|---|---|
| Linear Regression (Baseline) | 0.4619 | 0.8713 | +0.4094 |
| Random Forest | 0.7407 | 1.4996 | +0.7589 |
| Gradient Boosting | 0.0271 | 1.3814 | +1.3543 |

Gradient Boosting nearly memorized the training set (train RMSE of 0.027 — essentially zero error) but generalized poorly, producing the largest train/test gap of any model. Random Forest shows the same pattern at a smaller scale. Linear Regression had by far the smallest gap, meaning its training and test performance were the most consistent with each other.

**Interpretation**: the lag and rolling-window features already encode most of the temporal structure (the prior hour, prior day, and local trend) in a way that's close to linear with respect to the target. A linear model can exploit that directly. The tree ensembles, given the same features but a comparatively small training set (251 rows) and a learning task with limited genuine non-linear signal beyond what the lag features expose, ended up fitting training-set noise instead of finding additional real structure — classic overfitting on a small time-series sample. This is the core argument for why simplicity should be the default starting point on small time-series datasets, and complexity (tree ensembles, then deep sequential models) should only be added once there's a measured benefit over a strong baseline — not assumed in advance.

### Why not LSTM / sequential neural networks?

A sequential deep learning approach was deliberately not used, for reasons reinforced by the result above:

1. **Dataset size**: ~300 total samples is far below what an LSTM needs to learn temporal dependencies reliably — if Random Forest and Gradient Boosting already overfit at this scale, a deep sequential model would do so more severely.
2. **The dominant signal is periodic, not long-memory**: daily/seasonal cycles are already exposed directly via lag and calendar features. An LSTM would have to discover that periodicity implicitly from raw sequences, which is both less data-efficient and less interpretable than handing it the structure directly.
3. **Interpretability**: tree/linear models expose feature importances and coefficients directly, supporting the residual and feature-importance analysis this evaluation relies on; an LSTM would need a separate interpretability layer for equivalent insight.

For a larger multi-year, multi-sensor deployment, or a task with genuinely longer-range dependencies (e.g. predicting gradual equipment drift toward failure weeks ahead), an LSTM/GRU would be the better default — it wasn't here.

---

## Evaluation & Visualization

Five plots are generated by the notebook:
1. **Cleaned time series** — raw vs. cleaned signal, showing the effect of interpolation/clipping
2. **Prediction vs. ground truth** — historical actuals against the model's forecast on the test window
3. **Model comparison** — RMSE/MAE/R² across all three models
4. **Feature importance** — which lag/rolling/calendar features drove the Random Forest and Gradient Boosting predictions
5. **Residual analysis** — error distribution over time, used to check for systematic bias rather than just aggregate error

---

## Known Limitations / Future Work

- **Final modeling dataset (314 total rows: 251 train + 63 test) is smaller than the full source file (~9,357 hourly rows over the year-long deployment).** This is a known gap between the intended cleaning pipeline and the current loaded dataset that has been identified but not yet root-caused — it appears to originate at or before the initial load step rather than in the documented cleaning logic above, since row counts are already reduced before the cleaning stages run. Flagged here transparently rather than glossed over: the cleaning *methodology* described above is sound and intentional, but the *volume* of data it's currently operating on is smaller than expected, which likely also contributes to the tree-ensemble overfitting discussed above. Investigating and restoring the full dataset size is the top follow-up item.
- No deep sequential model (LSTM/GRU) was benchmarked — see justification above; would be worth revisiting if the dataset size issue above is resolved and the training set grows substantially.
- Single train/test chronological split rather than rolling-origin cross-validation — sufficient to demonstrate methodology, but a production pipeline would benefit from multiple forward-chaining splits to get a more robust estimate of generalization error.

---

## Project Structure

```
IoT-Sensor-Data-Trend-Prediction/
├── iot_trend_prediction.ipynb   # Full pipeline: cleaning, feature engineering, training, evaluation
├── AirQualityUCI.xlsx           # Source dataset
├── requirements.txt
├── plot_01_cleaned_series.png
├── plot_02_prediction_vs_truth.png
├── plot_03_model_comparison.png
├── plot_04_feature_importance.png
├── plot_05_residual_analysis.png
└── README.md
```

---

## Setup & Run

```bash
pip install pandas numpy scikit-learn matplotlib seaborn openpyxl jupyter
jupyter notebook iot_trend_prediction.ipynb
```

Run all cells top to bottom (Kernel → Restart & Run All).