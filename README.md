# GreenCharge: AI-Based Load Forecasting for Hostel Energy Systems

An end-to-end machine learning system for high-accuracy electricity load forecasting, anomaly detection, and energy load balancing across college hostel buildings.

Aligned with **United Nations Sustainable Development Goal 12 (Responsible Consumption & Production)**.

---

## ⚡ Key Results

Evaluated on out-of-sample validation data ($N = 12,997$ observations across 12 hostel buildings, 168-hour temporal lookback window):

| Model | $R^2$ Score | RMSE ($\text{kW}$) | MAE ($\text{kW}$) | MAPE ($\%$) | Architecture |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **XGBoost + BiLSTM** | **0.9880** | **11.96** | **7.95** | **4.62%** | Hybrid: Gradient Boosted Trees + 168h Residual BiLSTM |
| **XGBoost** | **0.9878** | **12.07** | **7.97** | **4.66%** | Gradient Boosted Decision Trees |
| **Random Forest** | **0.9829** | **14.33** | **9.90** | **5.72%** | Bagging Ensemble of 500 Trees |
| **ANN** | **0.9823** | **14.58** | **10.68** | **6.13%** | Deep Tabular Feedforward Neural Network |
| **Meta-Ensemble** | **0.9889** | **11.51** | **7.81** | **4.28%** | 5-Fold Cross-Validated Ridge Stacking |

---

## 🛠️ Architecture

```
Raw Building & Weather Telemetry
               │
               ▼
   [ Feature Engineering Engine ]
   ├── Cyclical temporal encodings (Hour, Day, Month)
   ├── Lag features & rolling consumption statistics
   └── Weather context (Temperature, Humidity, Wind)
               │
       ┌───────┴───────┐
       ▼               ▼
 [ XGBoost Base ]  [ 168h Sequence Window ]
       │               │
       ▼               ▼
 Base Predictions   Residuals
       │               │
       └───────┬───────┘
               ▼
   [ BiLSTM Residual Network ]
               │
               ▼
   [ Final Hybrid Predictions ]
   (R² = 0.9880 | RMSE = 11.96 kW)
```

---

## 📊 Modules

1. **Load Forecasting**: 168-hour window forecasting models combining tabular gradient boosting with bidirectional recurrent sequence correction.
2. **Anomaly Detection**: Statistical and autoencoder-based deviation detection with severity tiering (Low, Medium, High).
3. **Load Balancing & Peak Shaving**: Simulations for automated peak shaving, 100 kW rooftop solar integration, and staggered appliance scheduling.
4. **Editorial Dashboard**: Streamlit-based monitoring dashboard featuring publication-quality black-themed matplotlib and Plotly charts.

---

## 🚀 Getting Started

### 1. Installation
```bash
git clone https://github.com/MAK-T800/GreenCharge.git
cd GreenCharge
pip install -r requirements.txt
```

### 2. Run Data Processing & Training
```bash
python data_pipeline.py
python train.py
```

### 3. Launch Dashboard
```bash
streamlit run app.py
```

---

## 📁 Repository Structure

```
├── app.py                      # Streamlit dashboard
├── dashboard_utils.py          # Visualization & layout components
├── generate_plots.py           # Publication-quality matplotlib generator
├── data_pipeline.py            # Feature engineering & preprocessing
├── load_balancing.py           # Peak shaving & solar optimization
├── train.py                    # Multi-model training pipeline
├── config.py                   # Centralized configuration
├── models/
│   ├── tree_models.py          # XGBoost, Random Forest
│   ├── deep_models.py          # ANN, CNN (168h)
│   ├── hybrid_models.py        # XGBoost + BiLSTM, CNN + BiLSTM, Dual ANN
│   ├── ensemble.py             # Meta-learner stacking & inverse-variance blending
│   ├── anomaly_detection.py    # Isolation Forest & Autoencoder
│   └── evaluation.py           # Standardized metrics & reports
└── plots/                      # Generated evaluation plots
```
