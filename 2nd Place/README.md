# Poverty Prediction Challenge - 2nd Place Solution

**Author:** Driss Taki  
**Date:** 2026-02-06  
**Competition:** DrivenData Poverty Prediction Challenge

## Summary

This solution achieves 2nd place using a Leave-One-Survey-Out cross-validation strategy with LightGBM gradient boosting models. The approach focuses on preventing data leakage through careful feature engineering and employs weighted quantile calibration to improve poverty rate predictions.

**Key Innovations:**
- Survey-specific mean ratio features (anti-leakage design)
- Weighted quantile calibration for distribution matching
- Ensemble of survey-specific models
- Feature selection based on importance (top 75%)

## Setup

### Hardware & Software Requirements
- **OS:** Linux (tested on Google Colab Ubuntu)
- **RAM:** Minimum 8GB
- **Runtime:** ~15 minutes on standard CPU

### Dependencies

```bash
pip install -r requirements.txt
```

**Python Version:** 3.12.12

**Required packages:**
- numpy==2.0.2
- pandas==2.2.2
- scikit-learn==1.6.1
- lightgbm==4.6.0

## Data Setup

Place competition data files in `sample_data/`:
```
sample_data/
├── train_hh_features.csv
├── train_hh_gt.csv
└── test_hh_features.csv
```

## Running the Code

### Generate Predictions

```bash
python drivendata_poverty_prediction.py
```

This will:
1. Load and preprocess data
2. Create non-leakage features
3. Run Leave-One-Survey-Out CV
4. Select top 75% features by importance
5. Train final ensemble models
6. Generate submission files in `submission/`

**Output files:**
- `submission/predicted_household_consumption.csv`
- `submission/predicted_poverty_distribution.csv`

## Approach

### 1. Feature Engineering (No Leakage)

**Household-level features:**
- Per capita income/expenditure ratios
- Age statistics (mean, max, variance, min/max ratio)
- Education level aggregates
- Food consumption diversity and totals
- Asset counts and ownership indicators

**Survey-context features:**
- Ratio of household value to survey mean (computed only on training data)
- Prevents leakage by using fold-specific survey statistics

### 2. Modeling Strategy

**Cross-Validation:**
- Leave-One-Survey-Out (LOSO) to simulate real deployment
- Each fold uses other surveys' statistics for feature engineering

**Model Architecture:**
- LightGBM Regressor with L2 loss on log-transformed targets
- Hyperparameters:
  - `n_estimators=3000` with early stopping (patience=200)
  - `learning_rate=0.01`
  - `num_leaves=128`
  - `subsample=0.7`, `colsample_bytree=0.7`

**Ensemble:**
- Train one model per survey (excluding that survey from training)
- Average predictions from all models for final submission

### 3. Post-Processing

**Weighted Quantile Calibration:**
- Maps predicted distribution to match training distribution
- Improves poverty rate estimation accuracy
- Critical for the ws-wMAPE metric (90% poverty rates, 10% consumption)

**Feature Selection:**
- Retain top 75% features by mean importance across CV folds
- Removes noise and improves generalization

**Competition Metric:**
```
ws-wMAPE = 0.9 × wMAPE(poverty_rates) + 0.1 × wMAPE(consumption)
```

## Repository Structure

```
.
├── README.md                                   # This file
├── requirements.txt                            # Python dependencies
├── drivendata_poverty_prediction.py            # Main solution script
├── sample_data/                                # Input data (not included)
│   ├── train_hh_features.csv
│   ├── train_hh_gt.csv
│   └── test_hh_features.csv
└── submission/                                 # Output predictions (not included)
    ├── predicted_household_consumption.csv
    └── predicted_poverty_distribution.csv
```
