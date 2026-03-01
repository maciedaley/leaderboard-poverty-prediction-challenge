# Solution - Poverty Prediction Challenge

Username: dwivedy045

## Summary

This project implements a **LightGBM-based machine learning pipeline** for structured/tabular data prediction. The workflow covers:
- Data loading and preprocessing
- Feature engineering
- Cross-validation using grouped folds
- Model training with LightGBM
- Inference and prediction generation

The code is written in **Python** and designed to be reproducible from raw data to final predictions.

## Dependencies
All dependencies are listed with fixed versions to ensure reproducibility.

**requirements.txt**
```
numpy==1.26.4
pandas==2.2.0
lightgbm==4.2.0
scikit-learn==1.4.0
```

Python version used: **Python 3.11.0**

---

# Setup

1. Navigate to the Project Folder poverty_prediction_model. Create an environment using Python 3.11.0.
```bash
python -m venv venv
source venv/bin/activate  # Linux / Mac
venv\Scripts\activate   # Windows
```

2. Install the required Python packages:
```
pip install -r requirements.txt
```

3. Download the data from the competition page into `data/raw`

The structure of the directory before running training or inference should be:
```
poverty_prediction_model
├── data
│   └── raw             <- The original, immutable data dump.
│       ├── test_hh_features.csv
│       ├── train_hh_features.csv
│       ├── train_hh_gt.csv
│       └── train_rates_gt.csv
├── models              <- Trained and metadata models.
│   ├── lgbm_fold_0.txt
│   ├── lgbm_fold_1.txt
│   ├── lgbm_fold_2.txt
│   ├── features.json
│   ├── categorical_cols.json
│   └── reference_consumption.json
├── outputs             <- Predictions will be saved to outputs.
│   ├── predicted_household_consumption.csv
│   └── predicted_poverty_distribution.csv
├── src                 <- Source code for use in this project.
│   ├── __init__.py
│   ├── run_inference.py
│   └── run_training.py
├── README.md            <- The top-level README for developers using this project.
└── requirements.txt     <- The requirements file for reproducing the analysis environment.

```

# Hardware

The solution was run on Windows 10 Home.
- Number of CPUs: 4
- Processor: Intel(R) Core(TM) i5-7300HQ CPU @ 2.50GHz
- Memory: 8.00 GB

Both training and inference were run on CPU.
- Training time: ~1-2 minutes
- Inference time: ~1 minute

# Main Entry Points

## Run Training
Important Note: Virtual environment must be activated before running these commands.

```bash
python src/run_training.py
```

## Run Inference
Important Note: Virtual environment must be activated before running these commands.
```bash
python src/run_inference.py
```

Outputs:
- `outputs/predicted_household_consumption.csv`
- `outputs/predicted_poverty_distribution.csv`

---

# Core Imports Used
```python
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
import warnings
warnings.filterwarnings("ignore")
```

---

# Model Artifacts

Trained model weights:
```
models/lgbm_fold_0.txt
models/lgbm_fold_1.txt
models/lgbm_fold_2.txt
```

Additional required metadata:
```
models/
├── features.json
├── categorical_cols.json
└── reference_consumption.json
```

Important Note:
Inference will fail or produce incorrect results if any required metadata files are missing.

---

# Performing Inference on a New Data Point

## Step 1: Prepare Input Data

Create a CSV file with the same structure as `test_hh_features.csv`.

### Important Requirements

-   The file **must contain all features** listed in
    `models/features.json`
-   The file **must include categorical columns** listed in
    `models/categorical_cols.json`
-   Column names must **match exactly**
-   Data types for categorical columns must be compatible with training

------------------------------------------------------------------------

## Step 2: Place File in `data/raw`

Example:

    data/raw/new-test-data.csv

------------------------------------------------------------------------

## Step 3: Run Inference

``` bash
python src/run_inference.py --input data/raw/new-test-data.csv
```

------------------------------------------------------------------------

## Step 4: Output

Predictions will be saved to:

- `outputs/predicted_household_consumption.csv`
- `outputs/predicted_poverty_distribution.csv`

------------------------------------------------------------------------

## Notes

-   The trained LightGBM models located in the `models/` directory are
    automatically loaded.
-   The script applies the same feature ordering used during training.
-   Ensure that metadata files (`features.json`,
    `categorical_cols.json`, `reference_consumption.json`) are present
    in the `models/` directory.

---



