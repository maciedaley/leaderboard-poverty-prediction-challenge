import numpy as np
import pandas as pd
import lightgbm as lgb
import warnings
import json
import os
import glob
import argparse
warnings.filterwarnings("ignore")
os.makedirs("outputs", exist_ok=True)


# ===============================
# Step 1: Load test data
# ===============================

parser = argparse.ArgumentParser(description="Run inference on new data")
parser.add_argument(
        "--input",
        type=str,
        default="data/raw/test_hh_features.csv",
        help="Path to input CSV file"
    )
args = parser.parse_args()

input_path = args.input
X_test = pd.read_csv(input_path)
    
print("Loaded file:", input_path)
print("Test shape:", X_test.shape)



# Keep IDs separate
hh_ids = X_test[[ 'survey_id','hhid']].copy()

# ===============================
# Step 2: Load features for model
# ===============================
with open("models/features.json", "r") as f:
    FEATURES = json.load(f)
X_test_features = X_test[FEATURES].copy()

# Load & Convert categorical columns
with open("models/categorical_cols.json", "r") as f:
    CATEGORICAL_COLS = json.load(f)
for c in CATEGORICAL_COLS:
    X_test_features[c] = X_test_features[c].astype('category')

# ===============================
# Step 3: Predict household consumption
# ===============================
# Average predictions from multiple models



model_files = sorted(glob.glob("models/lgbm_fold_*.txt"))
models = [lgb.Booster(model_file=f) for f in model_files]

test_log_preds = np.mean([m.predict(X_test_features) for m in models], axis=0)

# Convert from log back to USD/day
X_test['cons_pred'] = np.exp(test_log_preds)

# ===============================
# Step 4: Optional quantile calibration
# ===============================
# Use survey 300000 from training as reference

with open("models/reference_consumption.json", "r") as f:
    ref_data = json.load(f)

#REF_SURVEY_ID = ref_data["survey_id"]
ref_cons = np.array(ref_data["values"])

def quantile_map(pred, ref):
    q = np.linspace(0, 1, 1001)
    pred_q = np.quantile(pred, q)
    ref_q = np.quantile(ref, q)
    return np.interp(pred, pred_q, ref_q)

X_test['cons_pred_calib'] = X_test['cons_pred']

for sid in X_test['survey_id'].unique():
    mask = X_test['survey_id'] == sid
    X_test.loc[mask, 'cons_pred_calib'] = quantile_map(
        X_test.loc[mask, 'cons_pred'].values,
        ref_cons
    )

# ===============================
# Step 5: Compute poverty rates
# ===============================
POVERTY_THRESHOLDS = [
    "3.17", "3.94", "4.60", "5.26", "5.88",
    "6.47", "7.06", "7.70", "8.40", "9.13",
    "9.87", "10.70", "11.62", "12.69",
    "14.03", "15.64", "17.76", "20.99", "27.37"
]

def poverty_rates(df, cons_col):
    w = df["weight"]
    rates = {}
    for t_str in POVERTY_THRESHOLDS:
        t = float(t_str)
        rates[f"pct_hh_below_{t_str}"] = (w * (df[cons_col] < t)).sum() / w.sum()
    return rates

poverty_rows = []
for sid, sdf in X_test.groupby("survey_id"):
    row = {"survey_id": sid}
    row.update(poverty_rates(sdf, "cons_pred_calib"))
    poverty_rows.append(row)

poverty_test = pd.DataFrame(poverty_rows)

# Optional check: monotonicity
assert (poverty_test.drop(columns=["survey_id"]).diff(axis=1).min().min() >= 0), "Non-monotonic poverty columns!"

# ===============================
# Step 6: Prepare household submission CSV
# ===============================
hh_submission = hh_ids.copy()
hh_submission['cons_ppp17'] = X_test['cons_pred_calib']
hh_submission.to_csv("outputs/predicted_household_consumption.csv", index=False)

# ===============================
# Step 7: Prepare poverty distribution CSV
# ===============================
poverty_test.to_csv("outputs/predicted_poverty_distribution.csv", index=False)

 

print("Prediction files created successfully!")




print(hh_submission.shape)      # ~103,024 rows
print(poverty_test.shape)       # 3 rows × 20 columns

# Monotonicity check
poverty_test.drop(columns=["survey_id"]).diff(axis=1).min().min()

 

