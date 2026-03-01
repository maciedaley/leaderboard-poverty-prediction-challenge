import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
import warnings
import os
import json
warnings.filterwarnings("ignore")
os.makedirs("models", exist_ok=True)


SEED = 42
np.random.seed(SEED)

#1 Load data
X_train = pd.read_csv("data/raw/train_hh_features.csv")
y_train = pd.read_csv("data/raw/train_hh_gt.csv")
rates_gt = pd.read_csv("data/raw/train_rates_gt.csv")

# Merge features + target
train = X_train.merge(
    y_train,
    on=["survey_id", "hhid"],
    how="inner"
)


#2
train["log_cons"] = np.log(train["cons_ppp17"].clip(lower=1e-3))

#3 Drop coulmns
DROP_COLS = [
    "hhid",
    "cons_ppp17",
    "log_cons"
]

FEATURES = [c for c in train.columns if c not in DROP_COLS]
#Save Metadata for Inference
with open("models/features.json", "w") as f:
    json.dump(FEATURES, f)

#4 Create category columns
CATEGORICAL_COLS = [
    c for c in FEATURES
    if train[c].dtype == "object"
]
with open("models/categorical_cols.json", "w") as f:
    json.dump(CATEGORICAL_COLS, f)

for c in CATEGORICAL_COLS:
    train[c] = train[c].astype("category")
    
#5 Grouped Cross Validation   
gkf = GroupKFold(n_splits=3)
groups = train["survey_id"]

#LightGBM Parameters
params = {
    "objective": "regression_l1",
    "learning_rate": 0.05,
    "num_leaves": 64,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "seed": SEED,
    "verbosity": -1
}

models = []

for fold, (tr_idx, val_idx) in enumerate(gkf.split(train, groups=groups)):
    print(f"Training fold {fold+1}")

    tr = train.iloc[tr_idx]
    val = train.iloc[val_idx]

    dtrain = lgb.Dataset(
        tr[FEATURES],
        tr["log_cons"],
        categorical_feature=CATEGORICAL_COLS
    )

    dval = lgb.Dataset(
        val[FEATURES],
        val["log_cons"],
        categorical_feature=CATEGORICAL_COLS
    )

    
    model = lgb.train(
    params,
    dtrain,
    valid_sets=[dval],
    num_boost_round=4000,
    callbacks=[
        lgb.early_stopping(stopping_rounds=200),
        lgb.log_evaluation(200)
        ]
    )
    models.append(model)
    # SAVE EACH MODEL
    model.save_model(f"models/lgbm_fold_{fold}.txt")
    

#6 Train model
train["log_cons_pred"] = np.mean(
    [m.predict(train[FEATURES]) for m in models],
    axis=0
)

train["cons_pred"] = np.exp(train["log_cons_pred"])

#7  Compute poverty rates
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
        rates[f"pct_hh_below_{t_str}"] = (
            (w * (df[cons_col] < t)).sum() / w.sum()
        )

    return rates

 


pred_rows = []

for sid, sdf in train.groupby("survey_id"):
    row = {"survey_id": sid}
    row.update(poverty_rates(sdf, "cons_pred"))
    pred_rows.append(row)

pred_rates = pd.DataFrame(pred_rows)
print(pred_rates)

#8  Compute Wmape
comparison = pred_rates.merge(
    rates_gt,
    on="survey_id",
    suffixes=("_pred", "_true")
)

comparison

def poverty_wmape(pred, true):
    ape = []
    for t in POVERTY_THRESHOLDS:
        p = pred[f"pct_hh_below_{t}_pred"]
        y = true[f"pct_hh_below_{t}_true"]
        ape.append(abs(p - y) / y)
    return np.mean(ape)

print("Poverty WMAPE:", poverty_wmape(
    comparison.filter(like="_pred"),
    comparison.filter(like="_true")
))

#9 Save reference consumption values
REF_SURVEY_ID = 300000
ref_cons_values = train.loc[
    train["survey_id"] == REF_SURVEY_ID,
    "cons_ppp17"
].values

if len(ref_cons_values) == 0:
    raise ValueError(f"No rows found for survey_id={REF_SURVEY_ID}")

ref_payload = {
    "survey_id": REF_SURVEY_ID,
    "values": ref_cons_values.tolist(),   # raw values
}

with open("models/reference_consumption.json", "w") as f:
    json.dump(ref_payload, f, indent=2)


 





