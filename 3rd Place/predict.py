"""
predict.py — Inference script for Poverty Prediction Challenge winner solution.

Loads saved model weights from the saved_models/ directory and generates
predictions on new household feature data WITHOUT retraining.

Usage:
    python predict.py --test test_hh_features.csv --output submission.zip

    # With a custom saved_models directory:
    python predict.py --test test_hh_features.csv --output submission.zip \
                      --models_dir ./saved_models

Requirements:
    - saved_models/ directory produced by running solution.py
    - Same Python environment as solution.py (see requirements.txt)
"""

import argparse
import os
import json
import zipfile
import warnings
import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from scipy.optimize import differential_evolution
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION (must match solution.py)
# =============================================================================
THRESHOLDS = ['3.17', '3.94', '4.60', '5.26', '5.88', '6.47', '7.06', '7.70',
              '8.40', '9.13', '9.87', '10.70', '11.62', '12.69', '14.03',
              '15.64', '17.76', '20.99', '27.37']
THRESHOLDS_NUMERIC = np.array([float(t) for t in THRESHOLDS])
COMPETITION_WEIGHTS = np.array([1 - abs(0.4 - (i+1)/20) for i in range(19)])

# =============================================================================
# ARGUMENT PARSING
# =============================================================================
parser = argparse.ArgumentParser(description='Run poverty prediction inference on new data.')
parser.add_argument('--test',       default=None,             help='Path to test_hh_features.csv')
parser.add_argument('--output',     default='submission.zip', help='Output zip filename')
parser.add_argument('--models_dir', default='saved_models',   help='Path to saved_models/ directory')
parser.add_argument('--weights',    default=None,
                    help='Optional: path to survey weights CSV (survey_id, hhid, weight). '
                         'If omitted, equal weights are assumed.')

# ── Jupyter / IPython compatibility ──────────────────────────────────────────
# argparse reads sys.argv which in Jupyter contains kernel flags, not our args.
# Detect notebook environment and fall back to safe defaults.
import sys
_in_notebook = 'ipykernel' in sys.modules or hasattr(__builtins__, '__IPYTHON__')

if _in_notebook:
    args = parser.parse_args(args=[])   # ignore sys.argv entirely in notebooks
else:
    args = parser.parse_args()

# ── Resolve --test path ───────────────────────────────────────────────────────
# If not passed on the command line (e.g. running inside a notebook),
# fall back to the standard competition filename or Kaggle path.
if args.test is None:
    for candidate in ['test_hh_features.csv',
                      '/kaggle/input/lionee/test_hh_features.csv']:
        if os.path.exists(candidate):
            args.test = candidate
            break
    if args.test is None:
        raise FileNotFoundError(
            "Could not find test_hh_features.csv automatically.\n"
            "Please pass the path explicitly:  --test /path/to/test_hh_features.csv"
        )

MODELS_DIR = args.models_dir
print("="*70)
print("POVERTY PREDICTION — INFERENCE MODE (no retraining)")
print("="*70)
print(f"  Test data:   {args.test}")
print(f"  Models dir:  {MODELS_DIR}")
print(f"  Output:      {args.output}")

# =============================================================================
# 1. VALIDATE SAVED MODELS DIRECTORY
# =============================================================================
required_files = [
    'config.json', 'feature_list.pkl', 'label_encoders.pkl',
    'survey_poverty_targets.json', 'survey_consumption_stats.json',
    'calibration_params.json'
]
missing = [f for f in required_files if not os.path.exists(f'{MODELS_DIR}/{f}')]
if missing:
    raise FileNotFoundError(
        f"Missing required model files in {MODELS_DIR}/: {missing}\n"
        f"Please run solution.py first to generate saved model artifacts."
    )

# =============================================================================
# 2. LOAD METADATA
# =============================================================================
print("\nLoading model metadata...")

with open(f'{MODELS_DIR}/config.json') as f:
    config = json.load(f)
n_folds = config['n_folds']

features = joblib.load(f'{MODELS_DIR}/feature_list.pkl')
label_encoders = joblib.load(f'{MODELS_DIR}/label_encoders.pkl')

with open(f'{MODELS_DIR}/survey_poverty_targets.json') as f:
    survey_poverty_targets = json.load(f)
with open(f'{MODELS_DIR}/survey_consumption_stats.json') as f:
    survey_consumption_stats = json.load(f)
with open(f'{MODELS_DIR}/calibration_params.json') as f:
    calibration_params = json.load(f)

print(f"  ✓ Config: {n_folds} folds, {len(features)} features")
print(f"  ✓ Encoders for {len(label_encoders)} categorical columns")
print(f"  ✓ Reference data for {len(survey_consumption_stats)} training surveys")

# =============================================================================
# 3. LOAD TRAINED MODELS
# =============================================================================
print("\nLoading trained models from disk...")

models_lgb_log, models_lgb_raw, models_xgb_log, models_cat_log = [], [], [], []

for fold in range(n_folds):
    lgb_log = joblib.load(f'{MODELS_DIR}/lgb_log_fold{fold}.pkl')
    lgb_raw = joblib.load(f'{MODELS_DIR}/lgb_raw_fold{fold}.pkl')
    
    xgb_model = xgb.XGBRegressor()
    xgb_model.load_model(f'{MODELS_DIR}/xgb_log_fold{fold}.json')
    
    cat_model = CatBoostRegressor()
    cat_model.load_model(f'{MODELS_DIR}/cat_log_fold{fold}.cbm')
    
    models_lgb_log.append(lgb_log)
    models_lgb_raw.append(lgb_raw)
    models_xgb_log.append(xgb_model)
    models_cat_log.append(cat_model)

print(f"  ✓ Loaded {n_folds} folds × 4 models = {n_folds * 4} total models")

# =============================================================================
# 4. LOAD & PREPROCESS NEW TEST DATA
# =============================================================================
print("\nLoading and preprocessing test data...")

test = pd.read_csv(args.test)
print(f"  Test shape: {test.shape}")
print(f"  Surveys: {test['survey_id'].nunique()}")

# Optionally load weights
if args.weights:
    weights_df = pd.read_csv(args.weights)
    test = test.merge(weights_df, on=['survey_id', 'hhid'], how='left')
    test['weight'] = test['weight'].fillna(1.0)
elif 'weight' not in test.columns:
    print("  ⚠ No weights file provided — using equal weights (1.0 per household)")
    test['weight'] = 1.0

# Apply same feature engineering as solution.py
def ultimate_features(df):
    df = df.copy()
    if 'hsize' in df.columns:
        df['hsize'] = df['hsize'].replace(0, 1)
    if 'sector1d' in df.columns:
        df['sector1d'] = df['sector1d'].fillna('Unemployed')
    if 'dweltyp' in df.columns:
        df['dweltyp'] = df['dweltyp'].fillna('Unknown')
    if 'educ_max' in df.columns:
        df['educ_max'] = df['educ_max'].fillna('None')
    if 'utl_exp_ppp17' in df.columns:
        df['utl_exp_ppp17'] = df['utl_exp_ppp17'].fillna(df['utl_exp_ppp17'].median())

    binary_map = {
        'Yes': 1, 'No': 0, 'Male': 1, 'Female': 0,
        'Urban': 1, 'Rural': 0, 'Employed': 1, 'Not employed': 0,
        'Owner': 1, 'Not owner': 0, 'Access': 1, 'No access': 0
    }
    binary_cols = ['male', 'urban', 'employed', 'owner', 'any_nonagric', 'water', 'toilet', 'sewer', 'elect']
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].map(binary_map).fillna(0).astype(float)

    consum_cols = [c for c in df.columns if 'consumed' in c.lower()]
    for col in consum_cols:
        if df[col].dtype == 'object':
            df[col] = df[col].map(binary_map).fillna(0).astype(float)

    if 'utl_exp_ppp17' in df.columns and 'hsize' in df.columns:
        df['utl_exp_per_capita'] = df['utl_exp_ppp17'] / df['hsize']
        df['log_utl_exp_pc'] = np.log1p(df['utl_exp_per_capita'])
        df['sqrt_utl_exp_pc'] = np.sqrt(df['utl_exp_per_capita'])
        df['has_utility_exp'] = (df['utl_exp_ppp17'] > 0).astype(int)

    if consum_cols:
        df['consumption_diversity'] = df[consum_cols].sum(axis=1)
        df['consumption_rate'] = df['consumption_diversity'] / len(consum_cols)
        df['log_consumption_diversity'] = np.log1p(df['consumption_diversity'])
        df['sqrt_consumption_diversity'] = np.sqrt(df['consumption_diversity'])
        if 'hsize' in df.columns:
            df['consumption_per_capita'] = df['consumption_diversity'] / df['hsize']
            df['log_cons_per_capita'] = np.log1p(df['consumption_per_capita'])

        staples = [c for c in consum_cols if any(x in c for x in ['100','300','1500','1600','1900'])]
        proteins = [c for c in consum_cols if any(x in c for x in ['400','700','800','900','2000','2100','2400'])]
        luxury   = [c for c in consum_cols if any(x in c for x in ['2500','2600','2700','2800','2900','3000'])]

        if staples:
            df['staples_consumed'] = df[staples].sum(axis=1)
            df['log_staples'] = np.log1p(df['staples_consumed'])
            df['staples_rate'] = df['staples_consumed'] / len(staples)
        if proteins:
            df['proteins_consumed'] = df[proteins].sum(axis=1)
            df['log_proteins'] = np.log1p(df['proteins_consumed'])
            df['has_protein'] = (df['proteins_consumed'] > 0).astype(int)
            df['proteins_rate'] = df['proteins_consumed'] / len(proteins)
            if 'hsize' in df.columns:
                df['protein_per_capita'] = df['proteins_consumed'] / df['hsize']
        if luxury:
            df['luxury_consumed'] = df[luxury].sum(axis=1)
            df['has_luxury'] = (df['luxury_consumed'] > 0).astype(int)
        if 'staples_consumed' in df.columns and 'proteins_consumed' in df.columns:
            df['protein_to_staple_ratio'] = (df['proteins_consumed'] + 1) / (df['staples_consumed'] + 1)
        try:
            low_items  = [c for c in consum_cols if int(c.replace('consumed','')) < 1000]
            high_items = [c for c in consum_cols if int(c.replace('consumed','')) >= 3000]
            if low_items:  df['cons_low']  = df[low_items].sum(axis=1)
            if high_items: df['cons_high'] = df[high_items].sum(axis=1)
        except: pass

    infra_cols = ['water','toilet','sewer','elect']
    available_infra = [c for c in infra_cols if c in df.columns]
    if available_infra:
        df['infra_score'] = df[available_infra].sum(axis=1)
        df['infra_rate'] = df['infra_score'] / len(available_infra)
        df['has_full_infra'] = (df['infra_score'] == len(available_infra)).astype(int)

    asset_cols = ['water','toilet','sewer','elect','owner']
    available_assets = [c for c in asset_cols if c in df.columns]
    if available_assets:
        df['asset_index'] = df[available_assets].sum(axis=1)
        if 'urban' in df.columns:
            df['asset_x_urban'] = df['asset_index'] * df['urban']

    if 'hsize' in df.columns:
        df['log_hsize'] = np.log1p(df['hsize'])
        df['hsize_sq'] = df['hsize'] ** 2
        df['hsize_inv'] = 1 / (df['hsize'] + 1)
        if 'num_children5' in df.columns and 'num_elderly' in df.columns:
            total_dependents = df['num_children5'] + df['num_elderly']
            df['total_dependents'] = total_dependents
            df['dependency_ratio'] = total_dependents / df['hsize']
        if 'num_adult_male' in df.columns and 'num_adult_female' in df.columns:
            total_adults = df['num_adult_male'] + df['num_adult_female']
            df['total_adults'] = total_adults
            df['adult_density'] = total_adults / df['hsize']
            df['children_per_adult'] = df['num_children5'] / (total_adults + 1)
            df['female_ratio'] = df['num_adult_female'] / (total_adults + 1)
        if 'num_children5' in df.columns:
            df['has_infant'] = (df['num_children5'] > 0).astype(int)
            df['many_children'] = (df['num_children5'] >= 3).astype(int)

    if 'educ_max' in df.columns and df['educ_max'].dtype != 'object':
        df['educ_max'] = pd.to_numeric(df['educ_max'], errors='coerce').fillna(0)
        df['log_educ'] = np.log1p(df['educ_max'])
        df['has_education'] = (df['educ_max'] > 0).astype(int)
        df['high_education'] = (df['educ_max'] >= 12).astype(int)

    if 'sworkershh' in df.columns:
        df['sworkershh'] = pd.to_numeric(df['sworkershh'], errors='coerce').fillna(0)
        df['log_workers'] = np.log1p(df['sworkershh'])
        df['has_workers'] = (df['sworkershh'] > 0).astype(int)
        if 'hsize' in df.columns:
            df['worker_ratio'] = df['sworkershh'] / (df['hsize'] + 1)

    if 'urban' in df.columns and 'educ_max' in df.columns:
        if pd.api.types.is_numeric_dtype(df['educ_max']):
            df['urban_x_educ'] = df['urban'] * df['educ_max']
            df['rural_low_educ'] = ((1 - df['urban']) * (df['educ_max'] < 6)).astype(int)

    if 'male' in df.columns and 'num_children5' in df.columns:
        df['female_x_children'] = (1 - df['male']) * df['num_children5']
        if 'has_infant' in df.columns:
            df['female_head_with_infant'] = ((1 - df['male']) * df['has_infant']).astype(int)

    return df

test = ultimate_features(test)

# Apply saved label encoders
for col, le in label_encoders.items():
    if col in test.columns:
        # Handle unseen categories gracefully
        known = set(le.classes_)
        test[col] = test[col].astype(str).apply(lambda x: x if x in known else le.classes_[0])
        test[col] = le.transform(test[col].astype(str))

# Align to exact feature list from training
test_X = test[features].fillna(-999)
print(f"  ✓ Feature matrix ready: {test_X.shape}")

# =============================================================================
# 5. ENSEMBLE PREDICTION WITH TTA
# =============================================================================
print("\nRunning TTA inference (3 passes)...")

def predict_ensemble(data_X):
    preds = np.zeros(len(data_X))
    for i in range(n_folds):
        p1 = np.expm1(models_lgb_log[i].predict(data_X))
        p2 = models_lgb_raw[i].predict(data_X)
        p3 = np.expm1(models_xgb_log[i].predict(data_X))
        p4 = np.expm1(models_cat_log[i].predict(data_X))
        preds += (p1 * 0.30 + p2 * 0.20 + p3 * 0.25 + p4 * 0.25) / n_folds
    return preds

pred_original = predict_ensemble(test_X)

test_X_down = test_X.copy()
if 'utl_exp_ppp17' in test_X_down.columns:
    utl_idx = test_X_down.columns.get_loc('utl_exp_ppp17')
    test_X_down.iloc[:, utl_idx] *= 0.97
    if 'utl_exp_per_capita' in test_X_down.columns:
        hsize_idx = test_X_down.columns.get_loc('hsize')
        utl_pc_idx = test_X_down.columns.get_loc('utl_exp_per_capita')
        test_X_down.iloc[:, utl_pc_idx] = test_X_down.iloc[:, utl_idx] / test_X_down.iloc[:, hsize_idx]
    if 'log_utl_exp_pc' in test_X_down.columns:
        test_X_down.iloc[:, test_X_down.columns.get_loc('log_utl_exp_pc')] = \
            np.log1p(test_X_down.iloc[:, test_X_down.columns.get_loc('utl_exp_per_capita')])
pred_down = predict_ensemble(test_X_down)

test_X_up = test_X.copy()
if 'utl_exp_ppp17' in test_X_up.columns:
    utl_idx = test_X_up.columns.get_loc('utl_exp_ppp17')
    test_X_up.iloc[:, utl_idx] *= 1.03
    if 'utl_exp_per_capita' in test_X_up.columns:
        hsize_idx = test_X_up.columns.get_loc('hsize')
        utl_pc_idx = test_X_up.columns.get_loc('utl_exp_per_capita')
        test_X_up.iloc[:, utl_pc_idx] = test_X_up.iloc[:, utl_idx] / test_X_up.iloc[:, hsize_idx]
    if 'log_utl_exp_pc' in test_X_up.columns:
        test_X_up.iloc[:, test_X_up.columns.get_loc('log_utl_exp_pc')] = \
            np.log1p(test_X_up.iloc[:, test_X_up.columns.get_loc('utl_exp_per_capita')])
pred_up = predict_ensemble(test_X_up)

raw_preds = (pred_original + pred_down + pred_up) / 3
print(f"  ✓ Raw predictions mean: {raw_preds.mean():.2f}")

# =============================================================================
# 6. APPLY CALIBRATION
# =============================================================================
print("\nApplying saved calibration parameters...")

def find_best_matching_survey(test_stats, train_stats_dict):
    best_sid, best_score = None, float('inf')
    for train_sid, train_stats in train_stats_dict.items():
        mean_sim = abs(test_stats['mean'] - train_stats['mean_cons']) / test_stats['mean']
        p40_sim  = abs(test_stats['p40']  - train_stats['p40'])        / test_stats['p40']
        score = mean_sim * 0.40 + p40_sim * 0.60
        if score < best_score:
            best_score, best_sid = score, train_sid
    return best_sid, best_score

def calibration_objective(params, preds, weights, target_rates, thresholds):
    scale, shift, gamma = params
    adjusted = np.maximum(preds ** gamma * scale + shift, 0.1)
    total_weight = weights.sum()
    pred_rates = np.array([weights[adjusted < t].sum() / total_weight for t in thresholds])
    wmape = np.sum(COMPETITION_WEIGHTS * np.abs(pred_rates - target_rates) / (target_rates + 1e-10)) \
            / np.sum(COMPETITION_WEIGHTS)
    return wmape + 0.0005 * (abs(gamma - 1.0) + abs(scale - 1.0))

final_preds = raw_preds.copy()

for sid in test['survey_id'].unique():
    mask = test['survey_id'] == sid
    survey_preds = raw_preds[mask]
    survey_weights = test.loc[mask, 'weight'].values

    str_sid = str(sid)

    # Use saved calibration if this survey was in the training test set
    if str_sid in calibration_params:
        p = calibration_params[str_sid]
        scale, shift, gamma = p['scale'], p['shift'], p['gamma']
        print(f"  Survey {sid}: using saved calibration (scale={scale:.4f}, γ={gamma:.4f})")
    else:
        # New survey not seen during training — find best match and re-optimise
        print(f"  Survey {sid}: new survey — running fresh calibration...")
        test_stats = {
            'mean': float(survey_preds.mean()),
            'p40': float(np.percentile(survey_preds, 40))
        }
        ref_sid, _ = find_best_matching_survey(test_stats, survey_consumption_stats)
        target_rates = np.array(survey_poverty_targets[ref_sid]['rates'])

        result = differential_evolution(
            calibration_objective,
            bounds=[(0.5, 1.5), (-5, 5), (0.8, 1.2)],
            args=(survey_preds, survey_weights, target_rates, THRESHOLDS_NUMERIC),
            seed=42, maxiter=80, popsize=15, workers=1, polish=True
        )
        scale, shift, gamma = result.x
        print(f"    → ref={ref_sid}, scale={scale:.4f}, γ={gamma:.4f}, WMAPE={result.fun:.6f}")

    calibrated = np.maximum(survey_preds ** gamma * scale + shift, 0.1)
    final_preds[mask] = calibrated

final_preds = np.clip(final_preds, 0.5, 150)
print(f"\n  ✓ Calibration complete. Final mean: {final_preds.mean():.2f}, P40: {np.percentile(final_preds, 40):.2f}")

# =============================================================================
# 7. BUILD & SAVE SUBMISSION
# =============================================================================
print("\nBuilding submission files...")

sub_cons = pd.DataFrame({
    'survey_id': test['survey_id'].values,
    'hhid': test['hhid'].values,
    'cons_ppp17': final_preds
})

poverty_rows = []
for sid in test['survey_id'].unique():
    mask = test['survey_id'] == sid
    preds   = final_preds[mask]
    weights = test.loc[mask, 'weight'].values
    total_w = weights.sum()
    row = {'survey_id': sid}
    for t_str, t_num in zip(THRESHOLDS, THRESHOLDS_NUMERIC):
        row[f'pct_hh_below_{t_str}'] = np.clip(weights[preds < t_num].sum() / total_w, 0, 1)
    poverty_rows.append(row)

sub_pov = pd.DataFrame(poverty_rows)
pov_cols = ['survey_id'] + [f'pct_hh_below_{t}' for t in THRESHOLDS]
sub_pov = sub_pov[pov_cols]

cons_path = args.output.replace('.zip', '_household_consumption.csv')
pov_path  = args.output.replace('.zip', '_poverty_distribution.csv')
sub_cons.to_csv(cons_path, index=False)
sub_pov.to_csv(pov_path,  index=False)

with zipfile.ZipFile(args.output, 'w', zipfile.ZIP_DEFLATED) as zf:
    zf.write(cons_path)
    zf.write(pov_path)

print(f"\n{'='*70}")
print("INFERENCE COMPLETE")
print(f"{'='*70}")
print(f"  {cons_path}")
print(f"  {pov_path}")
print(f"  {args.output}")
print(f"\nHouseholds predicted: {len(sub_cons)}")
print(f"Surveys:              {sub_cons['survey_id'].nunique()}")
