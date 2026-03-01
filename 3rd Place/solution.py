import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder
from scipy.optimize import differential_evolution
import zipfile
import joblib
import os
import json
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================
THRESHOLDS = ['3.17', '3.94', '4.60', '5.26', '5.88', '6.47', '7.06', '7.70', 
              '8.40', '9.13', '9.87', '10.70', '11.62', '12.69', '14.03', 
              '15.64', '17.76', '20.99', '27.37']
THRESHOLDS_NUMERIC = np.array([float(t) for t in THRESHOLDS])

COMPETITION_WEIGHTS = np.array([1 - abs(0.4 - (i+1)/20) for i in range(19)])

# Directory to save all model artifacts
MODELS_DIR = 'saved_models'
os.makedirs(MODELS_DIR, exist_ok=True)

print("="*80)
print("ULTIMATE POVERTY PREDICTION v2.0")
print("Features: Per-Capita + TTA + Dual-Target + P40-Matching")
print("="*80)
print(f"Competition metric focuses on P40 (threshold {THRESHOLDS[7]} = {THRESHOLDS_NUMERIC[7]})")
print(f"Model artifacts will be saved to: ./{MODELS_DIR}/")
print("="*80)

# =============================================================================
# 1. LOAD DATA
# =============================================================================
print("\nLoading data...")
try:
    train = pd.read_csv('/kaggle/input/lionee/train_hh_features.csv')
    test = pd.read_csv('/kaggle/input/lionee/test_hh_features.csv')
    train_gt = pd.read_csv('/kaggle/input/lionee/train_hh_gt.csv')
    train_rates_gt = pd.read_csv('/kaggle/input/lionee/train_rates_gt.csv')
    print("✓ Loaded from Kaggle paths")
except FileNotFoundError:
    try:
        train = pd.read_csv('train_hh_features.csv')
        test = pd.read_csv('test_hh_features.csv')
        train_gt = pd.read_csv('train_hh_gt.csv')
        train_rates_gt = pd.read_csv('train_rates_gt.csv')
        print("✓ Loaded from local paths")
    except FileNotFoundError as e:
        print("ERROR: Cannot find input files. Please check paths.")
        raise e

train = train.merge(train_gt, on=['survey_id', 'hhid'], how='left')
print(f"Train: {train.shape}, Test: {test.shape}")

if train['survey_id'].nunique() < 2:
    raise ValueError(f"ERROR: Only {train['survey_id'].nunique()} survey(s) in training data.")
if 'cons_ppp17' not in train.columns:
    raise ValueError("ERROR: Target variable 'cons_ppp17' not found.")

# =============================================================================
# 2. FEATURE ENGINEERING
# =============================================================================
print("\nFeature engineering...")

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
        'Yes': 1, 'No': 0, 
        'Male': 1, 'Female': 0,
        'Urban': 1, 'Rural': 0,
        'Employed': 1, 'Not employed': 0,
        'Owner': 1, 'Not owner': 0,
        'Access': 1, 'No access': 0
    }
    
    binary_cols = ['male', 'urban', 'employed', 'owner', 'any_nonagric',
                   'water', 'toilet', 'sewer', 'elect']
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
        
        staples = [c for c in consum_cols if any(x in c for x in ['100', '300', '1500', '1600', '1900'])]
        proteins = [c for c in consum_cols if any(x in c for x in ['400', '700', '800', '900', '2000', '2100', '2400'])]
        luxury = [c for c in consum_cols if any(x in c for x in ['2500', '2600', '2700', '2800', '2900', '3000'])]
        
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
            low_items = [c for c in consum_cols if int(c.replace('consumed', '')) < 1000]
            high_items = [c for c in consum_cols if int(c.replace('consumed', '')) >= 3000]
            if low_items:
                df['cons_low'] = df[low_items].sum(axis=1)
            if high_items:
                df['cons_high'] = df[high_items].sum(axis=1)
        except:
            pass
    
    infra_cols = ['water', 'toilet', 'sewer', 'elect']
    available_infra = [c for c in infra_cols if c in df.columns]
    if available_infra:
        df['infra_score'] = df[available_infra].sum(axis=1)
        df['infra_rate'] = df['infra_score'] / len(available_infra)
        df['has_full_infra'] = (df['infra_score'] == len(available_infra)).astype(int)
    
    asset_cols = ['water', 'toilet', 'sewer', 'elect', 'owner']
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
    
    if 'educ_max' in df.columns:
        if df['educ_max'].dtype != 'object':
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

train = ultimate_features(train)
test = ultimate_features(test)
print(f"Feature engineering complete. Train shape: {train.shape}")

# =============================================================================
# 3. SURVEY-LEVEL TARGETS
# =============================================================================
print("\nBuilding survey-level targets...")

survey_poverty_targets = {}
survey_consumption_stats = {}

for sid in train['survey_id'].unique():
    mask = train['survey_id'] == sid
    survey_data = train[mask]
    
    if sid in train_rates_gt['survey_id'].values:
        actual_rates = train_rates_gt[train_rates_gt['survey_id'] == sid].iloc[0, 1:].values
    else:
        weights = survey_data['weight'].values
        consumption = survey_data['cons_ppp17'].values
        total_weight = weights.sum()
        actual_rates = np.array([
            weights[consumption < t].sum() / total_weight 
            for t in THRESHOLDS_NUMERIC
        ])
    
    cons = survey_data['cons_ppp17']
    stats = {
        'rates': actual_rates.tolist(),   # tolist() for JSON serialization
        'mean_cons': float(cons.mean()),
        'median_cons': float(cons.median()),
        'std_cons': float(cons.std()),
        'p40': float(cons.quantile(0.4)),
    }
    survey_poverty_targets[sid] = stats
    survey_consumption_stats[sid] = stats.copy()

print(f"Survey targets built for {len(survey_poverty_targets)} surveys")

# =============================================================================
# 4. PREPARE FEATURES
# =============================================================================
print("\nPreparing features...")

exclude_cols = ['survey_id', 'hhid', 'household_id', 'cons_ppp17', 'weight', 
                'strata', 'com', 'row_id', 'country', 'year', 'sample']
features = [c for c in train.columns if c not in exclude_cols]

# ── SAVE: Label encoders ──────────────────────────────────────────────────────
cat_feats = [c for c in features if train[c].dtype == 'object']
print(f"Encoding {len(cat_feats)} categorical features...")

label_encoders = {}
for col in cat_feats:
    le = LabelEncoder()
    full_list = pd.concat([train[col], test[col]], axis=0).astype(str)
    le.fit(full_list)
    train[col] = le.transform(train[col].astype(str))
    test[col] = le.transform(test[col].astype(str))
    label_encoders[col] = le

# Save label encoders and feature list
joblib.dump(label_encoders, f'{MODELS_DIR}/label_encoders.pkl')
joblib.dump(features, f'{MODELS_DIR}/feature_list.pkl')
print(f"  ✓ Saved label encoders ({len(label_encoders)} columns) → {MODELS_DIR}/label_encoders.pkl")
print(f"  ✓ Saved feature list ({len(features)} features)  → {MODELS_DIR}/feature_list.pkl")

# Save survey metadata (needed by predict.py for calibration)
with open(f'{MODELS_DIR}/survey_poverty_targets.json', 'w') as f:
    json.dump({str(k): v for k, v in survey_poverty_targets.items()}, f, indent=2)
with open(f'{MODELS_DIR}/survey_consumption_stats.json', 'w') as f:
    json.dump({str(k): v for k, v in survey_consumption_stats.items()}, f, indent=2)
print(f"  ✓ Saved survey metadata → {MODELS_DIR}/survey_poverty_targets.json")

X = train[features].fillna(-999)
y_log = np.log1p(train['cons_ppp17'])
y_raw = train['cons_ppp17']
groups = train['survey_id']
test_X = test[features].fillna(-999)

print(f"Total features: {len(features)}")

# =============================================================================
# 5. COMPETITION METRIC
# =============================================================================
def calculate_weighted_mape(predicted_rates, actual_rates):
    errors = np.abs(predicted_rates - actual_rates) / (actual_rates + 1e-10)
    return np.sum(COMPETITION_WEIGHTS * errors) / np.sum(COMPETITION_WEIGHTS)

# =============================================================================
# 6. TRAIN MODELS + SAVE EACH FOLD
# =============================================================================
print("\n" + "="*80)
print("TRAINING MODELS")
print("="*80)

n_unique_surveys = train['survey_id'].nunique()
n_folds = min(5, n_unique_surveys)
print(f"Using {n_folds} folds")
gkf = GroupKFold(n_splits=n_folds)

models_lgb_log = []
models_lgb_raw = []
models_xgb_log = []
models_cat_log = []

for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y_log, groups)):
    print(f"\nTraining Fold {fold + 1}/{n_folds}...")
    
    X_train, y_train_log = X.iloc[train_idx], y_log.iloc[train_idx]
    X_val, y_val_log = X.iloc[val_idx], y_log.iloc[val_idx]
    y_train_raw = y_raw.iloc[train_idx]
    y_val_raw = y_raw.iloc[val_idx]
    
    # LightGBM (Log target)
    lgb_log = lgb.LGBMRegressor(
        objective='regression', metric='mae', learning_rate=0.02,
        num_leaves=95, n_estimators=3000, random_state=42 + fold,
        verbose=-1, n_jobs=-1
    )
    lgb_log.fit(X_train, y_train_log,
                eval_set=[(X_val, y_val_log)],
                callbacks=[lgb.early_stopping(150, verbose=False)])
    models_lgb_log.append(lgb_log)
    joblib.dump(lgb_log, f'{MODELS_DIR}/lgb_log_fold{fold}.pkl')

    # LightGBM (Raw target)
    lgb_raw = lgb.LGBMRegressor(
        objective='regression', metric='mae', learning_rate=0.02,
        num_leaves=95, n_estimators=3000, random_state=42 + fold + 100,
        verbose=-1, n_jobs=-1
    )
    lgb_raw.fit(X_train, y_train_raw,
                eval_set=[(X_val, y_val_raw)],
                callbacks=[lgb.early_stopping(150, verbose=False)])
    models_lgb_raw.append(lgb_raw)
    joblib.dump(lgb_raw, f'{MODELS_DIR}/lgb_raw_fold{fold}.pkl')

    # XGBoost (Log target)
    xgb_log = xgb.XGBRegressor(
        objective='reg:squarederror', learning_rate=0.02,
        max_depth=9, n_estimators=3000, random_state=42 + fold,
        verbosity=0, n_jobs=-1
    )
    xgb_log.fit(X_train, y_train_log,
                eval_set=[(X_val, y_val_log)],
                verbose=False)
    models_xgb_log.append(xgb_log)
    xgb_log.save_model(f'{MODELS_DIR}/xgb_log_fold{fold}.json')

    # CatBoost (Log target)
    cat_log = CatBoostRegressor(
        iterations=3000, learning_rate=0.02, depth=9,
        loss_function='RMSE', random_seed=42 + fold,
        verbose=False, thread_count=-1
    )
    cat_log.fit(X_train, y_train_log,
                eval_set=(X_val, y_val_log),
                verbose=False)
    models_cat_log.append(cat_log)
    cat_log.save_model(f'{MODELS_DIR}/cat_log_fold{fold}.cbm')

    print(f"  ✓ Fold {fold+1} models saved to {MODELS_DIR}/")

# Save fold count so predict.py knows how many to load
with open(f'{MODELS_DIR}/config.json', 'w') as f:
    json.dump({'n_folds': n_folds, 'n_features': len(features)}, f, indent=2)

print(f"\n✓ All {n_folds} folds × 4 models saved → {MODELS_DIR}/")
print(f"  Files: lgb_log_fold*.pkl, lgb_raw_fold*.pkl, xgb_log_fold*.json, cat_log_fold*.cbm")

# =============================================================================
# 7. TEST-TIME AUGMENTATION (TTA)
# =============================================================================
print("\n" + "="*80)
print("TEST-TIME AUGMENTATION (TTA)")
print("="*80)

def predict_ensemble_on_data(data_X):
    predictions = np.zeros(len(data_X))
    n_models = len(models_lgb_log)
    for i in range(n_models):
        p1 = np.expm1(models_lgb_log[i].predict(data_X))
        p2 = models_lgb_raw[i].predict(data_X)
        p3 = np.expm1(models_xgb_log[i].predict(data_X))
        p4 = np.expm1(models_cat_log[i].predict(data_X))
        fold_pred = (p1 * 0.30 + p2 * 0.20 + p3 * 0.25 + p4 * 0.25)
        predictions += fold_pred / n_models
    return predictions

print("  1/3: Original test data...")
pred_original = predict_ensemble_on_data(test_X)

print("  2/3: Utility expense -3%...")
test_X_down = test_X.copy()
if 'utl_exp_ppp17' in test_X_down.columns:
    utl_idx = test_X_down.columns.get_loc('utl_exp_ppp17')
    test_X_down.iloc[:, utl_idx] *= 0.97
    if 'utl_exp_per_capita' in test_X_down.columns:
        hsize_idx = test_X_down.columns.get_loc('hsize')
        utl_pc_idx = test_X_down.columns.get_loc('utl_exp_per_capita')
        test_X_down.iloc[:, utl_pc_idx] = test_X_down.iloc[:, utl_idx] / test_X_down.iloc[:, hsize_idx]
    if 'log_utl_exp_pc' in test_X_down.columns:
        log_utl_pc_idx = test_X_down.columns.get_loc('log_utl_exp_pc')
        test_X_down.iloc[:, log_utl_pc_idx] = np.log1p(test_X_down.iloc[:, utl_pc_idx])
pred_down = predict_ensemble_on_data(test_X_down)

print("  3/3: Utility expense +3%...")
test_X_up = test_X.copy()
if 'utl_exp_ppp17' in test_X_up.columns:
    utl_idx = test_X_up.columns.get_loc('utl_exp_ppp17')
    test_X_up.iloc[:, utl_idx] *= 1.03
    if 'utl_exp_per_capita' in test_X_up.columns:
        hsize_idx = test_X_up.columns.get_loc('hsize')
        utl_pc_idx = test_X_up.columns.get_loc('utl_exp_per_capita')
        test_X_up.iloc[:, utl_pc_idx] = test_X_up.iloc[:, utl_idx] / test_X_up.iloc[:, hsize_idx]
    if 'log_utl_exp_pc' in test_X_up.columns:
        log_utl_pc_idx = test_X_up.columns.get_loc('log_utl_exp_pc')
        test_X_up.iloc[:, log_utl_pc_idx] = np.log1p(test_X_up.iloc[:, utl_pc_idx])
pred_up = predict_ensemble_on_data(test_X_up)

test_preds_ensemble = (pred_original + pred_down + pred_up) / 3
print(f"\n✓ TTA complete. Final mean: {test_preds_ensemble.mean():.2f}")

# =============================================================================
# 8. SURVEY MATCHING
# =============================================================================
print("\n" + "="*80)
print("SURVEY MATCHING (P40-Focused)")
print("="*80)

def find_best_matching_survey(test_stats, train_stats_dict):
    best_sid, best_score = None, float('inf')
    for train_sid, train_stats in train_stats_dict.items():
        mean_sim = abs(test_stats['mean'] - train_stats['mean_cons']) / test_stats['mean']
        p40_sim = abs(test_stats['p40'] - train_stats['p40']) / test_stats['p40']
        combined_score = mean_sim * 0.40 + p40_sim * 0.60
        if combined_score < best_score:
            best_score = combined_score
            best_sid = train_sid
    return best_sid, best_score

test_survey_stats = {}
for sid in test['survey_id'].unique():
    mask = test['survey_id'] == sid
    preds = test_preds_ensemble[mask]
    test_survey_stats[sid] = {
        'mean': preds.mean(),
        'median': np.median(preds),
        'p40': np.percentile(preds, 40),
    }

survey_matches = {}
for test_sid, test_stats in test_survey_stats.items():
    best_train_sid, similarity = find_best_matching_survey(test_stats, survey_consumption_stats)
    survey_matches[test_sid] = {'reference': best_train_sid, 'similarity': similarity}
    print(f"Survey {test_sid} -> Reference {best_train_sid} (similarity: {similarity:.4f})")

# Save survey matches
with open(f'{MODELS_DIR}/survey_matches.json', 'w') as f:
    json.dump({str(k): {'reference': str(v['reference']), 'similarity': v['similarity']}
               for k, v in survey_matches.items()}, f, indent=2)
print(f"✓ Saved survey matches → {MODELS_DIR}/survey_matches.json")

# =============================================================================
# 9. CALIBRATION
# =============================================================================
print("\n" + "="*80)
print("CALIBRATION")
print("="*80)

def calibration_objective(params, preds, weights, target_rates, thresholds):
    scale, shift, gamma = params
    adjusted = np.maximum(preds ** gamma * scale + shift, 0.1)
    total_weight = weights.sum()
    pred_rates = np.array([weights[adjusted < t].sum() / total_weight for t in thresholds])
    wmape = calculate_weighted_mape(pred_rates, target_rates)
    penalty = 0.0005 * (abs(gamma - 1.0) + abs(scale - 1.0))
    return wmape + penalty

calibrated_preds = test_preds_ensemble.copy()
calibration_results = {}

for sid in test['survey_id'].unique():
    mask = test['survey_id'] == sid
    survey_preds = test_preds_ensemble[mask]
    survey_weights = test.loc[mask, 'weight'].values
    ref_sid = survey_matches[sid]['reference']
    target_rates = np.array(survey_poverty_targets[ref_sid]['rates'])
    
    print(f"\nCalibrating Survey {sid} (reference: {ref_sid})...")
    
    result = differential_evolution(
        calibration_objective,
        bounds=[(0.5, 1.5), (-5, 5), (0.8, 1.2)],
        args=(survey_preds, survey_weights, target_rates, THRESHOLDS_NUMERIC),
        seed=42, maxiter=80, popsize=15, atol=1e-6, tol=1e-6,
        workers=1, polish=True
    )
    
    scale, shift, gamma = result.x
    calibrated = np.maximum(survey_preds ** gamma * scale + shift, 0.1)
    calibrated_preds[mask] = calibrated
    
    calibration_results[sid] = {
        'scale': float(scale), 'shift': float(shift), 'gamma': float(gamma),
        'expected_wmape': float(result.fun),
        'reference': str(ref_sid),
        'similarity': survey_matches[sid]['similarity']
    }
    print(f"  scale={scale:.4f}, shift={shift:.4f}, gamma={gamma:.4f}, WMAPE={result.fun:.6f}")

# Save calibration parameters so predict.py can apply them to new data
with open(f'{MODELS_DIR}/calibration_params.json', 'w') as f:
    json.dump({str(k): v for k, v in calibration_results.items()}, f, indent=2)
print(f"\n✓ Saved calibration params → {MODELS_DIR}/calibration_params.json")

# =============================================================================
# 10. SUBMISSION
# =============================================================================
print("\n" + "="*80)
print("GENERATING SUBMISSION")
print("="*80)

final_preds = np.clip(calibrated_preds, 0.5, 150)

sub_cons = pd.DataFrame({
    'survey_id': test['survey_id'],
    'hhid': test['hhid'],
    'cons_ppp17': final_preds
})

poverty_rows = []
for sid in test['survey_id'].unique():
    mask = test['survey_id'] == sid
    preds = final_preds[mask]
    weights = test.loc[mask, 'weight'].values
    total_weight = weights.sum()
    row = {'survey_id': sid}
    for t_str, t_num in zip(THRESHOLDS, THRESHOLDS_NUMERIC):
        row[f'pct_hh_below_{t_str}'] = np.clip(weights[preds < t_num].sum() / total_weight, 0, 1)
    poverty_rows.append(row)

sub_pov = pd.DataFrame(poverty_rows)
pov_cols = ['survey_id'] + [f'pct_hh_below_{t}' for t in THRESHOLDS]
sub_pov = sub_pov[pov_cols]

sub_cons.to_csv('predicted_household_consumption.csv', index=False)
sub_pov.to_csv('predicted_poverty_distribution.csv', index=False)
with zipfile.ZipFile('submission.zip', 'w', zipfile.ZIP_DEFLATED) as zf:
    zf.write('predicted_household_consumption.csv')
    zf.write('predicted_poverty_distribution.csv')

print(f"\n{'='*80}")
print("COMPLETE!")
print(f"{'='*80}")
print(f"\nSubmission files:  predicted_household_consumption.csv")
print(f"                   predicted_poverty_distribution.csv")
print(f"                   submission.zip")
print(f"\nSaved model artifacts in ./{MODELS_DIR}/:")
print(f"  lgb_log_fold0..{n_folds-1}.pkl      — LightGBM log-target models")
print(f"  lgb_raw_fold0..{n_folds-1}.pkl      — LightGBM raw-target models")
print(f"  xgb_log_fold0..{n_folds-1}.json     — XGBoost log-target models")
print(f"  cat_log_fold0..{n_folds-1}.cbm      — CatBoost log-target models")
print(f"  label_encoders.pkl                  — Categorical encoders")
print(f"  feature_list.pkl                    — Feature column order")
print(f"  survey_poverty_targets.json         — Training survey rate targets")
print(f"  survey_consumption_stats.json       — Training survey statistics")
print(f"  survey_matches.json                 — Test-to-train survey mapping")
print(f"  calibration_params.json             — Per-survey scale/shift/gamma")
print(f"  config.json                         — n_folds, n_features")
print(f"\nTo run inference on new data WITHOUT retraining, use:")
print(f"  python predict.py --test new_test_features.csv --output my_submission.zip")
