"""
DrivenData Competition: Poverty Prediction Challenge
Author: Driss Taki
Date: 2026-02-06
"""

import os
import logging
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_error

# ==============================================================================
# LOGGING CONFIGURATION
# ==============================================================================
def setup_logging(log_file=None, level=logging.INFO):
    """
    Configure logging for the application.
    
    Args:
        log_file: Optional path to log file. If None, logs to console only.
        level: Logging level (default: INFO)
    """
    handlers = [logging.StreamHandler()]
    
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=handlers
    )
    
    return logging.getLogger(__name__)


# ==============================================================================
# 1️⃣ CONFIGURATION
# ==============================================================================
DATA_DIR = "sample_data"
OUTPUT_DIR = "submission"
LOG_FILE = os.path.join(OUTPUT_DIR, "training.log")
EXCLUDED_COLS = ["survey_id", "household_id"]

# Competition thresholds and weights
POVERTY_THRESHOLDS = np.array([
    3.17, 3.94, 4.60, 5.26, 5.88, 6.47, 7.06, 7.70, 8.40, 9.13,
    9.87, 10.70, 11.62, 12.69, 14.03, 15.64, 17.76, 20.99, 27.37
])
P_STAR = 40
POVERTY_WEIGHTS = 1 / (1 + np.abs(np.arange(5, 100, 5) - P_STAR))


# ==============================================================================
# 2️⃣ FEATURE ENGINEERING (NO LEAKAGE)
# ==============================================================================
def create_non_leakage_features(df, logger):
    """
    Creates features that don't induce leakage (not context-based).
    These features are computed independently for each household.
    
    Args:
        df: Input DataFrame
        logger: Logger instance
        
    Returns:
        DataFrame with additional features
    """
    logger.debug("Creating non-leakage features")
    df = df.copy()
    
    # Per capita ratios
    if "income" in df.columns and "household_size" in df.columns:
        df["income_per_capita"] = df["income"] / df["household_size"].clip(lower=1)
        logger.debug("Created income_per_capita feature")
        
    if "total_expenditure" in df.columns and "household_size" in df.columns:
        df["expenditure_per_capita"] = df["total_expenditure"] / df["household_size"].clip(lower=1)
        logger.debug("Created expenditure_per_capita feature")
    
    # Age features
    age_cols = [c for c in df.columns if "age" in c.lower() and df[c].dtype != object]
    if len(age_cols) > 0:
        df["mean_household_age"] = df[age_cols].mean(axis=1)
        df["max_household_age"] = df[age_cols].max(axis=1)
        df["var_household_age"] = df[age_cols].var(axis=1).fillna(0)
        df["age_min_max_ratio"] = df[age_cols].min(axis=1) / (df[age_cols].max(axis=1).clip(lower=1) + 1e-8)
        logger.debug(f"Created {4} age-related features from {len(age_cols)} age columns")
    
    # Education features
    edu_cols = [c for c in df.columns if ("education" in c.lower() or "school" in c.lower()) and df[c].dtype != object]
    if len(edu_cols) > 0:
        df["household_education_level"] = df[edu_cols].mean(axis=1)
        df["has_education_count"] = (df[edu_cols] > 0).sum(axis=1)
        logger.debug(f"Created {2} education features from {len(edu_cols)} education columns")
    
    # Food consumption
    food_cols = [c for c in df.columns if "food" in c.lower() and df[c].dtype != object]
    if len(food_cols) > 0:
        df["total_food_consumption"] = df[food_cols].sum(axis=1)
        df["food_diversity"] = (df[food_cols] > 0).sum(axis=1)
        if "total_expenditure" in df.columns:
            df["food_to_expenditure_ratio"] = df["total_food_consumption"] / (df["total_expenditure"].clip(lower=1) + 1e-8)
        logger.debug(f"Created food consumption features from {len(food_cols)} food columns")
    
    # Assets
    asset_cols = [c for c in df.columns if ("asset" in c.lower() or "owns" in c.lower()) and df[c].dtype != object]
    if len(asset_cols) > 0:
        df["total_assets"] = df[asset_cols].sum(axis=1)
        df["asset_count"] = (df[asset_cols] > 0).sum(axis=1)
        logger.debug(f"Created {2} asset features from {len(asset_cols)} asset columns")
    
    return df


def apply_survey_mean_ratios(df, survey_means_map, numeric_cols_for_ratio, logger):
    """
    Applies ratio features based on survey means computed from TRAIN only.
    This prevents leakage by using only training survey statistics.
    
    Args:
        df: Input DataFrame
        survey_means_map: Dictionary of survey-specific means
        numeric_cols_for_ratio: Columns to create ratios for
        logger: Logger instance
        
    Returns:
        DataFrame with survey ratio features
    """
    df = df.copy()
    features_created = 0
    
    for col in numeric_cols_for_ratio:
        if f"{col}_survey_mean_ratio" not in df.columns:
            if col in survey_means_map:
                survey_means = df["survey_id"].map(survey_means_map[col])
                global_mean = survey_means_map['GLOBAL_MEAN'][col]
                df[f"{col}_survey_mean_ratio"] = df[col] / (survey_means.fillna(global_mean) + 1e-8)
                features_created += 1
            else:
                df[f"{col}_survey_mean_ratio"] = 1.0
    
    logger.debug(f"Created {features_created} survey mean ratio features")
    return df


# ==============================================================================
# 3️⃣ WEIGHTED QUANTILE CALIBRATION
# ==============================================================================
class WeightedQuantileCalibrator:
    """
    Calibrates predictions to match the distribution of training targets.
    Uses weighted quantile mapping for better poverty rate estimation.
    """
    def __init__(self, logger=None):
        self.ref_cdf = None
        self.ref_values = None
        self.logger = logger or logging.getLogger(__name__)
    
    def fit(self, y_true, weights):
        """Fit calibrator on training data."""
        sorted_idx = np.argsort(y_true)
        y_sorted = y_true[sorted_idx]
        w_sorted = weights[sorted_idx]
        cdf = np.cumsum(w_sorted) / np.sum(w_sorted)
        self.ref_cdf = cdf
        self.ref_values = y_sorted
        self.logger.debug(f"Calibrator fitted on {len(y_true)} samples")
        return self
    
    def calibrate(self, y_pred, weights):
        """Calibrate predictions to match training distribution."""
        sorted_idx = np.argsort(y_pred)
        y_sorted = y_pred[sorted_idx]
        w_sorted = weights[sorted_idx]
        cdf_pred = np.cumsum(w_sorted) / np.sum(w_sorted)
        y_cal = np.interp(cdf_pred, self.ref_cdf, self.ref_values)
        inverse_idx = np.argsort(sorted_idx)
        return y_cal[inverse_idx]


# ==============================================================================
# 4️⃣ EVALUATION METRICS
# ==============================================================================
def wmape(y_true, y_pred, weights=None, eps=1e-8):
    """Weighted Mean Absolute Percentage Error."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if weights is None:
        weights = np.ones_like(y_true)
    return np.sum(weights * np.abs(y_pred - y_true) / (y_true + eps)) / np.sum(weights)


def calc_poverty_rates(y, thresholds, weights):
    """Calculate poverty rates at various thresholds."""
    rates = []
    total_w = np.sum(weights)
    for t in thresholds:
        mask = y < t
        rates.append(np.sum(weights[mask]) / total_w)
    return np.array(rates)


def ws_wmape(y_true, y_pred, weights, thresholds, threshold_weights):
    """Competition metric: weighted combination of poverty rates and consumption."""
    pr_true = calc_poverty_rates(y_true, thresholds, weights)
    pr_pred = calc_poverty_rates(y_pred, thresholds, weights)
    poverty_score = wmape(pr_true, pr_pred, threshold_weights)
    consumption_score = wmape(y_true, y_pred, weights)
    return 0.9 * poverty_score + 0.1 * consumption_score


# ==============================================================================
# 5️⃣ UTILITY FUNCTIONS
# ==============================================================================
def safe_select(df, columns):
    """Safely select columns, filling missing ones with 0."""
    return df.reindex(columns=columns, fill_value=0)


# ==============================================================================
# 6️⃣ MAIN TRAINING PIPELINE
# ==============================================================================
def main():
    """Main training and prediction pipeline."""
    
    # Initialize logging
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logger = setup_logging(log_file=LOG_FILE, level=logging.INFO)
    
    logger.info("="*80)
    logger.info("DrivenData Poverty Prediction - Professional Solution")
    logger.info("Author: Driss Taki")
    logger.info("="*80)
    
    try:
        # ----------------------------------------------------------------------
        # Step 1: Load Data
        # ----------------------------------------------------------------------
        logger.info("Step 1/8: Loading data...")
        
        df_x_train = pd.read_csv(os.path.join(DATA_DIR, "train_hh_features.csv"))
        df_y_train = pd.read_csv(os.path.join(DATA_DIR, "train_hh_gt.csv"))
        df_x_test = pd.read_csv(os.path.join(DATA_DIR, "test_hh_features.csv"))
        
        # Ensure ID alignment
        if "hhid" in df_x_train.columns:
            df_x_train.rename(columns={"hhid": "household_id"}, inplace=True)
        if "hhid" in df_x_test.columns:
            df_x_test.rename(columns={"hhid": "household_id"}, inplace=True)
        
        if "household_id" not in df_x_train.columns:
            df_x_train["household_id"] = df_x_train.index.values
        if "household_id" not in df_x_test.columns:
            df_x_test["household_id"] = df_x_test.index.values
        
        logger.info(f"  Train shape: {df_x_train.shape}")
        logger.info(f"  Test shape: {df_x_test.shape}")
        
        # ----------------------------------------------------------------------
        # Step 2: Feature Engineering
        # ----------------------------------------------------------------------
        logger.info("Step 2/8: Creating non-leakage features...")
        
        df_x_train = create_non_leakage_features(df_x_train, logger)
        df_x_test_base = create_non_leakage_features(df_x_test, logger)
        
        # Define numeric columns for ratio features
        all_cols = [c for c in df_x_train.columns if c not in EXCLUDED_COLS]
        NUMERIC_COLS_FOR_RATIO = df_x_train[all_cols].select_dtypes(include=[np.number]).columns
        logger.info(f"  Identified {len(NUMERIC_COLS_FOR_RATIO)} numeric columns for ratio features")
        
        # ----------------------------------------------------------------------
        # Step 3: Define Universal Feature Space
        # ----------------------------------------------------------------------
        logger.info("Step 3/8: Defining universal feature space...")
        
        temp_df_train = df_x_train.copy()
        temp_survey_means_map = {}
        cols_to_ratio = temp_df_train.columns.intersection(NUMERIC_COLS_FOR_RATIO)
        
        for col in cols_to_ratio:
            temp_survey_means_map[col] = temp_df_train.groupby("survey_id")[col].mean().to_dict()
        temp_survey_means_map['GLOBAL_MEAN'] = temp_df_train.loc[:, cols_to_ratio].mean().to_dict()
        
        ALL_FE_COLS_DF = apply_survey_mean_ratios(
            temp_df_train, temp_survey_means_map, NUMERIC_COLS_FOR_RATIO, logger
        ).drop(columns=NUMERIC_COLS_FOR_RATIO, errors='ignore').drop(columns=EXCLUDED_COLS, errors='ignore')
        
        ALL_FE_COLS = ALL_FE_COLS_DF.columns.tolist()
        logger.info(f"  Total feature columns: {len(ALL_FE_COLS)}")
        
        # ----------------------------------------------------------------------
        # Step 4: Cross-Validation (Leave-One-Survey-Out)
        # ----------------------------------------------------------------------
        logger.info("Step 4/8: Running Leave-One-Survey-Out cross-validation...")
        
        SURVEYS = df_x_train["survey_id"].unique()
        logger.info(f"  Training on {len(SURVEYS)} surveys")
        
        oof_pred = np.zeros(len(df_x_train))
        feature_importances = pd.DataFrame()
        cv_mse_list = []
        
        for survey in SURVEYS:
            logger.info(f"  Processing fold: Survey {survey}")
            
            mask_valid = df_x_train["survey_id"] == survey
            mask_train = ~mask_valid
            
            X_train_fold = df_x_train.loc[mask_train].copy()
            X_valid_fold = df_x_train.loc[mask_valid].copy()
            
            # Calculate survey means ONLY on training fold (anti-leakage)
            survey_means_map = {}
            cols_to_ratio = X_train_fold.columns.intersection(NUMERIC_COLS_FOR_RATIO)
            
            for col in cols_to_ratio:
                survey_means_map[col] = X_train_fold.groupby("survey_id")[col].mean().to_dict()
            survey_means_map['GLOBAL_MEAN'] = X_train_fold.loc[:, cols_to_ratio].mean().to_dict()
            
            # Apply ratio features
            X_train_temp = apply_survey_mean_ratios(
                X_train_fold, survey_means_map, NUMERIC_COLS_FOR_RATIO, logger
            ).drop(columns=NUMERIC_COLS_FOR_RATIO, errors='ignore').drop(columns=EXCLUDED_COLS, errors='ignore')
            
            X_valid_temp = apply_survey_mean_ratios(
                X_valid_fold, survey_means_map, NUMERIC_COLS_FOR_RATIO, logger
            ).drop(columns=NUMERIC_COLS_FOR_RATIO, errors='ignore').drop(columns=EXCLUDED_COLS, errors='ignore')
            
            # Force column alignment
            X_train_final = X_train_temp.reindex(columns=ALL_FE_COLS, fill_value=0)
            X_valid_final = X_valid_temp.reindex(columns=ALL_FE_COLS, fill_value=0)
            
            y_train = df_y_train.loc[mask_train, "cons_ppp17"]
            y_valid = df_y_train.loc[mask_valid, "cons_ppp17"]
            
            # Handle categorical columns
            cat_cols_all = [c for c in ALL_FE_COLS if X_train_final[c].dtype == 'object' or X_train_final[c].dtype.name == 'category']
            cat_cols = [c for c in cat_cols_all if c in X_valid_final.columns and c in X_train_final.columns]
            
            for c in cat_cols:
                X_train_final[c] = X_train_final[c].astype("category")
                X_valid_final[c] = X_valid_final[c].astype("category")
            
            y_train_log = np.log1p(y_train)
            
            # Train model
            model = lgb.LGBMRegressor(
                objective="regression_l2",
                n_estimators=3000,
                learning_rate=0.01,
                num_leaves=128,
                subsample=0.7,
                colsample_bytree=0.7,
                random_state=42 + int(survey),
                n_jobs=-1,
                verbose=-1
            )
            
            model.fit(
                X_train_final, y_train_log,
                categorical_feature=cat_cols,
                eval_set=[(X_valid_final, np.log1p(y_valid))],
                callbacks=[lgb.early_stopping(200, verbose=False)]
            )
            
            y_pred = np.expm1(model.predict(X_valid_final))
            
            # Calibration
            calibrator = WeightedQuantileCalibrator(logger=logger)
            calibrator.fit(y_train.values, np.ones_like(y_train))
            y_wq = calibrator.calibrate(y_pred, np.ones_like(y_valid))
            
            cv_mse = mean_squared_error(np.log1p(y_valid), model.predict(X_valid_final))
            cv_mse_list.append(cv_mse)
            score_ws_wmape = ws_wmape(y_valid, y_wq, np.ones_like(y_valid), POVERTY_THRESHOLDS, POVERTY_WEIGHTS)
            
            logger.info(f"    MSE (log): {cv_mse:.6f} | ws-wMAPE: {score_ws_wmape*100:.4f}")
            
            oof_pred[mask_valid] = y_wq
            
            # Store feature importances
            fold_importance = pd.DataFrame({
                'feature': ALL_FE_COLS,
                'importance': model.feature_importances_
            })
            feature_importances = pd.concat([feature_importances, fold_importance], ignore_index=True)
        
        # Global OOF score
        global_score = ws_wmape(
            df_y_train["cons_ppp17"], oof_pred, np.ones_like(oof_pred),
            POVERTY_THRESHOLDS, POVERTY_WEIGHTS
        )
        avg_cv_mse = np.mean(cv_mse_list)
        
        logger.info(f"  Overall OOF ws-wMAPE: {global_score*100:.4f}")
        logger.info(f"  Average CV MSE (log): {avg_cv_mse:.6f}")
        
        # ----------------------------------------------------------------------
        # Step 5: Feature Selection
        # ----------------------------------------------------------------------
        logger.info("Step 5/8: Selecting features based on importance...")
        
        importance_mean = feature_importances.groupby('feature')['importance'].mean().reset_index()
        importance_mean.sort_values(by='importance', ascending=False, inplace=True)
        
        N_TOTAL_FEATURES = len(importance_mean)
        N_FEATURES_TO_KEEP = int(N_TOTAL_FEATURES * 0.75)
        selected_features_temp = importance_mean.head(N_FEATURES_TO_KEEP)['feature'].tolist()
        
        # Avoid raw columns that were removed
        COLS_TO_AVOID = set(NUMERIC_COLS_FOR_RATIO) | set(EXCLUDED_COLS)
        selected_features = [f for f in selected_features_temp if f not in COLS_TO_AVOID]
        
        logger.info(f"  Selected {len(selected_features)}/{N_TOTAL_FEATURES} features (top 75%)")
        
        # ----------------------------------------------------------------------
        # Step 6: Validation with Selected Features
        # ----------------------------------------------------------------------
        logger.info("Step 6/8: Validating with selected features...")
        
        oof_pred_selected = np.zeros(len(df_x_train))
        
        for survey in SURVEYS:
            mask_valid = df_x_train["survey_id"] == survey
            mask_train = ~mask_valid
            
            X_train_fold = df_x_train.loc[mask_train].copy()
            X_valid_fold = df_x_train.loc[mask_valid].copy()
            
            # Calculate survey means
            survey_means_map = {}
            cols_to_ratio = X_train_fold.columns.intersection(NUMERIC_COLS_FOR_RATIO)
            for col in cols_to_ratio:
                survey_means_map[col] = X_train_fold.groupby("survey_id")[col].mean().to_dict()
            survey_means_map['GLOBAL_MEAN'] = X_train_fold.loc[:, cols_to_ratio].mean().to_dict()
            
            # Apply ratios
            X_train_temp = apply_survey_mean_ratios(
                X_train_fold, survey_means_map, NUMERIC_COLS_FOR_RATIO, logger
            ).drop(columns=NUMERIC_COLS_FOR_RATIO, errors='ignore').drop(columns=EXCLUDED_COLS, errors='ignore')
            
            X_valid_temp = apply_survey_mean_ratios(
                X_valid_fold, survey_means_map, NUMERIC_COLS_FOR_RATIO, logger
            ).drop(columns=NUMERIC_COLS_FOR_RATIO, errors='ignore').drop(columns=EXCLUDED_COLS, errors='ignore')
            
            # Alignment and selection
            X_train_final = safe_select(X_train_temp, selected_features)
            X_valid_final = safe_select(X_valid_temp, selected_features)
            
            y_train = df_y_train.loc[mask_train, "cons_ppp17"]
            y_valid = df_y_train.loc[mask_valid, "cons_ppp17"]
            
            # Categorical columns
            cat_cols = [c for c in selected_features if X_train_final[c].dtype == 'object' or X_train_final[c].dtype.name == 'category']
            for c in cat_cols:
                X_train_final[c] = X_train_final[c].astype("category")
                X_valid_final[c] = X_valid_final[c].astype("category")
            
            y_train_log = np.log1p(y_train)
            
            # Train
            model = lgb.LGBMRegressor(
                objective="regression_l2",
                n_estimators=3000,
                learning_rate=0.01,
                num_leaves=128,
                subsample=0.7,
                colsample_bytree=0.7,
                random_state=42 + int(survey),
                n_jobs=-1,
                verbose=-1
            )
            
            model.fit(
                X_train_final, y_train_log,
                categorical_feature=cat_cols,
                eval_set=[(X_valid_final, np.log1p(y_valid))],
                callbacks=[lgb.early_stopping(200, verbose=False)]
            )
            
            y_pred = np.expm1(model.predict(X_valid_final))
            calibrator = WeightedQuantileCalibrator(logger=logger)
            calibrator.fit(y_train.values, np.ones_like(y_train))
            y_wq = calibrator.calibrate(y_pred, np.ones_like(y_valid))
            
            oof_pred_selected[mask_valid] = y_wq
        
        global_score_selected = ws_wmape(
            df_y_train["cons_ppp17"], oof_pred_selected, np.ones_like(oof_pred_selected),
            POVERTY_THRESHOLDS, POVERTY_WEIGHTS
        )
        
        logger.info(f"  ws-wMAPE BEFORE selection: {global_score*100:.4f}")
        logger.info(f"  ws-wMAPE AFTER selection:  {global_score_selected*100:.4f}")
        
        # ----------------------------------------------------------------------
        # Step 7: Train Final Models
        # ----------------------------------------------------------------------
        logger.info("Step 7/8: Training final ensemble models...")
        
        submission_models = []
        
        for i, survey_model in enumerate(SURVEYS):
            logger.info(f"  Training model {i+1}/{len(SURVEYS)} (excluding survey {survey_model})")
            
            mask_train = df_x_train["survey_id"] != survey_model
            
            X_train_fold = df_x_train.loc[mask_train].copy()
            y_train_full = df_y_train.loc[mask_train, "cons_ppp17"]
            
            # Calculate survey means
            survey_means_map = {}
            cols_to_ratio = X_train_fold.columns.intersection(NUMERIC_COLS_FOR_RATIO)
            for col in cols_to_ratio:
                survey_means_map[col] = X_train_fold.groupby("survey_id")[col].mean().to_dict()
            survey_means_map['GLOBAL_MEAN'] = X_train_fold.loc[:, cols_to_ratio].mean().to_dict()
            
            # Apply ratios
            X_train_temp = apply_survey_mean_ratios(
                X_train_fold, survey_means_map, NUMERIC_COLS_FOR_RATIO, logger
            ).drop(columns=NUMERIC_COLS_FOR_RATIO, errors='ignore').drop(columns=EXCLUDED_COLS, errors='ignore')
            
            # Secure alignment
            X_train_final = safe_select(X_train_temp, selected_features)
            
            # Categorical columns
            cat_cols_train = [c for c in selected_features if X_train_final[c].dtype == 'object' or X_train_final[c].dtype.name == 'category']
            for c in cat_cols_train:
                X_train_final[c] = X_train_final[c].astype("category")
            
            # Train
            y_train_log_full = np.log1p(y_train_full)
            model = lgb.LGBMRegressor(
                objective="regression_l2",
                n_estimators=3000,
                learning_rate=0.01,
                num_leaves=128,
                subsample=0.7,
                colsample_bytree=0.7,
                random_state=100 + i,
                n_jobs=-1,
                verbose=-1
            )
            model.fit(X_train_final, y_train_log_full, categorical_feature=cat_cols_train)
            
            # Calibration
            calibrator = WeightedQuantileCalibrator(logger=logger)
            calibrator.fit(y_train_full.values, np.ones_like(y_train_log_full))
            
            # Store model
            submission_models.append((model, calibrator, survey_means_map))
        
        logger.info(f"  Successfully trained {len(submission_models)} models")
        
        # ----------------------------------------------------------------------
        # Step 8: Generate Submission
        # ----------------------------------------------------------------------
        logger.info("Step 8/8: Generating submission files...")
        
        # Prepare test base
        ID_COLS = ["survey_id", "household_id"]
        cols_for_test_base = [c for c in df_x_test_base.columns if c not in NUMERIC_COLS_FOR_RATIO and c not in EXCLUDED_COLS]
        X_test_base = df_x_test_base.loc[:, ID_COLS + cols_for_test_base].copy()
        
        predictions_list = []
        poverty_list = []
        
        test_surveys = sorted(X_test_base["survey_id"].unique())
        logger.info(f"  Processing {len(test_surveys)} test surveys")
        
        for survey in test_surveys:
            mask = X_test_base["survey_id"] == survey
            
            X_test_survey_base = X_test_base.loc[mask].copy()
            X_test_survey_full = df_x_test_base.loc[mask].copy()
            
            preds = []
            
            for model, calibrator, survey_means_map in submission_models:
                # Create ratio features
                X_test_temp = apply_survey_mean_ratios(
                    X_test_survey_full, survey_means_map, NUMERIC_COLS_FOR_RATIO, logger
                ).drop(columns=NUMERIC_COLS_FOR_RATIO, errors='ignore').drop(columns=EXCLUDED_COLS, errors='ignore')
                
                # Secure selection
                X_test_final = safe_select(X_test_temp, selected_features)
                
                # Convert categorical columns
                cat_cols_test = [c for c in selected_features if X_test_final[c].dtype == 'object' or X_test_final[c].dtype.name == 'category']
                for c in cat_cols_test:
                    if c in X_test_final.columns:
                        X_test_final[c] = X_test_final[c].astype("category")
                
                # Predict
                y_pred = np.expm1(model.predict(X_test_final))
                y_pred_cal = calibrator.calibrate(y_pred, np.ones_like(y_pred))
                preds.append(y_pred_cal)
            
            y_final = np.mean(preds, axis=0)
            
            # Store household predictions
            predictions_list.append(pd.DataFrame({
                "survey_id": X_test_survey_base["survey_id"].values,
                "household_id": X_test_survey_base["household_id"].values,
                "cons_ppp17": y_final
            }))
            
            # Poverty rates for this survey
            poverty_rates = calc_poverty_rates(y_final, POVERTY_THRESHOLDS, np.ones_like(y_final))
            poverty_list.append([survey] + poverty_rates.tolist())
            
            logger.debug(f"  Processed survey {survey}: {len(y_final)} predictions")
        
        # Create submission files
        df_submission_consumption = pd.concat(predictions_list, ignore_index=True)
        consumption_file = os.path.join(OUTPUT_DIR, "predicted_household_consumption.csv")
        df_submission_consumption.to_csv(consumption_file, index=False)
        logger.info(f"  Created: {consumption_file}")
        
        poverty_columns = ["survey_id"] + [f"pct_hh_below_{t:.2f}" for t in POVERTY_THRESHOLDS]
        df_submission_poverty = pd.DataFrame(poverty_list, columns=poverty_columns)
        poverty_file = os.path.join(OUTPUT_DIR, "predicted_poverty_distribution.csv")
        df_submission_poverty.to_csv(poverty_file, index=False)
        logger.info(f"  Created: {poverty_file}")
        
        logger.info("="*80)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        logger.info(f"Submission files saved to: {OUTPUT_DIR}/")
        logger.info(f"Training log saved to: {LOG_FILE}")
        logger.info("="*80)
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {str(e)}", exc_info=True)
        raise


# ==============================================================================
# 7️⃣ ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    main()
