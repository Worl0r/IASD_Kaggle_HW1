# 1. Imports
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from xgboost import XGBClassifier
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import mean_squared_error
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PowerTransformer, StandardScaler
from utils import get_logger, load_config
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import (
    mean_squared_error,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    log_loss,
)
from scipy.stats import randint, uniform
from sklearn.calibration import CalibratedClassifierCV



import matplotlib

matplotlib.use("WebAgg")

logger = get_logger()

cfg = load_config("configuration.yaml")


def feature_engineering(df_train, df_test):
    df_train = df_train.copy()
    df_test = df_test.copy()

    cat_cols = ["var3", "var8", "var10"]
    num_cols = ["var1", "var2", "var4", "var5", "var6"]

    # Drop correlated columns
    # df_train = df_train.drop(columns=["var7", "var9"])
    # df_test = df_test.drop(columns=["var7", "var9"])

    # 1. MISSING VALUES TREATMENT
    for col in ["var5", "var10"]:
        # Create missing flags
        df_train[f'{col}_missing'] = df_train[col].isna().astype(int)
        df_test[f'{col}_missing'] = df_test[col].isna().astype(int)
        
        # Create and fit imputer on training data
        imputer = SimpleImputer(strategy='median')
        imputer.fit(df_train[[col]])
        
        # Transform both training and test data using the same fitted imputer
        df_train[col] = imputer.transform(df_train[[col]])
        df_test[col] = imputer.transform(df_test[[col]])

    # convert var10 to int64
    df_train["var10"] = df_train["var10"].astype('int64')
    df_test["var10"] = df_test["var10"].astype('int64')

    # 2. OUTLIER TREATMENT
    # Cap outliers at percentiles instead of removing them - 1% most extreme values
    for col in ["var1", "var4", "var5"]:
        # Get 1st and 99th percentiles
        p01 = df_train[col].quantile(0.0)
        p99 = df_train[col].quantile(0.99)
        
        # Cap values
        df_train[col] = df_train[col].clip(p01, p99)
        df_test[col] = df_test[col].clip(p01, p99)


    # 3. FEATURE TRANSFORMATIONS
    # Apply log transformation to highly skewed variables
    # for col in ["var1", "var4", "var5"]:
    for col in ["var1", "var4"]:
        df_train[f'{col}_log'] = np.log(df_train[col] + 1)  # Adding 1 to avoid log(0)
        df_test[f'{col}_log'] = np.log(df_test[col] + 1)  # Adding 1 to avoid log(0)


    # Apply Box-Cox transformation to normalize distributions
    pt = PowerTransformer(method='box-cox')
    # box_cox_cols = ['var2', 'var6']  # Columns identified with extreme values
    box_cox_cols = ['var2', 'var6', "var5"]  # Columns identified with extreme values

    # Create a temporary array for fitting the transformer (Box-Cox needs positive values)
    temp_train_data = df_train[box_cox_cols].copy()
    temp_test_data = df_test[box_cox_cols].copy()

    # Add 1 to ensure positivity for Box-Cox
    for col in box_cox_cols:
        temp_train_data[col] = temp_train_data[col] + 1
        temp_test_data[col] = temp_test_data[col] + 1

    # Fit on training data, transform both training and test data
    transformed_train_data = pt.fit_transform(temp_train_data)
    transformed_test_data = pt.transform(temp_test_data)  # Only transform, not fit_transform

    # Add the transformed columns to the dataframes
    for i, col in enumerate(box_cox_cols):
        df_train[f'{col}_boxcox'] = transformed_train_data[:, i]
        df_test[f'{col}_boxcox'] = transformed_test_data[:, i]


    # 4. FEATURE SCALING - Added as requested
    # Apply standard scaling to all numerical features
    scaler = StandardScaler()
    
    # Identify numerical columns to scale (original and derived)
    # scale_cols = num_cols + [f'{col}_log' for col in ["var1", "var4", "var5"]] + [f'{col}_boxcox' for col in box_cox_cols]
    scale_cols = [f'{col}_log' for col in ["var1", "var4"]] + [f'{col}_boxcox' for col in box_cox_cols]
    
    # Fit scaler on training data
    scaler.fit(df_train[scale_cols])
    
    # Transform both training and test data
    scaled_train_data = scaler.transform(df_train[scale_cols])
    scaled_test_data = scaler.transform(df_test[scale_cols])
    
    # Add scaled features to dataframes
    for i, col in enumerate(scale_cols):
        df_train[f'{col}_scaled'] = scaled_train_data[:, i]
        df_test[f'{col}_scaled'] = scaled_test_data[:, i]


    # 6. BINNING
    # Create binned versions of numerical variables
    for col in num_cols:
        # Create 5 bins based on quantiles
        df_train[f'{col}_bin'] = pd.qcut(df_train[col], q=5, labels=False, duplicates='drop')
        
        # Get the bin edges from the training set
        bin_edges = pd.qcut(df_train[col], q=5, retbins=True, duplicates='drop')[1]
        
        # Apply same binning to test set
        df_test[f'{col}_bin'] = pd.cut(df_test[col], bins=bin_edges, labels=False, include_lowest=True)
        
        # Handle potential NaNs from binning
        df_train[f'{col}_bin'] = df_train[f'{col}_bin'].fillna(-1).astype(int)
        df_test[f'{col}_bin'] = df_test[f'{col}_bin'].fillna(-1).astype(int)

    for col in ["var3", "var7", "var8", "var9", "var10"]:
        df_train[f'{col}_threshold'] = df_train[col].apply(lambda x: x if x <= 3  else 4)
        df_test[f'{col}_threshold'] = df_test[col].apply(lambda x: x if x <= 3 else 4)

    df_train = df_train.drop(columns=num_cols)
    df_test = df_test.drop(columns=num_cols)

    df_train = df_train.drop(columns=["var3", "var7", "var8", "var9", "var10"])
    df_test = df_test.drop(columns=["var3", "var7", "var8", "var9", "var10"])

    return df_train, df_test


def tune_hyperparameters(X, y, model_name, model):
    """
    Perform hyperparameter tuning using RandomizedSearchCV
    """
    logger.info(f"Tuning hyperparameters for {model_name}...")
    
    param_grid = None
    

    if model_name == "catboost":
        param_grid = {
            'learning_rate': uniform(0.01, 0.1),
            'depth': randint(6, 10),
            'l2_leaf_reg': uniform(5, 10),
            'iterations': randint(150, 400),
            # 'scale_pos_weight': [5, 10, 15, 20, 25, 30, 35]
            # 'bagging_temperature': [0.5, 1, 2],
        }
    if model_name == "logreg":
        param_grid = {
            'C': uniform(4, 10),
            'penalty': ['l2'],
            'solver': ['newton-cholesky'],
            # 'l1_ratio': uniform(0, 1) if 'elasticnet' else None,
            'max_iter': [3500, 4000],
        }
    elif model_name == "mlp":
        param_grid = {
            'hidden_layer_sizes': [(128,)],
            'activation': ['relu'],
            'alpha': uniform(0.005, 0.02),
            'learning_rate_init': uniform(0.001, 0.01),
            'max_iter': [800, 1200, 1600]
        }
    elif model_name == "xgb":
        param_grid = {
            'learning_rate': uniform(0.01, 0.05),
            'n_estimators': randint(150, 250),
            'max_depth': randint(4, 10),
            'subsample': uniform(0.6, 0.4),
            'colsample_bytree': uniform(0.6, 0.4),
            'gamma': uniform(0, 0.5),
            'min_child_weight': randint(4, 10),
            'scale_pos_weight': [5, 10, 15, 20, 25, 30, 35]
        }
    
    if param_grid is None:
        logger.info(f"No parameter grid defined for {model_name}. Skipping tuning.")
        return model
    
    # Define the randomized search
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=40,  # Number of parameter settings sampled
        cv=3,  # 3-fold cross-validation
        verbose=1,
        random_state=42,
        n_jobs=-1,  # Use all available cores
        scoring='neg_mean_squared_error',  # Use AUC as the scoring metric
        error_score='raise',  # Raise errors instead of setting scores to NaN
    )
    
    # Fit the randomized search
    random_search.fit(X, y)
    
    # Log the best parameters and score
    logger.info(f"Best parameters for {model_name}: {random_search.best_params_}")
    logger.info(f"Best score for {model_name}: {random_search.best_score_:.4f}")
    
    return random_search.best_estimator_


def main():
    # Raw data
    train = pd.read_csv(cfg['path']['train_path'])
    test = pd.read_csv(cfg['path']['test_path'])

    train, test = feature_engineering(train, test)

    y = train["TARGET"]
    X = train.drop(columns=["TARGET"])

    test_ids = test["ID"]
    X = X.drop(columns=["ID"])
    test = test.drop(columns=["ID"])

    # X, test = X.align(test, join="left", axis=1, fill_value=0)

    n_splits = 5
    # Use stratified k-fold to maintain class distribution
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Define base models
    base_models = [
        # ("lgbm", LGBMClassifier(
        #     random_state=42, 
        #     class_weight='balanced',
        #     colsample_bytree=0.913,
        #     learning_rate=0.089,
        #     min_child_samples=9,
        #     n_estimators=283,
        #     num_leaves=21,
        #     reg_alpha=1.802,
        #     reg_lambda=1.872,
        #     scale_pos_weight=1,
        #     sumsample=0.987,
        #     )),
        # ("catboost", CatBoostClassifier(
        #     verbose=0, 
        #     random_state=42,
        #     scale_pos_weight=14,  # 14 au lieu de 5
        #     iterations=300,
        #     depth=9,
        #     l2_leaf_reg=5.21,
        #     learning_rate=0.107,
        #     )),
        ("catboost", CatBoostClassifier(
            verbose=0, 
            random_state=42,
            # scale_pos_weight=5,
            # iterations=170,
            # depth=6,
            # l2_leaf_reg=6.56,
            # learning_rate=0.0256,
            )),
        ("logreg", LogisticRegression(
            class_weight="balanced", 
            random_state=42, 
            # max_iter = 3500,
            # solver="newton-cholesky",
            # penalty="l2",
            # C=7.454,    # pas sur.
            )),
        ("mlp", MLPClassifier(
            random_state=42,
            # alpha=0.012903,
            # max_iter=1600,
            # activation="relu",
            # hidden_layer_sizes=(128,),
            # learning_rate_init=0.01027,
            )),
        ("xgb", XGBClassifier(
            random_state=42,
            # max_depth=7,
            # n_estimators=200,
            # min_child_weight=5,
            # scale_pos_weight=5,
            # learning_rate=0.04479,
            # subsample=0.6967,
            # colsample_bytree=0.6836,
            # gamma=0.2707,
            )),
    ]

    # Get a smaller subset for hyperparameter tuning
    from sklearn.model_selection import train_test_split
    X_tune, _, y_tune, _ = train_test_split(X, y, test_size=0.7, random_state=42, stratify=y)

    # Tune hyperparameters and update the models
    tuned_models = []
    for name, model in base_models:
        tuned_model = tune_hyperparameters(X_tune, y_tune, name, model)
        tuned_models.append((name, tuned_model))

    S_train = np.zeros((X.shape[0], len(tuned_models)))
    S_test = np.zeros((test.shape[0], len(tuned_models)))

    for i, (name, model) in enumerate(tuned_models):
        logger.info(f"\n🔁 Training model: {name}")
        S_test_i = np.zeros((test.shape[0], n_splits))

        fold_rmse_scores = []
        for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
            logger.info(f" - Fold {fold + 1}")
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
            
            # No need for additional scaling since we already did it in feature engineering
            calibrated_model = CalibratedClassifierCV(
                estimator=model,
                method='isotonic',  # or 'sigmoid'
                cv=3
            )
            calibrated_model.fit(X_train, y_train)

            # For predictions
            if hasattr(calibrated_model, 'predict_proba'):
                y_pred_prob = calibrated_model.predict_proba(X_val)[:, 1]
                y_pred_val = y_pred_prob  # Use probabilities directly for RMSE
                test_pred_prob = calibrated_model.predict_proba(test)[:, 1]
                S_test_i[:, fold] = test_pred_prob
            else:
                y_pred_val = calibrated_model.predict(X_val)
                S_test_i[:, fold] = calibrated_model.predict(test)
                y_pred_prob = y_pred_val  # Fallback if no predict_proba

            S_train[val_idx, i] = y_pred_prob

            # Calculate metrics
            rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
            fold_rmse_scores.append(rmse)

            # Additional metrics for information
            y_pred = (y_pred_prob > 0.5).astype(int)  # Convert probabilities to class predictions
            auc = roc_auc_score(y_val, y_pred_prob)
            precision = precision_score(y_val, y_pred)
            recall = recall_score(y_val, y_pred)
            f1 = f1_score(y_val, y_pred)
            logloss = log_loss(y_val, y_pred_prob)

            logger.info(
                f" - Fold {fold + 1} RMSE: {rmse:.4f}, AUC: {auc:.4f},"
                f" Precision: {precision:.4f}, Recall: {recall:.4f},"
                f" F1: {f1:.4f}, Log Loss: {logloss:.4f}"
            )

        S_test[:, i] = S_test_i.mean(axis=1)
        logger.info(f"Average RMSE for {name}: {np.mean(fold_rmse_scores):.4f}")

    # Train the meta-model (stacker)
    logger.info("\n🎯 Training meta-model (Ridge) optimized for RMSE")

    # Tune the meta-model
    meta_model_params = {
        'alpha': uniform(0.001, 10),  # Wider range to find optimal regularization
        'fit_intercept': [True, False]
    }

    meta_model = Ridge()
    meta_model_search = RandomizedSearchCV(
        meta_model,
        param_distributions=meta_model_params,
        n_iter=30,  # Increased iterations for better search
        cv=5,  # Increased from 3 to 5 for more robust validation
        verbose=1,
        random_state=42,
        scoring='neg_mean_squared_error'
    )

    meta_model_search.fit(S_train, y)
    logger.info(f"Best meta-model parameters: {meta_model_search.best_params_}")
    logger.info(f"Best meta-model RMSE: {np.sqrt(-meta_model_search.best_score_):.4f}")
    meta_model = meta_model_search.best_estimator_

    # Final prediction using the tuned meta-model
    y_pred = meta_model.predict(S_train)

    # Calculate metrics for the meta-model with emphasis on RMSE
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    logger.info(f"Meta-model final RMSE: {rmse:.4f}")

    # Additional metrics for information
    binary_preds = (y_pred > 0.5).astype(int)
    auc = roc_auc_score(y, y_pred)
    precision = precision_score(y, binary_preds)
    recall = recall_score(y, binary_preds)
    f1 = f1_score(y, binary_preds)

    logger.info(
        f"Additional metrics - AUC: {auc:.4f},"
        f" Precision: {precision:.4f}, Recall: {recall:.4f},"
        f" F1: {f1:.4f}"
    )

    # Generate final predictions
    final_preds = meta_model.predict(S_test)

    # Create submission file
    submission = pd.DataFrame({"ID": test_ids, "TARGET": final_preds})
    submission.to_csv("output/submission_training_12_8.csv", index=False)


if __name__ == "__main__":
    main()