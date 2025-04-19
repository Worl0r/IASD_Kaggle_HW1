# 1. Imports
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.metrics import mean_squared_error
from imblearn.over_sampling import SMOTE
from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import PowerTransformer
from utils import get_logger, load_config
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
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
from category_encoders import TargetEncoder



import matplotlib

matplotlib.use("WebAgg")

logger = get_logger()

cfg = load_config("configuration.yaml")


def feature_engineering(df_train, df_test):
    df_train = df_train.copy()
    df_test = df_test.copy()

    # cat_cols = ["var3", "var7", "var8", "var9", "var10"]
    # num_cols = ["var1", "var2", "var4", "var5", "var6"]
    cat_cols = ["var3", "var7", "var8", "var10"]
    num_cols = ["var2", "var4", "var5", "var6"]

    # Drop correlated columns
    df_train = df_train.drop(columns=["var1", "var9"])
    df_test = df_test.drop(columns=["var1", "var9"])

    # 1. MISSING VALUES TREATMENT
    # Create missing value flags and impute missing values
    # for col in ["var5", "var10"]:
    #     df_train[f'{col}_missing'] = df_train[col].isna().astype(int)
    #     imputer = SimpleImputer(strategy='median')
    #     df_train[col] = imputer.fit_transform(df_train[[col]])

    #     df_train[f'{col}_missing'] = df_train[col].isna().astype(int)
    #     imputer = SimpleImputer(strategy='median')
    #     df_test[col] = imputer.transform(df_test[[col]])
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


    # 2. OUTLIER TREATMENT
    # Cap outliers at percentiles instead of removing them - 1% most extreme values
    # for col in ["var1", "var4", "var8"]:
    for col in ["var4", "var8"]:
        # Get 1st and 99th percentiles
        p01 = df_train[col].quantile(0.0)
        p99 = df_train[col].quantile(0.99)
        
        # Cap values
        df_train[col] = df_train[col].clip(p01, p99)
        df_test[col] = df_test[col].clip(p01, p99)


    # 3. FEATURE TRANSFORMATIONS
    # Apply log transformation to highly skewed variables (like var4)
    # for col in ["var1", "var4", "var5"]:
    for col in ["var4", "var5"]:
        df_train[f'{col}_log'] = np.log(df_train[col] + 1)  # Adding 1 to avoid log(0)
        df_test[f'{col}_log'] = np.log(df_test[col] + 1)  # Adding 1 to avoid log(0)


    # Apply Box-Cox transformation to normalize distributions
    pt = PowerTransformer(method='box-cox')
    box_cox_cols = ['var2', 'var6']  # Columns identified with extreme values

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


    # 5. CATEGORICAL ENCODING
    # Target encoding for categorical variables
    y = df_train['TARGET']
    for col in cat_cols:
        # Using Target Encoder which often works well for binary classification
        encoder = TargetEncoder()
        # Fit and transform the training data
        df_train[f'{col}_target_enc'] = encoder.fit_transform(df_train[col], y)
        # Only transform the test data (without y)
        df_test[f'{col}_target_enc'] = encoder.transform(df_test[col])


    # 7. BINNING
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

    return df_train, df_test


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

    X, test = X.align(test, join="left", axis=1, fill_value=0)

    n_splits = 5
    # Use stratified k-fold to maintain class distribution
    from sklearn.model_selection import StratifiedKFold
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    base_models = [
        ("lgbm", LGBMClassifier(
            n_estimators=200,
            learning_rate=0.05,
            num_leaves=31,
            class_weight='balanced',
            random_state=42,
            scale_pos_weight=10  # Adjust based on class ratio
        )),
        ("catboost", CatBoostClassifier(verbose=0, random_state=42)),
        ("logreg", LogisticRegression(
            max_iter=2000,  # Increase 1000
            solver='saga',  # Try different solver
            class_weight="balanced",
            C=0.1  # Try regularization
        )),
        # ("ridge_classifier", RidgeClassifier(max_iter=100, class_weight='balanced')),
        (
            "mlp",
            MLPClassifier(
                hidden_layer_sizes=(128,), max_iter=600, random_state=42
            ),
        ),
        ("xgb", XGBClassifier(random_state=42, n_estimators=200)),
    ]

    S_train = np.zeros((X.shape[0], len(base_models)))
    S_test = np.zeros((test.shape[0], len(base_models)))

    for i, (name, model) in enumerate(base_models):
        logger.info(f"\n🔁 Training model: {name}")
        S_test_i = np.zeros((test.shape[0], n_splits))

        # And use it with
        for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
            logger.info(f" - Fold {fold + 1}")
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
            
            test_fold = test.copy()
            test_fold = test_fold.replace([np.inf, -np.inf], np.nan)
            test_fold = test_fold.fillna(test_fold.median())

            # Add feature scaling before model training
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            test_fold_scaled = scaler.transform(test_fold)

            model.fit(X_train_scaled, y_train)

            y_pred_val = model.predict(X_val_scaled)
            y_pred_test = model.predict(test_fold_scaled)
            y_pred_prob = model.predict_proba(X_val_scaled)[:, 1]

            S_train[val_idx, i] = y_pred_val
            S_test_i[:, fold] = y_pred_test

            rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
            auc = roc_auc_score(y_val, y_pred_prob)
            precision = precision_score(y_val, y_pred_val)
            recall = recall_score(y_val, y_pred_val)
            f1 = f1_score(y_val, y_pred_val)
            logloss = log_loss(y_val, y_pred_prob)

            logger.info(
                f" - Fold {fold + 1} RMSE: {rmse:.4f}, AUC: {auc:.4f},"
                f" Precision: {precision:.4f}, Recall: {recall:.4f},"
                f" F1: {f1:.4f}, Log Loss: {logloss:.4f}"
            )

        S_test[:, i] = S_test_i.mean(axis=1)

    # 6. Entraînement du modèle méta (stacker)
    logger.info("\n🎯 Training meta-model (Ridge)")
    meta_model = Ridge()
    meta_model.fit(S_train, y)

    # Validation croisée sur les prédictions des modèles de base
    y_pred = meta_model.predict(S_train)

    rmse = np.sqrt(mean_squared_error(y, y_pred))
    auc = roc_auc_score(y, (y_pred > 0.5).astype(int))
    precision = precision_score(y, (y_pred > 0.5).astype(int))
    recall = recall_score(y, (y_pred > 0.5).astype(int))
    f1 = f1_score(y, (y_pred > 0.5).astype(int))
    logloss = log_loss(y, y_pred)

    logger.info(
        f"Meta-model performance:  RMSE: {rmse:.4f}, AUC: {auc:.4f},"
        f" Precision: {precision:.4f}, Recall: {recall:.4f},"
        f" F1: {f1:.4f}, Log Loss: {logloss:.4f}"
    )

    # 7. Prédictions finales
    final_preds = meta_model.predict(S_test)

    submission = pd.DataFrame({"id": test_ids, "target": final_preds})
    submission.to_csv("output/submission_3.0.0.csv", index=False)


if __name__ == "__main__":
    main()
