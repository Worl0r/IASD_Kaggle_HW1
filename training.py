# 1. Imports
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from imblearn.over_sampling import SMOTE
from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier

import matplotlib

matplotlib.use("WebAgg")

logger = get_logger()

cfg = load_config("configuration.yaml")


def feature_engineering(df):
    df = df.copy()

    # 1. Missing flags
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            df[f"{col}_missing"] = df[col].isnull().astype(int)

    # 2. Imputation
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())

    # 3. Log transform
    from scipy.stats import skew

    for col in df.columns:
        if col != "TARGET" and df[col].nunique() > 10:
            if abs(skew(df[col])) > 1:
                df[f"{col}_log"] = np.log1p(df[col])

    # 4. Exemples de ratios et interactions
    if "var3" in df.columns and "var5" in df.columns:
        df["var3_over_var5"] = df["var3"] / (df["var5"] + 1)

    if "var1" in df.columns and "var2" in df.columns:
        df["var1_plus_var2"] = df["var1"] + df["var2"]

    # 5. Binning
    if "var4" in df.columns:
        df["var4_bin"] = pd.cut(
            df["var4"], bins=[0, 30, 40, 50, 60, 70, 100], labels=False
        )

    # 6. High value flags
    for col in df.columns:
        if col != "TARGET" and df[col].nunique() > 10:
            df[f"{col}_high"] = (df[col] > df[col].quantile(0.95)).astype(int)

    return df


def main():
    # Raw data
    # train = pd.read_csv(cfg['path']['train_path'])
    # test = pd.read_csv(cfg['path']['test_path'])

    # train = feature_engineering(train)
    # test = feature_engineering(test)

    # Processed data

    train = pd.read_csv(
        os.path.join(
            cfg["path"]["processed_data"],
            "train"
            + "_"
            + str(cfg["FEATURE_ENGINEERING"]["version"])
            + ".csv",
        )
    )
    test = pd.read_csv(
        os.path.join(
            cfg["path"]["processed_data"],
            "test" + "_" + str(cfg["FEATURE_ENGINEERING"]["version"]) + ".csv",
        )
    )

    y = train["TARGET"]
    X = train.drop(columns=["TARGET"])

    test_ids = test["ID"]
    X = X.drop(columns=["ID"])
    test = test.drop(columns=["ID"])

    X = pd.get_dummies(X)
    test = pd.get_dummies(test)
    X, test = X.align(test, join="left", axis=1, fill_value=0)

    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # base_models = [
    #     ("rf", RandomForestClassifier(n_estimators=100, random_state=42)),
    #     (
    #         "xgb",
    #         XGBClassifier(
    #             n_estimators=100,
    #             random_state=42,
    #             verbosity=0,
    #             use_label_encoder=False,
    #             eval_metric="logloss",
    #         ),
    #     ),
    #     ("lgb", LGBMClassifier(n_estimators=100, random_state=42)),
    #     ("svc", SVC(probability=True, random_state=42)),
    #     ("gb", GradientBoostingClassifier(n_estimators=100, random_state=42))
    # ]
    # base_models = [
    #     ("lgbm", LGBMClassifier(random_state=42)),
    #     ("catboost", CatBoostClassifier(verbose=0, random_state=42)),
    #     ("logreg", LogisticRegression(max_iter=1000, class_weight="balanced")),
    #     # (
    #     #     "mlp",
    #     #     MLPClassifier(
    #     #         hidden_layer_sizes=(128,), max_iter=600, random_state=42
    #     #     ),
    #     # ),
    #     ("xgb", XGBClassifier(random_state=42, np_estimators=200)),
    # ]

    base_models = [
        (
            "LR",
            LogisticRegression(**{"C": 0.7678243129497218, "penalty": "l1"}),
        ),
        ("KNN", KNeighborsClassifier(n_neighbors=15)),
        (
            "CART",
            DecisionTreeClassifier(
                **{
                    "criterion": "gini",
                    "max_depth": 3,
                    "max_features": 2,
                    "min_samples_leaf": 3,
                }
            ),
        ),
        ("NB", GaussianNB()),
        (
            "SVM",
            SVC(**{"C": 1.7, "kernel": "linear", "probability": True}),
        ),
        (
            "AB",
            AdaBoostClassifier(**{"learning_rate": 0.05, "n_estimators": 150}),
        ),
        (
            "GBM",
            GradientBoostingClassifier(
                **{"learning_rate": 0.01, "n_estimators": 100}
            ),
        ),
        ("RF", RandomForestClassifier()),
        ("ET", ExtraTreesClassifier()),
    ]

    S_train = np.zeros((X.shape[0], len(base_models)))
    S_test = np.zeros((test.shape[0], len(base_models)))

    for i, (name, model) in enumerate(base_models):
        logger.info(f"\n🔁 Training model: {name}")
        S_test_i = np.zeros((test.shape[0], n_splits))

        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            logger.info(f" - Fold {fold + 1}")
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

            model.fit(X_train, y_train)

            y_pred_val = model.predict(X_val)
            y_pred_test = model.predict(test)
            y_pred_prob = model.predict_proba(X_val)[:, 1]

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
    submission.to_csv(
        f"output/submission_"
        f"{str(cfg['FEATURE_ENGINEERING']['version']) + str(cfg['FEATURE_ENGINEERING']['version_training'])}.csv",
        index=False,
    )


if __name__ == "__main__":
    main()
