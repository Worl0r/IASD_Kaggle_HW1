from os import ftruncate
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils.validation import _num_features
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from imblearn.over_sampling import SMOTE
import lightgbm as lgb
from utils import get_logger, load_config
import matplotlib.pyplot as plt
import seaborn as sns
import featuretools as ft
import matplotlib
import pickle

matplotlib.use("WebAgg")
logger = get_logger()

cfg = load_config("configuration.yaml")


def feature_engineering(
    train, train_original, log_col, product_col, divided_col
):
    ## Feature Engineering
    # Indication of missing values
    train["missing_var5"] = train["var5"].isnull().astype(int)
    train["missing_var10"] = train["var10"].isnull().astype(int)

    # Fill missing values with mean
    train["var5"] = train["var5"].fillna(train["var5"].mean())
    train["var10"] = train["var10"].fillna(0)

    # Log transform for skewed variables
    for col in log_col:
        train[col] = np.log1p(train[col])

    # Binning
    train["var2_bin"] = pd.cut(
        train["var2"], bins=[0, 30, 40, 50, 60, 70, 100], labels=False
    )
    train["var5_bin"] = pd.qcut(
        train["var5"], q=5, labels=False, duplicates="drop"
    )

    # Special variable
    train["special"] = train["var3"] + train["var9"] + train["var7"]

    # Threshold for outliers
    train["special_thresh"] = (train["special"] > 0).astype(int)
    train["var1_thresh"] = (train["var1"] > 0.9).astype(int)
    train["var4_thresh"] = (train["var4"] > 1).astype(int)

    # Creat new variables
    for col in product_col:
        for col2 in divided_col:
            train[f"{col}_divided_{col2}"] = train[col] / (train[col2] + 1e-5)
            train[f"{col}_product_{col2}"] = train[col] * (train[col2] + 1e-5)

    # Summary
    logger.info(
        f"{train.shape[1] - train_original.shape[1]}"
        f" nouvelles colonnes ajoutées."
    )
    logger.info(
        f"Aperçu des nouvelles colonnes :"
        f" {train.columns.difference(train_original.columns).tolist()}"
    )
    logger.info(train.head())

    return train


def plot_corr_matrix(corr_matrix, filename):
    plt.figure(figsize=(20, 20))

    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Matrice de corrélation")

    plt.savefig("./figures/data_processing/" + filename + ".png")
    plt.close()


def remove_highly_correlated_features(X, threshold=0.9, plot=True):
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    to_drop = [
        column for column in upper.columns if any(upper[column] > threshold)
    ]

    logger.info(
        f"🔍 {len(to_drop)} colonnes supprimées pour forte corrélation (> {threshold})"
    )

    X = X.drop(columns=to_drop)

    # Plot correlation matrix
    if plot:
        plot_corr_matrix(
            corr_matrix,
            filename="corr_matrix",
        )
        plot_corr_matrix(
            X.corr().abs(),
            filename=f"corr_matrix_{len(to_drop)}_features_dropped",
        )

    return X, to_drop


def select_important_features(X, y, lgb_params, num_top_features):
    model = lgb.LGBMClassifier(**lgb_params, n_estimators=100)
    model.fit(X, y)

    feature_importance = pd.DataFrame(
        {"feature": X.columns, "importance": model.feature_importances_}
    ).sort_values(by="importance", ascending=False)

    top_features = feature_importance.head(num_top_features)[
        "feature"
    ].tolist()
    logger.info(f"🎯 {num_top_features} meilleures features sélectionnées.")
    return X[top_features], feature_importance


def pipeline(
    df, df_original, y, log_col, product_col, divided_col, lgb_params
):
    df = feature_engineering(
        df, df_original, log_col, product_col, divided_col
    )

    df, col_to_drop = remove_highly_correlated_features(
        df, threshold=0.9, plot=False
    )

    return df


def use_featuretools(df_train, df_test, lgb_params, num_top_features):
    train_original = df_train.copy()
    test_original = df_test.copy()

    df_train["client_id"] = df_train.index
    y = df_train["TARGET"]
    df_train = df_train.drop(columns=["TARGET", "ID"])
    df_test = df_test.drop(columns=["ID"])
    df_test["client_id"] = df_test.index

    es = ft.EntitySet(id="train")
    es = es.add_dataframe(
        dataframe_name="clients", dataframe=df_train, index="client_id"
    )
    feature_matrix_train, feature_defs = ft.dfs(
        entityset=es,
        target_dataframe_name="clients",
        max_depth=2,
        trans_primitives=[
            "add_numeric",
            "multiply_numeric",
            "divide_numeric",
            "percentile",
            "absolute",
            "negate",
        ],
        verbose=True,
    )

    logger.info(
        f"✅ Il y a maintenant {feature_matrix_train.shape[1]} colonnes !"
    )

    X_generated = feature_matrix_train.copy()
    X_generated["TARGET"] = y.values

    es_test = ft.EntitySet(id="test")

    es_test = es_test.add_dataframe(
        dataframe_name="clients", dataframe=df_test, index="customer_id"
    )

    feature_matrix_test = ft.calculate_feature_matrix(
        features=feature_defs, entityset=es_test, verbose=True
    )

    feature_matrix_train = feature_matrix_train.reset_index(drop=True)
    feature_matrix_test = feature_matrix_test.reset_index(drop=True)

    X_generated, col_to_drop = remove_highly_correlated_features(
        X_generated, threshold=0.9, plot=False
    )
    feature_matrix_test = feature_matrix_test.drop(
        columns=col_to_drop, errors="ignore"
    )

    X_generated, feature_importance = select_important_features(
        X_generated, y, lgb_params, num_top_features=num_top_features
    )
    logger.info(
        f"\n✅ Shape finale des features sélectionnées: {X_generated.shape}"
    )
    feature_matrix_test = feature_matrix_test.drop(
        columns=[
            col
            for col in feature_matrix_test.columns
            if col not in X_generated.columns
        ]
    )

    X_generated = pd.concat([train_original["ID"], X_generated], axis=1)
    feature_matrix_test = pd.concat(
        [test_original["ID"], feature_matrix_test], axis=1
    )
    return X_generated, feature_matrix_test


def use_manual_feature_engineering(
    train,
    test,
    log_col,
    product_col,
    divided_col,
    lgb_params,
):
    train_original = train.copy()
    test_original = test.copy()

    y = train["TARGET"]
    train = train.drop(columns=["TARGET", "ID"])
    test = test.drop(columns=["ID"])

    pip_train = pipeline(
        train, train_original, y, log_col, product_col, divided_col, lgb_params
    )
    pip_test = pipeline(
        test, test_original, y, log_col, product_col, divided_col, lgb_params
    )

    pip_train = pd.concat([train_original["ID"], pip_train], axis=1)
    pip_train = pd.concat([pip_train, y], axis=1)
    pip_test = pd.concat([test_original["ID"], pip_test], axis=1)

    return pip_train, pip_test


def main():
    train = pd.read_csv(cfg["path"]["train_path"])
    test = pd.read_csv(cfg["path"]["test_path"])

    y = train["TARGET"]

    version = str(cfg["FEATURE_ENGINEERING"]["version"])

    lgb_params = {
        "objective": "binary",
        "metric": "auc",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "n_jobs": -1,
        "scale_pos_weight": (y == 0).sum() / (y == 1).sum(),
    }

    if cfg["FEATURE_ENGINEERING"]["mode"] == "featuretools":
        pip_train, pip_test = use_featuretools(
            train,
            test,
            lgb_params,
            num_top_features=40,
        )

    elif cfg["FEATURE_ENGINEERING"]["mode"] == "manual":
        log_col = ["var5", "var1", "var4"]
        product_col = ["var5", "var4"]  # train.columns
        divided_col = ["var6", "var10", "var2", "var5"]  # train.columns

        pip_train, pip_test = use_manual_feature_engineering(
            train,
            test,
            log_col,
            product_col,
            divided_col,
            lgb_params,
        )
    else:
        raise ValueError(
            "Mode de feature engineering non reconnu. "
            "Veuillez choisir entre 'featuretools' ou 'manual'."
        )

    # Save
    pip_train.to_csv(f"./data/processed/train_{version}.csv", index=False)
    pip_test.to_csv(f"./data/processed/test_{version}.csv", index=False)
    logger.info(
        "✅ Données traitées et enregistrées dans le dossier ./data/processed/"
    )


if __name__ == "__main__":
    main()
