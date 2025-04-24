from os import ftruncate
import numpy as np
import pandas as pd
import lightgbm as lgb
from utils import get_logger, load_config
import matplotlib.pyplot as plt
import seaborn as sns
import featuretools as ft
import matplotlib
import pickle
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

matplotlib.use("WebAgg")
logger = get_logger()

cfg = load_config("configuration.yaml")


def remove_outliers(df, column_name, iqr_multiplier=2):
    # From: https://www.kaggle.com/code/python4sp/credit-scoring-end-to-end-auto-submit-g-c
    q1 = df[column_name].quantile(0.25)
    q3 = df[column_name].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - iqr_multiplier * iqr
    upper_bound = q3 + iqr_multiplier * iqr
    df = df[
        (df[column_name] >= lower_bound) & (df[column_name] <= upper_bound)
    ]
    return df


def feature_engineering(
    train,
    train_original,
    log_col,
    product_col,
    divided_col,
    mode,
    iqr_multiplier,
):
    exclude_col = [
        "ID",
        "TARGET",
    ]

    features_col = train.columns.tolist()
    logger.info(f"🔍 {len(features_col)} columns in the dataset")

    ## Feature Engineering
    # Indication of missing values
    missing_var5 = train["var5"].isna()

    # Fill missing values with mean
    si = SimpleImputer(strategy="mean").fit(train[["var5"]])
    train["var5"] = si.transform(train[["var5"]])

    # TODO: Remove when var5 is zero ?

    # Adjust var10: The debt ratio
    train.loc[missing_var5, "var4"] = (
        train.loc[missing_var5, "var4"] / train.loc[missing_var5, "var5"]
    )

    # Remove outliers for debt ratio
    # if mode == "train":
    # logger.info(f"Shape before outlier removal: {train.shape}")
    # for col in features_col:
    #     if col in exclude_col:
    #         continue
    #     train = remove_outliers(train, col, iqr_multiplier=iqr_multiplier)

    # logger.info(f"Shape after outlier removal: {train.shape}")

    # # TODO: Custom
    # train.drop(
    #     train[(train["var3"] > 50) & (train["TARGET"] == 0)].index,
    #     inplace=True,
    # )
    # train.drop(
    #     train[(train.var4 > 5) & (train["TARGET"] == 0)].index,
    #     inplace=True,
    # )
    # train.drop(
    #     train[(train.var5 > 30000) & (train["TARGET"] == 0)].index,
    #     inplace=True,
    # )
    # train.drop(
    #     train[(train.var6 > 20) & (train["TARGET"] == 0)].index,
    #     inplace=True,
    # )
    # train.drop(
    #     train[(train.var10 > 6) & (train["TARGET"] == 0)].index,
    #     inplace=True,
    # )

    # Replace na values
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

    # Standardization
    # scaler = StandardScaler()

    # keep_cols = train[exclude_col].reset_index(drop=True)
    # train = pd.DataFrame(
    #     scaler.fit_transform(train.drop(columns=exclude_col)),
    #     columns=[col for col in train.columns if not col in exclude_col],
    # )
    # train[exclude_col] = keep_cols

    # Summary
    logger.info(
        f"{train.shape[1] - train_original.shape[1]}"
        f" nouvelles colonnes ajoutées."
    )
    logger.info(
        f"Aperçu des nouvelles colonnes :"
        f" {train.columns.difference(train_original.columns).tolist()}"
    )

    return train


def plot_corr_matrix(corr_matrix, filename):
    plt.figure(figsize=(20, 20))

    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Matrice de corrélation")

    plt.savefig("./figures/data_processing/" + filename + ".png")
    plt.close()


def remove_highly_correlated_features(
    X, threshold=0.9, plot=True, col_test_to_drop=None
):
    if col_test_to_drop is None:
        corr_matrix = X.corr().abs()
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

        to_drop = [
            column
            for column in upper.columns
            if any(upper[column] > threshold) and column != "ID"
        ]

        logger.info(
            f"🔍 {len(to_drop)} colonnes supprimées"
            f" pour forte corrélation (> {threshold})"
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
    else:
        logger.info(
            f"🔍 {len(col_test_to_drop)} colonnes supprimées pour"
            f"forte corrélation dans train dataset"
        )
        return X.drop(columns=col_test_to_drop), col_test_to_drop


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
    df,
    df_original,
    log_col,
    product_col,
    divided_col,
    col_test_to_drop,
    iqr_multiplier,
):
    mode = "train" if col_test_to_drop is None else "test"

    df = feature_engineering(
        df,
        df_original,
        log_col,
        product_col,
        divided_col,
        mode,
        iqr_multiplier,
    )

    df, col_to_drop = remove_highly_correlated_features(
        df,
        threshold=0.9,
        plot=False,
        col_test_to_drop=col_test_to_drop,
    )

    return df, col_to_drop


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
        X_generated,
        threshold=0.9,
        plot=False,
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
    iqr_multiplier,
):
    train_original = train.copy()
    test_original = test.copy()

    pip_train, col_to_drop = pipeline(
        train,
        train_original,
        log_col,
        product_col,
        divided_col,
        None,
        iqr_multiplier,
    )
    pip_test, _ = pipeline(
        test,
        test_original,
        log_col,
        product_col,
        divided_col,
        col_to_drop,
        None,
    )

    assert pip_train.drop(columns=["TARGET"]).columns.equals(
        pip_test.columns
    ), "Les colonnes de train et test ne sont pas les mêmes !"

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
        iqr_multiplier = 2.5

        pip_train, pip_test = use_manual_feature_engineering(
            train,
            test,
            log_col,
            product_col,
            divided_col,
            iqr_multiplier,
        )
    else:
        raise ValueError(
            "Mode de feature engineering non reconnu. "
            "Veuillez choisir entre 'featuretools' ou 'manual'."
        )
    # Checking
    logger.info(
        f"Summary of the nan values in the train set: {pip_train.isna().sum()}"
    )
    logger.info(
        f"Summary of the nan values in the test set: {pip_test.isna().sum()}"
    )

    # Save
    pip_train.to_csv(f"./data/processed/train_{version}.csv", index=False)
    pip_test.to_csv(f"./data/processed/test_{version}.csv", index=False)
    logger.info(
        "✅ Données traitées et enregistrées dans le dossier ./data/processed/"
    )


if __name__ == "__main__":
    main()
