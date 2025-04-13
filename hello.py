import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew

from utils import get_logger, load_config

logger = get_logger()

cfg = load_config("configuration.yaml")

def main():
    df = pd.read_csv(cfg['path']['train_path'])

    figure_path = cfg['path']['raw_data_figures']

    logger.info("Aperçu du dataset :")
    logger.info(df.head())

    logger.info("\nInfos générales :")
    logger.info(df.info())

    logger.info("\nStatistiques descriptives :")
    logger.info(df.describe())

    logger.info("\nDistribution de la target :")
    logger.info(df['TARGET'].value_counts(normalize=True))

    logger.info("\nValeurs manquantes :")
    logger.info(df.isnull().sum())

    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numerical_cols.remove('TARGET')  # on exclut la cible

    logger.info("\nSkewness des variables :")
    for col in numerical_cols:
        s = skew(df[col].dropna())
        logger.info(f"{col} : skewness = {s:.2f}")

    for col in numerical_cols:
        plt.figure(figsize=(6, 4))
        sns.histplot(df[col], bins=50, kde=True)
        plt.title(f'Distribution de {col}')
        plt.xlabel(col)
        plt.ylabel('Fréquence')
        plt.tight_layout()

        path = os.path.join(figure_path, f"{col}_distribution.png")
        plt.savefig(path)

    plt.figure(figsize=(10, 8))
    corr = df[numerical_cols + ['TARGET']].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Matrice de corrélation")
    path = os.path.join(figure_path, "correlation_matrix.png")
    plt.savefig(path)

    for col in numerical_cols:
        plt.figure(figsize=(6, 4))
        sns.boxplot(x='TARGET', y=col, data=df)
        plt.title(f'{col} par classe TARGET')
        plt.tight_layout()
        path = os.path.join(figure_path, f"{col}_boxplot.png")
        plt.savefig(path)



if __name__ == "__main__":
    main()
