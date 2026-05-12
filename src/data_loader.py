"""Carrega o dataset Breast Cancer Wisconsin direto do sklearn."""

import logging
from pathlib import Path

import pandas as pd
from sklearn.datasets import load_breast_cancer

logger = logging.getLogger(__name__)

TARGET_COL = "diagnosis"
DEFAULT_CSV_PATH = Path(__file__).resolve().parent.parent / "data" / "breast_cancer.csv"


def load_data(save_csv=True, csv_path=DEFAULT_CSV_PATH):
    """Carrega o dataset e devolve um DataFrame com as 30 features + coluna diagnosis.

    O sklearn entrega 0 = maligno e 1 = benigno. Eu inverti pra ficar 1 = maligno
    porque é a classe positiva (a que a gente quer detectar).
    """
    raw = load_breast_cancer(as_frame=True)
    df = raw.frame.copy()
    df = df.rename(columns={"target": TARGET_COL})
    df[TARGET_COL] = (1 - df[TARGET_COL]).astype(int)

    if save_csv:
        csv_path = Path(csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)
        logger.info("Dataset salvo em %s", csv_path)

    return df


def get_feature_names():
    return list(load_breast_cancer().feature_names)


def get_data_info():
    """Resumo do dataset: shape, distribuição de classes, nulos."""
    df = load_data(save_csv=False)
    class_counts = df[TARGET_COL].value_counts().sort_index().to_dict()
    total = len(df)
    class_pct = {int(k): round(v / total * 100, 2) for k, v in class_counts.items()}
    nulls = df.isna().sum().to_dict()

    return {
        "shape": df.shape,
        "class_counts": {int(k): int(v) for k, v in class_counts.items()},
        "class_percent": class_pct,
        "features": [c for c in df.columns if c != TARGET_COL],
        "nulls_per_column": {k: int(v) for k, v in nulls.items()},
        "target_mapping": {0: "Benigno", 1: "Maligno"},
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    info = get_data_info()
    print("Shape:", info["shape"])
    print("Classes:", info["class_counts"])
    print("Percentual:", info["class_percent"])
    load_data(save_csv=True)
