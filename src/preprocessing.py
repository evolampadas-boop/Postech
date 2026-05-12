"""Pré-processamento e split dos dados."""

import logging

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

logger = logging.getLogger(__name__)


def build_preprocessing_pipeline():
    # imputer só por garantia, o dataset não tem nulos. scaler é importante
    # porque KNN e SVM são sensíveis à escala.
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])


def build_preprocessing_pipeline_pca(n_components=10):
    """Mesma coisa do pipeline padrão mas com PCA no final.

    Útil pra visualização 2D/3D e como redução de dimensionalidade.
    """
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=n_components, random_state=42)),
    ])


def split_data(df, target_col="diagnosis", test_size=0.2, val_size=0.1, random_state=42):
    """Split estratificado em treino/validação/teste.

    O val_size é proporcional ao dataset original. Faço em dois passos pra manter
    a proporção das classes em cada conjunto.
    """
    if not 0 < test_size < 1:
        raise ValueError("test_size deve estar entre 0 e 1")
    if not 0 <= val_size < 1:
        raise ValueError("val_size deve estar entre 0 e 1")
    if test_size + val_size >= 1:
        raise ValueError("test_size + val_size precisa ser menor que 1")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    if val_size > 0:
        relative_val = val_size / (1.0 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=relative_val,
            stratify=y_temp,
            random_state=random_state,
        )
    else:
        X_train, y_train = X_temp, y_temp
        X_val = X_temp.iloc[0:0]
        y_val = y_temp.iloc[0:0]

    logger.info("Split: treino=%d val=%d teste=%d", len(X_train), len(X_val), len(X_test))
    return X_train, X_val, X_test, y_train, y_val, y_test


def encode_target(series):
    """Garante 0/1 numérico. Se vier categórico aplica LabelEncoder."""
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(int)
    enc = LabelEncoder()
    return pd.Series(enc.fit_transform(series.astype(str)), index=series.index, name=series.name)


def get_correlation_matrix(df):
    return df.select_dtypes(include=["number"]).corr()
