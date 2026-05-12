"""Testes do data loader e do pré-processamento."""

import numpy as np

from src.data_loader import load_data
from src.preprocessing import (
    build_preprocessing_pipeline,
    encode_target,
    get_correlation_matrix,
    split_data,
)


def test_load_data_shape():
    df = load_data(save_csv=False)
    assert df.shape == (569, 31)
    assert "diagnosis" in df.columns


def test_no_nulls_after_pipeline():
    df = load_data(save_csv=False)
    X = df.drop(columns=["diagnosis"])
    pipe = build_preprocessing_pipeline()
    X_t = pipe.fit_transform(X)
    assert not np.isnan(X_t).any()


def test_split_proportions():
    df = load_data(save_csv=False)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df, test_size=0.2, val_size=0.1)
    total = len(df)
    assert abs(len(X_test) / total - 0.2) < 0.02
    assert abs(len(X_val) / total - 0.1) < 0.02
    assert len(X_train) + len(X_val) + len(X_test) == total
    assert len(y_train) == len(X_train)


def test_split_stratified():
    # garante que a proporção de classes é mantida nos três conjuntos
    df = load_data(save_csv=False)
    _, _, _, y_train, y_val, y_test = split_data(df)
    orig_ratio = df["diagnosis"].mean()
    for y in (y_train, y_val, y_test):
        assert abs(y.mean() - orig_ratio) < 0.05


def test_encode_target_passthrough():
    df = load_data(save_csv=False)
    encoded = encode_target(df["diagnosis"])
    assert set(encoded.unique()).issubset({0, 1})


def test_correlation_matrix_square():
    df = load_data(save_csv=False)
    corr = get_correlation_matrix(df)
    assert corr.shape[0] == corr.shape[1]
