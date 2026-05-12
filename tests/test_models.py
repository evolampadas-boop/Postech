"""Testes dos modelos: treino, predição, persistência e recall mínimo."""

import tempfile
from pathlib import Path

import numpy as np
from sklearn.metrics import recall_score

from src.data_loader import load_data
from src.models import (
    get_all_models,
    load_model,
    save_model,
    train_all_models,
)
from src.preprocessing import build_preprocessing_pipeline, split_data


def _split():
    df = load_data(save_csv=False)
    return split_data(df)


def test_all_models_train_without_error():
    X_train, _, X_test, y_train, y_test, _ = _split()
    models = get_all_models()
    pipe = build_preprocessing_pipeline()
    trained = train_all_models(models, pipe, X_train, y_train)

    assert set(trained.keys()) == set(models.keys())
    for name, p in trained.items():
        preds = p.predict(X_test)
        assert len(preds) == len(X_test)
        assert set(np.unique(preds)).issubset({0, 1}), f"{name} retornou classes inválidas"


def test_at_least_one_model_recall_above_threshold():
    # Esse teste serve de "sanidade" - se nenhum modelo conseguir 85% de
    # recall, algo de errado tem com o pipeline ou com os dados.
    X_train, _, X_test, y_train, _, y_test = _split()
    models = get_all_models()
    pipe = build_preprocessing_pipeline()
    trained = train_all_models(models, pipe, X_train, y_train)

    recalls = [(name, recall_score(y_test, p.predict(X_test))) for name, p in trained.items()]
    best = max(recalls, key=lambda x: x[1])
    assert best[1] > 0.85, f"Melhor recall foi {best[1]} ({best[0]})"


def test_save_and_load_model_consistency():
    X_train, _, X_test, y_train, _, _ = _split()
    models = get_all_models()
    pipe = build_preprocessing_pipeline()
    trained = train_all_models({"Logistic Regression": models["Logistic Regression"]}, pipe, X_train, y_train)
    original = trained["Logistic Regression"]

    with tempfile.TemporaryDirectory() as tmp:
        save_model(original, "lr_test", path=Path(tmp))
        loaded = load_model("lr_test", path=Path(tmp))
        np.testing.assert_array_equal(original.predict(X_test), loaded.predict(X_test))
