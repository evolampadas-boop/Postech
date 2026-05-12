"""Testes da API. Treino um modelo só pra esses testes e aponto o Predictor pra ele."""

import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api import main as api_main
from api.predictor import Predictor
from src.data_loader import load_data
from src.models import get_all_models, save_model, train_all_models
from src.preprocessing import build_preprocessing_pipeline, split_data


@pytest.fixture(scope="module")
def trained_models_dir():
    df = load_data(save_csv=False)
    X_train, _, _, y_train, _, _ = split_data(df)
    pipe = build_preprocessing_pipeline()
    models = {"Logistic Regression": get_all_models()["Logistic Regression"]}
    trained = train_all_models(models, pipe, X_train, y_train)

    tmpdir = tempfile.mkdtemp()
    save_model(trained["Logistic Regression"], "best_model", path=Path(tmpdir))
    yield Path(tmpdir)


@pytest.fixture(scope="module")
def client(trained_models_dir):
    # Injeto um Predictor que aponta pro diretório temporário
    api_main._predictor = Predictor(models_dir=trained_models_dir)
    yield TestClient(api_main.app)
    api_main._predictor = None


@pytest.fixture
def valid_payload():
    return {
        "mean radius": 17.99, "mean texture": 10.38, "mean perimeter": 122.8,
        "mean area": 1001.0, "mean smoothness": 0.1184, "mean compactness": 0.2776,
        "mean concavity": 0.3001, "mean concave points": 0.1471, "mean symmetry": 0.2419,
        "mean fractal dimension": 0.07871,
        "radius error": 1.095, "texture error": 0.9053, "perimeter error": 8.589,
        "area error": 153.4, "smoothness error": 0.006399, "compactness error": 0.04904,
        "concavity error": 0.05373, "concave points error": 0.01587,
        "symmetry error": 0.03003, "fractal dimension error": 0.006193,
        "worst radius": 25.38, "worst texture": 17.33, "worst perimeter": 184.6,
        "worst area": 2019.0, "worst smoothness": 0.1622, "worst compactness": 0.6656,
        "worst concavity": 0.7119, "worst concave points": 0.2654,
        "worst symmetry": 0.4601, "worst fractal dimension": 0.1189,
    }


def test_health_returns_200(client):
    r = client.get("/")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "name" in body
    assert "version" in body


def test_model_info_returns_features(client):
    r = client.get("/model/info")
    assert r.status_code == 200
    body = r.json()
    assert "model_name" in body
    assert len(body["features"]) == 30


def test_predict_valid_payload(client, valid_payload):
    r = client.post("/predict", json=valid_payload)
    assert r.status_code == 200
    body = r.json()
    assert body["diagnosis"] in {"Maligno", "Benigno"}
    assert body["diagnosis_code"] in {0, 1}
    assert 0.0 <= body["confidence"] <= 100.0
    assert body["risk_level"]
    assert body["model_used"]


def test_predict_invalid_payload_returns_422(client):
    r = client.post("/predict", json={"mean radius": 1.0})
    assert r.status_code == 422


def test_disclaimer_always_present(client, valid_payload):
    # O disclaimer é exigência do projeto - tem que vir sempre.
    r = client.post("/predict", json=valid_payload)
    assert r.status_code == 200
    assert "disclaimer" in r.json()
    assert r.json()["disclaimer"]
