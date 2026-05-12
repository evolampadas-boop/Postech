"""Modelos do projeto: criação, treino, salvar/carregar e seleção do melhor."""

import logging
from pathlib import Path

import joblib
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

logger = logging.getLogger(__name__)

DEFAULT_MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def get_all_models():
    """Os 5 modelos que vou comparar.

    Escolhi modelos bem distintos: um linear, um baseado em vizinhança, um de
    margem (SVM), e dois de árvore (um ensemble e uma árvore simples).
    """
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "SVM": SVC(probability=True, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    }


def _build_full_pipeline(preprocessing_pipeline, model):
    return Pipeline([("preprocessing", preprocessing_pipeline), ("model", model)])


def train_single_model(model_name, model, preprocessing_pipeline, X_train, y_train):
    logger.info("Treinando: %s", model_name)
    pipeline = _build_full_pipeline(clone(preprocessing_pipeline), clone(model))
    pipeline.fit(X_train, y_train)
    return pipeline


def train_all_models(models, preprocessing_pipeline, X_train, y_train):
    trained = {}
    for name, estimator in models.items():
        trained[name] = train_single_model(name, estimator, preprocessing_pipeline, X_train, y_train)
    return trained


def save_model(pipeline, model_name, path=DEFAULT_MODELS_DIR):
    """Salva o pipeline com joblib em <path>/<model_name>.pkl."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    safe_name = model_name.lower().replace(" ", "_")
    out = path / f"{safe_name}.pkl"
    joblib.dump(pipeline, out)
    logger.info("Modelo '%s' salvo em %s", model_name, out)
    return out


def load_model(model_name, path=DEFAULT_MODELS_DIR):
    path = Path(path)
    safe_name = model_name.lower().replace(" ", "_")
    candidate = path / f"{safe_name}.pkl"
    if not candidate.exists():
        # fallback caso alguém tenha salvo com nome diferente
        candidate = path / model_name
    if not candidate.exists():
        raise FileNotFoundError(f"Modelo não encontrado: {candidate}")
    return joblib.load(candidate)


def get_best_model(results_dict, metric="recall"):
    """Pega o nome e métricas do modelo com maior valor da métrica informada."""
    if not results_dict:
        raise ValueError("Não tem resultados pra comparar")
    best_name = max(results_dict.keys(), key=lambda k: results_dict[k].get(metric, float("-inf")))
    return best_name, results_dict[best_name]
