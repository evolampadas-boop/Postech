"""Avaliação dos modelos: métricas, validação cruzada e plots."""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score

logger = logging.getLogger(__name__)


def _predict_proba_safe(pipeline, X):
    # Nem todo modelo tem predict_proba (SVC sem probability=True, por exemplo).
    try:
        return pipeline.predict_proba(X)[:, 1]
    except Exception:
        return None


def evaluate_model(pipeline, X_test, y_test, model_name=""):
    """Calcula as principais métricas de classificação binária."""
    y_pred = pipeline.predict(X_test)
    y_proba = _predict_proba_safe(pipeline, X_test)

    return {
        "model_name": model_name,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)) if y_proba is not None else None,
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred, zero_division=0),
    }


def evaluate_all_models(trained_pipelines, X_test, y_test):
    """Aplica evaluate_model em vários e devolve um DataFrame comparativo."""
    rows = []
    for name, pipe in trained_pipelines.items():
        m = evaluate_model(pipe, X_test, y_test, model_name=name)
        rows.append({
            "model": name,
            "accuracy": m["accuracy"],
            "precision": m["precision"],
            "recall": m["recall"],
            "f1_score": m["f1_score"],
            "roc_auc": m["roc_auc"],
        })
    return pd.DataFrame(rows).sort_values("recall", ascending=False).reset_index(drop=True)


def cross_validate_model(pipeline, X, y, cv=5, scoring="recall"):
    """Validação cruzada estratificada. Retorna (média, desvio)."""
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, X, y, cv=skf, scoring=scoring, n_jobs=-1)
    return float(scores.mean()), float(scores.std())


def plot_confusion_matrix(pipeline, X_test, y_test, model_name, save_path=None):
    y_pred = pipeline.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    total = cm.sum()

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    # uso o display do sklearn só pra desenhar a grade — o texto eu refaço
    # depois, porque quero mostrar valor absoluto E porcentagem em cada célula
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Benigno", "Maligno"])
    disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    for txt in ax.texts:
        txt.remove()
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            v = cm[i, j]
            pct = v / total * 100
            color = "white" if v > thresh else "black"
            ax.text(j, i, f"{v}\n({pct:.1f}%)", ha="center", va="center", color=color, fontsize=11)

    ax.set_title(f"Matriz de Confusão — {model_name}")
    ax.set_xlabel("O que o modelo previu")
    ax.set_ylabel("O que era de verdade")
    fig.text(0.5, -0.02,
             "Diagonal (canto sup. esq. e inf. dir.) = acertos. "
             "Canto inf. esq. = falso negativo (maligno que passou batido — o erro mais grave).",
             ha="center", fontsize=8.5, color="dimgray", wrap=True)
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_roc_curves(trained_pipelines, X_test, y_test, save_path=None):
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, pipe in trained_pipelines.items():
        y_proba = _predict_proba_safe(pipe, X_test)
        if y_proba is None:
            continue
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        ax.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})", linewidth=2)
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Linha do acaso (AUC = 0,5)")
    ax.set_xlabel("Taxa de falsos positivos (benigno classificado como maligno)")
    ax.set_ylabel("Taxa de verdadeiros positivos (recall)")
    ax.set_title("Curvas ROC — comparação entre modelos")
    ax.legend(loc="lower right", framealpha=0.9)
    ax.grid(alpha=0.3)
    fig.text(0.5, -0.02,
             "Quanto mais a curva sobe rápido pro canto superior esquerdo, melhor o modelo. "
             "AUC = área sob a curva (1,0 = perfeito; 0,5 = aleatório).",
             ha="center", fontsize=8.5, color="dimgray")
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_metrics_comparison(results_df, save_path=None):
    metrics_to_plot = ["accuracy", "recall", "f1_score"]
    metric_pt = {"accuracy": "Acurácia", "recall": "Recall", "f1_score": "F1-Score"}
    melted = results_df.melt(id_vars="model", value_vars=metrics_to_plot, var_name="metric", value_name="score")
    melted["metric"] = melted["metric"].map(metric_pt)

    fig, ax = plt.subplots(figsize=(11, 6))
    sns.barplot(data=melted, x="model", y="score", hue="metric", ax=ax)
    ax.set_title("Comparação de métricas por modelo")
    ax.set_ylim(0, 1.10)
    ax.set_xlabel("Modelo")
    ax.set_ylabel("Pontuação (0 a 1)")
    ax.legend(title="Métrica", loc="lower right")

    # valor em cima de cada barra — sem isso fica difícil comparar
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", padding=2, fontsize=8)

    plt.xticks(rotation=20, ha="right")
    fig.text(0.5, -0.02,
             "Recall é a métrica que importa mais aqui: quanto maior, menos tumores malignos "
             "passam batido. Acurácia sozinha engana em problema de classe desbalanceada.",
             ha="center", fontsize=8.5, color="dimgray")
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
