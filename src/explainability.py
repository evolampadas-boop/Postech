"""Explicabilidade do modelo: feature importance e SHAP."""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.inspection import permutation_importance

logger = logging.getLogger(__name__)

# Tentei usar SHAP mas no meu Windows o Controle de Aplicativo bloqueia o
# import em algumas situações. Por isso esse try/except: se não der, pulo
# os plots de SHAP e mantenho só feature importance.
try:
    import shap
    SHAP_AVAILABLE = True
except (ImportError, OSError) as _err:
    shap = None
    SHAP_AVAILABLE = False
    logger.warning("SHAP indisponível (%s). Vou pular as funções de SHAP.", _err)


def _extract_model_and_preprocessing(pipeline):
    model = pipeline.named_steps.get("model")
    pre = pipeline.named_steps.get("preprocessing")
    if model is None:
        raise ValueError("Pipeline não tem step 'model'.")
    return model, pre


def plot_feature_importance(trained_pipeline, feature_names, model_name, top_n=15,
                            X_val=None, y_val=None, save_path=None):
    """Plota as features mais importantes do modelo.

    Tenta usar feature_importances_ (árvores) ou coef_ (modelos lineares).
    Se nenhum dos dois existir, cai em permutation importance — que precisa
    de X_val e y_val.
    """
    model, _ = _extract_model_and_preprocessing(trained_pipeline)

    if hasattr(model, "feature_importances_"):
        importances = np.asarray(model.feature_importances_)
    elif hasattr(model, "coef_"):
        importances = np.abs(np.ravel(model.coef_))
    else:
        if X_val is None or y_val is None:
            raise ValueError("Modelo sem feature_importances_; passa X_val e y_val pra usar permutation importance.")
        result = permutation_importance(trained_pipeline, X_val, y_val, n_repeats=10, random_state=42, n_jobs=-1)
        importances = result.importances_mean

    order = np.argsort(importances)[::-1][:top_n]
    top_features = [feature_names[i] for i in order]
    top_values = importances[order]

    fig, ax = plt.subplots(figsize=(9, max(4, top_n * 0.32)))
    bars = ax.barh(range(len(top_features))[::-1], top_values[::-1], color="steelblue")
    ax.set_yticks(range(len(top_features))[::-1])
    ax.set_yticklabels(top_features[::-1])
    ax.set_xlabel("Importância (quanto maior, mais a feature pesa na decisão)")
    ax.set_title(f"Top {top_n} variáveis mais importantes — {model_name}")
    ax.grid(axis="x", alpha=0.3)
    # margem extra à direita pra caber o número anotado
    ax.set_xlim(right=ax.get_xlim()[1] * 1.12)
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
    fig.text(0.5, -0.02,
             "Para a Regressão Logística uso o |coeficiente|; para árvores, o feature_importances_; "
             "se nada disso existir, caio em permutation importance.",
             ha="center", fontsize=8.5, color="dimgray")
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def compute_shap_values(trained_pipeline, X_data, feature_names):
    """Calcula SHAP values. Escolhe o explainer adequado ao tipo de modelo."""
    if not SHAP_AVAILABLE:
        raise RuntimeError("SHAP não está disponível nesse ambiente.")

    model, pre = _extract_model_and_preprocessing(trained_pipeline)
    X_transformed = pre.transform(X_data) if pre is not None else X_data.values
    import pandas as pd
    X_df = pd.DataFrame(X_transformed, columns=feature_names, index=X_data.index)

    class_name = model.__class__.__name__
    tree_models = {"RandomForestClassifier", "DecisionTreeClassifier", "GradientBoostingClassifier", "ExtraTreesClassifier"}

    if class_name in tree_models or hasattr(model, "estimators_"):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_df)
        # alguns retornam lista [classe_0, classe_1] — pego a classe positiva
        if isinstance(shap_values, list) and len(shap_values) == 2:
            shap_values = shap_values[1]
        expected_value = explainer.expected_value
        if isinstance(expected_value, (list, np.ndarray)) and np.ndim(expected_value) > 0:
            expected_value = expected_value[1] if len(expected_value) > 1 else expected_value[0]
    elif class_name == "LogisticRegression":
        explainer = shap.LinearExplainer(model, X_df)
        shap_values = explainer.shap_values(X_df)
        expected_value = explainer.expected_value
    else:
        # último recurso: KernelExplainer (lento)
        background = shap.sample(X_df, min(50, len(X_df)), random_state=42)
        explainer = shap.KernelExplainer(model.predict_proba, background)
        shap_values = explainer.shap_values(X_df, nsamples=100, silent=True)
        if isinstance(shap_values, list) and len(shap_values) == 2:
            shap_values = shap_values[1]
        expected_value = explainer.expected_value
        if isinstance(expected_value, (list, np.ndarray)) and np.ndim(expected_value) > 0:
            expected_value = expected_value[1] if len(expected_value) > 1 else expected_value[0]

    return {
        "values": shap_values,
        "expected_value": expected_value,
        "X_transformed": X_df,
        "feature_names": feature_names,
    }


def plot_shap_summary(shap_values, X_data, feature_names, save_path=None):
    fig = plt.figure(figsize=(9, 6))
    shap.summary_plot(shap_values["values"], shap_values["X_transformed"],
                      feature_names=feature_names, show=False)
    plt.title("Resumo SHAP — impacto de cada variável")
    plt.xlabel("Valor SHAP")
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_shap_waterfall(shap_values, X_data, feature_names, sample_index=0, save_path=None):
    values = shap_values["values"]
    expected_value = shap_values["expected_value"]
    X_t = shap_values["X_transformed"]
    fig = plt.figure(figsize=(9, 6))
    explanation = shap.Explanation(
        values=values[sample_index],
        base_values=expected_value,
        data=X_t.iloc[sample_index].values,
        feature_names=feature_names,
    )
    shap.plots.waterfall(explanation, show=False)
    plt.title(f"SHAP waterfall — paciente {sample_index}")
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_shap_bar(shap_values, feature_names, save_path=None, top_n=15):
    """Bar plot com a média absoluta dos SHAP values por feature.

    Limito ao top_n porque com 30 features o gráfico fica grande demais
    e as features de baixa importância só atrapalham a leitura.
    """
    values = shap_values["values"]
    mean_abs = np.mean(np.abs(values), axis=0)
    order = np.argsort(mean_abs)[::-1][:top_n]
    names = np.array(feature_names)[order][::-1]
    vals = mean_abs[order][::-1]

    fig, ax = plt.subplots(figsize=(9, max(4, top_n * 0.32)))
    bars = ax.barh(names, vals, color="darkorange")
    ax.set_xlabel("|SHAP| médio — quanto a feature mexe na predição, em média")
    ax.set_title(f"Top {top_n} variáveis mais influentes (SHAP)")
    ax.grid(axis="x", alpha=0.3)
    ax.set_xlim(right=ax.get_xlim()[1] * 1.12)
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
    fig.text(0.5, -0.02,
             "Valor alto = essa variável empurra com força a predição (pra maligno ou benigno). "
             "Aqui só olhamos a magnitude do impacto, não a direção.",
             ha="center", fontsize=8.5, color="dimgray")
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
