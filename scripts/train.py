"""Script de treino: roda o pipeline completo e salva o melhor modelo.

Exemplos:
    python scripts/train.py
    python scripts/train.py --model "Random Forest"
    python scripts/train.py --metric f1_score --save-all
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Inclui a raiz do projeto no PYTHONPATH pra conseguir importar src/ e api/
# quando o script é chamado de qualquer pasta.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_loader import get_feature_names, load_data
from src.evaluate import cross_validate_model, evaluate_all_models, evaluate_model
from src.models import (
    get_all_models,
    get_best_model,
    save_model,
    train_all_models,
)
from src.preprocessing import build_preprocessing_pipeline, split_data

logger = logging.getLogger("train")


def parse_args():
    p = argparse.ArgumentParser(description="Treina os modelos do Tech Challenge Fase 1.")
    p.add_argument("--model", type=str, default=None, help="Treinar só um modelo (ex: 'Random Forest').")
    p.add_argument("--metric", type=str, default="recall", help="Métrica usada pra escolher o melhor modelo.")
    p.add_argument("--save-all", action="store_true", help="Salvar todos os modelos, não só o melhor.")
    return p.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args()

    print("Carregando dataset...")
    df = load_data(save_csv=True)
    print(f"Shape: {df.shape}")

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

    preprocessing = build_preprocessing_pipeline()
    all_models = get_all_models()

    if args.model:
        if args.model not in all_models:
            print(f"Modelo '{args.model}' não existe. Opções: {list(all_models)}")
            return 1
        models_to_train = {args.model: all_models[args.model]}
    else:
        models_to_train = all_models

    print(f"Treinando {len(models_to_train)} modelo(s)...")
    trained = train_all_models(models_to_train, preprocessing, X_train, y_train)

    print("\nAvaliação na validação:")
    val_results_df = evaluate_all_models(trained, X_val, y_val)
    print(val_results_df.to_string(index=False))

    print(f"\nValidação cruzada (5-fold, scoring={args.metric}):")
    cv_summary = {}
    for name, pipe in trained.items():
        mean, std = cross_validate_model(pipe, X_train, y_train, cv=5, scoring=args.metric)
        cv_summary[name] = {"mean": mean, "std": std}
        print(f"  {name:<22} {mean:.4f} ± {std:.4f}")

    val_results_dict = {row["model"]: row.to_dict() for _, row in val_results_df.iterrows()}
    best_name, best_metrics = get_best_model(val_results_dict, metric=args.metric)
    best_pipeline = trained[best_name]
    print(f"\nMelhor modelo (por {args.metric}): {best_name} ({best_metrics[args.metric]:.4f})")

    print("Avaliando o melhor modelo no conjunto de teste (holdout)...")
    test_metrics = evaluate_model(best_pipeline, X_test, y_test, model_name=best_name)

    models_dir = ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    save_model(best_pipeline, "best_model", path=models_dir)

    if args.save_all:
        for name, pipe in trained.items():
            save_model(pipe, name, path=models_dir)

    results_payload = {
        "best_model": best_name,
        "selection_metric": args.metric,
        "features": get_feature_names(),
        "validation_results": val_results_dict,
        "cross_validation": cv_summary,
        "test_metrics": {k: v for k, v in test_metrics.items() if k != "classification_report"},
        "test_classification_report": test_metrics["classification_report"],
    }
    out_path = models_dir / "training_results.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2, ensure_ascii=False)
    print(f"Resultados salvos em {out_path}")

    print("\n========= RELATÓRIO FINAL =========")
    print(f"Melhor modelo: {best_name}")
    print(f"Métrica de seleção: {args.metric}")
    print("\nMétricas no teste:")
    for k in ("accuracy", "precision", "recall", "f1_score", "roc_auc"):
        v = test_metrics.get(k)
        if isinstance(v, float):
            print(f"  {k:>12}: {v:.4f}")
        else:
            print(f"  {k:>12}: {v}")
    print("\nClassification report (teste):")
    print(test_metrics["classification_report"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
