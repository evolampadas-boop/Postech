"""Avalia o melhor modelo já salvo. Útil pra reproduzir as métricas sem retreinar."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_loader import load_data
from src.evaluate import evaluate_model
from src.models import load_model
from src.preprocessing import split_data


def main():
    df = load_data(save_csv=False)
    _, _, X_test, _, _, y_test = split_data(df)
    pipeline = load_model("best_model")
    metrics = evaluate_model(pipeline, X_test, y_test, model_name="best_model")

    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1:        {metrics['f1_score']:.4f}")
    print(f"ROC AUC:   {metrics['roc_auc']}")
    print()
    print(metrics["classification_report"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
