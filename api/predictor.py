"""Classe Predictor - carrega o best_model salvo e faz as predições."""

import json
import logging
from pathlib import Path

import pandas as pd

from api.schemas import DISCLAIMER
from src.data_loader import get_feature_names
from src.models import load_model

logger = logging.getLogger(__name__)

DEFAULT_MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


class Predictor:
    def __init__(self, models_dir=DEFAULT_MODELS_DIR):
        self.models_dir = Path(models_dir)
        self.feature_names = get_feature_names()
        self.metrics = {}
        self.model_name = "best_model"

        # Tento carregar os resultados do treino pra expor as métricas no /model/info.
        results_path = self.models_dir / "training_results.json"
        if results_path.exists():
            try:
                with results_path.open(encoding="utf-8") as f:
                    payload = json.load(f)
                self.model_name = payload.get("best_model", self.model_name)
                self.metrics = payload.get("test_metrics", {})
            except Exception as exc:
                logger.warning("Falha ao ler training_results.json: %s", exc)

        try:
            self.pipeline = load_model("best_model", path=self.models_dir)
            logger.info("Modelo '%s' carregado de %s", self.model_name, self.models_dir)
        except FileNotFoundError as exc:
            logger.error("Modelo não encontrado. Rode `python scripts/train.py` antes.")
            raise RuntimeError(str(exc)) from exc

    def _build_dataframe(self, patient_data):
        missing = [f for f in self.feature_names if f not in patient_data]
        if missing:
            raise ValueError(f"Features ausentes: {missing}")
        row = {f: float(patient_data[f]) for f in self.feature_names}
        return pd.DataFrame([row], columns=self.feature_names)

    def predict(self, patient_data):
        df = self._build_dataframe(patient_data)
        pred = int(self.pipeline.predict(df)[0])

        try:
            proba = self.pipeline.predict_proba(df)[0]
            class_proba = float(proba[pred])
        except Exception:
            class_proba = 1.0

        confidence_pct = round(class_proba * 100, 2)
        diagnosis = "Maligno" if pred == 1 else "Benigno"

        # Critério de risco que defini: confiança >= 80% é decisivo, entre 60 e 80
        # é médio, abaixo de 60 sugere repetir o exame.
        if confidence_pct < 60:
            risk_level = "Incerto — recomenda-se exames adicionais"
        elif confidence_pct < 80:
            risk_level = "Médio"
        else:
            risk_level = "Alto" if pred == 1 else "Baixo"

        return {
            "diagnosis": diagnosis,
            "diagnosis_code": pred,
            "confidence": confidence_pct,
            "risk_level": risk_level,
            "disclaimer": DISCLAIMER,
            "model_used": self.model_name,
        }

    def get_model_info(self):
        return {
            "model_name": self.model_name,
            "features": self.feature_names,
            "metrics": self.metrics,
        }
