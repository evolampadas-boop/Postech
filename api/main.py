"""API FastAPI que serve as predições do melhor modelo."""

import logging
import time

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.predictor import Predictor
from api.schemas import (
    DISCLAIMER,
    HealthResponse,
    ModelInfoResponse,
    PatientInput,
    PredictionResponse,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("api")

APP_NAME = "Sistema de Suporte ao Diagnóstico Médico"
APP_VERSION = "1.0.0"

app = FastAPI(
    title=APP_NAME,
    description="API de apoio ao diagnóstico de câncer de mama (Tech Challenge Fase 1 - FIAP PosTech).",
    version=APP_VERSION,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Carrego o Predictor sob demanda pra facilitar os testes (assim a fixture
# de teste consegue injetar um Predictor próprio).
_predictor = None


def get_predictor():
    global _predictor
    if _predictor is None:
        try:
            _predictor = Predictor()
        except Exception as exc:
            logger.exception("Falha ao inicializar Predictor")
            raise HTTPException(
                status_code=503,
                detail="Modelo não disponível. Rode `python scripts/train.py` antes.",
            ) from exc
    return _predictor


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    elapsed = (time.time() - start) * 1000
    logger.info("%s %s -> %d (%.1f ms)", request.method, request.url.path, response.status_code, elapsed)
    return response


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    return JSONResponse(status_code=422, content={"detail": str(exc), "disclaimer": DISCLAIMER})


@app.get("/", response_model=HealthResponse, tags=["health"])
def health_check():
    return HealthResponse(name=APP_NAME, version=APP_VERSION, status="ok")


@app.get("/model/info", response_model=ModelInfoResponse, tags=["model"])
def model_info():
    """Nome do modelo, features esperadas e métricas do treino."""
    return get_predictor().get_model_info()


@app.post("/predict", response_model=PredictionResponse, tags=["predict"])
def predict(patient: PatientInput):
    """Recebe os 30 valores e devolve o diagnóstico estimado."""
    predictor = get_predictor()
    try:
        return predictor.predict(patient.to_feature_dict())
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Erro inesperado na predição")
        raise HTTPException(status_code=500, detail="Erro interno ao processar a predição.") from exc
