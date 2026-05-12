"""Schemas Pydantic da API.

Os campos de PatientInput são as 30 features do Breast Cancer Wisconsin.
Coloquei exemplos pra facilitar testar pelo Swagger.
"""

from pydantic import BaseModel, Field

DISCLAIMER = (
    "Este resultado é uma estimativa estatística produzida por um modelo de Machine Learning "
    "e NÃO substitui a avaliação de um médico qualificado. Sempre consulte um profissional de saúde."
)


class PatientInput(BaseModel):
    # Uso o alias com espaço porque é assim que o sklearn entrega os nomes
    # (e o pipeline interno espera nesse formato).
    mean_radius: float = Field(..., alias="mean radius", example=17.99)
    mean_texture: float = Field(..., alias="mean texture", example=10.38)
    mean_perimeter: float = Field(..., alias="mean perimeter", example=122.8)
    mean_area: float = Field(..., alias="mean area", example=1001.0)
    mean_smoothness: float = Field(..., alias="mean smoothness", example=0.1184)
    mean_compactness: float = Field(..., alias="mean compactness", example=0.2776)
    mean_concavity: float = Field(..., alias="mean concavity", example=0.3001)
    mean_concave_points: float = Field(..., alias="mean concave points", example=0.1471)
    mean_symmetry: float = Field(..., alias="mean symmetry", example=0.2419)
    mean_fractal_dimension: float = Field(..., alias="mean fractal dimension", example=0.07871)

    radius_error: float = Field(..., alias="radius error", example=1.095)
    texture_error: float = Field(..., alias="texture error", example=0.9053)
    perimeter_error: float = Field(..., alias="perimeter error", example=8.589)
    area_error: float = Field(..., alias="area error", example=153.4)
    smoothness_error: float = Field(..., alias="smoothness error", example=0.006399)
    compactness_error: float = Field(..., alias="compactness error", example=0.04904)
    concavity_error: float = Field(..., alias="concavity error", example=0.05373)
    concave_points_error: float = Field(..., alias="concave points error", example=0.01587)
    symmetry_error: float = Field(..., alias="symmetry error", example=0.03003)
    fractal_dimension_error: float = Field(..., alias="fractal dimension error", example=0.006193)

    worst_radius: float = Field(..., alias="worst radius", example=25.38)
    worst_texture: float = Field(..., alias="worst texture", example=17.33)
    worst_perimeter: float = Field(..., alias="worst perimeter", example=184.6)
    worst_area: float = Field(..., alias="worst area", example=2019.0)
    worst_smoothness: float = Field(..., alias="worst smoothness", example=0.1622)
    worst_compactness: float = Field(..., alias="worst compactness", example=0.6656)
    worst_concavity: float = Field(..., alias="worst concavity", example=0.7119)
    worst_concave_points: float = Field(..., alias="worst concave points", example=0.2654)
    worst_symmetry: float = Field(..., alias="worst symmetry", example=0.4601)
    worst_fractal_dimension: float = Field(..., alias="worst fractal dimension", example=0.1189)

    class Config:
        allow_population_by_field_name = True
        schema_extra = {
            "example": {
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
        }

    def to_feature_dict(self):
        return self.dict(by_alias=True)


class PredictionResponse(BaseModel):
    diagnosis: str
    diagnosis_code: int
    confidence: float
    risk_level: str
    disclaimer: str = Field(default=DISCLAIMER)
    model_used: str


class ModelInfoResponse(BaseModel):
    model_name: str
    features: list
    metrics: dict


class HealthResponse(BaseModel):
    name: str
    version: str
    status: str
