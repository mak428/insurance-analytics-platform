from typing import Optional
from pydantic import BaseModel, Field


class SeverityFitRequest(BaseModel):
    source: str = Field("liability", description="'liability' or 'generic'")


class CompoundPoissonRequest(BaseModel):
    source: str = "liability"
    severity_distribution: Optional[str] = Field(None, description="gamma/lognormal/pareto/weibull; default = best AIC")


class MonteCarloRequest(BaseModel):
    source: str = "liability"
    severity_distribution: Optional[str] = None
    n_trials: int = Field(10_000, ge=100, le=200_000)
    seed: int = 42


class StopLossRequest(BaseModel):
    retention: float = Field(..., gt=0)
    lam: float = Field(..., description="Poisson claims per unit exposure")
    exposure: float = Field(..., gt=0)
    severity_distribution: str = "pareto"
    severity_params: list[float]
    lattice_width: float = Field(50.0, gt=0)
    lattice_points: int = Field(2000, gt=10)


class QuadratureRequest(BaseModel):
    distribution: str
    params: list[float]
    limit: float = Field(..., gt=0)


class CredibilityRequest(BaseModel):
    entity_col: str = "ID"
    exposure_col: str = "exposure_time"
    loss_col: str = "cost_claims_year"
    period_col: str = "period"
    max_entities_returned: int = Field(100, ge=1, le=5000)


class ForecastRequest(BaseModel):
    horizon: int = Field(6, ge=1, le=24)
    holdout: int = Field(6, ge=0, le=12)


class MLPredictRequest(BaseModel):
    Age: float
    Gender: str
    Income: float
    Marital_Status: str
    Education: str
    Occupation: str
    model: str = Field("gbm", description="'gbm' or 'glm'")
