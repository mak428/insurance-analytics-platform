from fastapi import APIRouter, HTTPException
from app.data_loader import load_severity_sample
from app.models.schemas import SeverityFitRequest
from app.services.severity_service import fit_all, tail_comparison

router = APIRouter(prefix="/api/severity", tags=["severity (topics 2, 3)"])


@router.post("/fit")
def fit_severity(req: SeverityFitRequest):
    """MLE-fit Gamma, Lognormal, Pareto (Lomax), Weibull; ranked by AIC."""
    try:
        data = load_severity_sample(req.source)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    result = fit_all(data)
    result["tail_comparison"] = tail_comparison(data, result)
    return result
