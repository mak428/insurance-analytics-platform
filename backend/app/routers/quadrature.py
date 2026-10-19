from fastapi import APIRouter, HTTPException
from app.models.schemas import QuadratureRequest
from app.services.quadrature_service import convergence_study, limited_expected_value

router = APIRouter(prefix="/api/quadrature", tags=["quadrature (topic 7)"])


@router.post("/limited-expected-value")
def lev(req: QuadratureRequest):
    try:
        return limited_expected_value(req.distribution, req.params, req.limit)
    except KeyError:
        raise HTTPException(status_code=400, detail=f"Unknown distribution {req.distribution}")


@router.post("/convergence")
def convergence(req: QuadratureRequest):
    try:
        return convergence_study(req.distribution, req.params, req.limit)
    except KeyError:
        raise HTTPException(status_code=400, detail=f"Unknown distribution {req.distribution}")
