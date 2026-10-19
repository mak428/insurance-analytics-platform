import numpy as np
from fastapi import APIRouter, HTTPException
from app.models.schemas import StopLossRequest
from app.services.stop_loss_service import (discretize_severity, panjer_recursion,
                                            stop_loss_from_aggregate_pmf, stop_loss_via_quadrature)

router = APIRouter(prefix="/api/stop-loss", tags=["stop-loss (topic 6)"])


@router.post("/premium")
def stop_loss_premium(req: StopLossRequest):
    """E[(S-d)+] via Panjer recursion and via quadrature on a moment-matched approximation."""
    try:
        probs = discretize_severity(req.severity_distribution, req.severity_params,
                                    req.lattice_width, req.lattice_points)
    except KeyError:
        raise HTTPException(status_code=400, detail=f"Unknown distribution {req.severity_distribution}")
    g = panjer_recursion(req.lam * req.exposure, probs)
    levels = np.arange(len(g)) * req.lattice_width
    mean = float(np.sum(levels * g))
    std = float(np.sqrt(np.sum(g * (levels - mean) ** 2)))
    return {"retention": req.retention,
            "panjer": {"stop_loss_premium": stop_loss_from_aggregate_pmf(g, req.lattice_width, req.retention),
                       "implied_aggregate_mean": mean, "implied_aggregate_std": std},
            "quadrature_on_moment_matched_approx": stop_loss_via_quadrature(mean, std, req.retention)}
