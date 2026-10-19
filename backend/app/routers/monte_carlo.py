from fastapi import APIRouter
from app.models.schemas import MonteCarloRequest
from app.routers._shared import frequency_and_severity
from app.services.monte_carlo_service import simulate_aggregate_loss

router = APIRouter(prefix="/api/monte-carlo", tags=["monte carlo (topic 5)"])


@router.post("/simulate")
def simulate(req: MonteCarloRequest):
    _, pf, exposure, chosen = frequency_and_severity(req.source, req.severity_distribution)
    out = simulate_aggregate_loss(pf.lambda_hat, exposure, chosen["distribution"],
                                  chosen["scipy_params"], req.n_trials, req.seed)
    return out | {"severity_distribution": chosen["distribution"], "lambda_hat": pf.lambda_hat}
