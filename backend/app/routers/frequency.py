from dataclasses import asdict
from fastapi import APIRouter
from app.data_loader import load_liability_claims
from app.models.schemas import CompoundPoissonRequest
from app.routers._shared import frequency_and_severity
from app.services.frequency_service import compound_poisson_moments, fit_negbin_mle, fit_poisson_mle
from app.services.severity_service import DISTRIBUTIONS

router = APIRouter(prefix="/api/frequency", tags=["frequency & compound Poisson (topics 1, 3)"])


@router.post("/fit")
def fit_frequency():
    """MLE for claim frequency: Poisson vs Negative Binomial (AIC)."""
    df = load_liability_claims()
    n, e = df["ClaimNb"].to_numpy(float), df["Exposure"].to_numpy(float)
    p, nb = fit_poisson_mle(n, e), fit_negbin_mle(n, e)
    p_aic = 2 - 2 * p.log_likelihood
    return {"poisson": asdict(p) | {"aic": p_aic}, "negative_binomial": asdict(nb),
            "recommendation": "poisson" if p_aic <= nb.aic else "negative_binomial"}


@router.post("/compound-poisson")
def compound_poisson(req: CompoundPoissonRequest):
    """Closed-form moments of annual aggregate loss S under a compound Poisson model."""
    df, pf, exposure, chosen = frequency_and_severity(req.source, req.severity_distribution)
    dist = DISTRIBUTIONS[chosen["distribution"]]
    m1, m2 = float(dist.mean(*chosen["scipy_params"])), float(dist.moment(2, *chosen["scipy_params"]))
    return {"lambda_hat": pf.lambda_hat, "total_exposure": exposure,
            "severity_distribution": chosen["distribution"], "severity_params": chosen["scipy_params"],
            "severity_mean": m1, "severity_second_moment": m2,
            "moments": compound_poisson_moments(pf.lambda_hat, exposure, m1, m2),
            "empirical_total_claim_amount": float(df["ClaimAmount"].sum())}
