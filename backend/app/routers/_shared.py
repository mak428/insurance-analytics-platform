"""Helpers shared by routers that need 'frequency + best severity fit'."""
from fastapi import HTTPException
from app.data_loader import load_liability_claims, load_severity_sample
from app.services.frequency_service import fit_poisson_mle
from app.services.severity_service import fit_all


def frequency_and_severity(source: str, severity_distribution: str | None):
    df = load_liability_claims()
    n, e = df["ClaimNb"].to_numpy(float), df["Exposure"].to_numpy(float)
    pf = fit_poisson_mle(n, e)
    try:
        fits = fit_all(load_severity_sample(source))
    except ValueError as err:
        raise HTTPException(status_code=400, detail=str(err))
    if severity_distribution:
        chosen = next((f for f in fits["fits"] if f["distribution"] == severity_distribution), None)
        if chosen is None:
            raise HTTPException(status_code=400, detail=f"Unknown distribution {severity_distribution}")
    else:
        chosen = fits["fits"][0]
    return df, pf, float(e.sum()), chosen
