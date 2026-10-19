from fastapi import APIRouter
from app.data_loader import load_synthetic_policies
from app.models.schemas import ForecastRequest
from app.services import timeseries_service as ts

router = APIRouter(prefix="/api/timeseries", tags=["time series (topic 8)"])


@router.get("/monthly")
def monthly_series():
    s = ts.build_monthly_series(load_synthetic_policies())
    return {"dates": [d.strftime("%Y-%m-%d") for d in s.index], "values": s.tolist(),
            "stationarity": ts.stationarity_check(s), "decomposition": ts.decompose(s)}


@router.post("/forecast")
def forecast(req: ForecastRequest):
    return ts.forecast(ts.build_monthly_series(load_synthetic_policies()), req.horizon, req.holdout)
