from fastapi import APIRouter
from app.data_loader import load_health_panel
from app.models.schemas import CredibilityRequest
from app.services.credibility_service import fit_buhlmann_straub, validate_out_of_sample

router = APIRouter(prefix="/api/credibility", tags=["credibility (topic 4)"])


@router.post("/buhlmann-straub")
def buhlmann_straub(req: CredibilityRequest):
    df = load_health_panel()
    fit = fit_buhlmann_straub(df, req.entity_col, req.exposure_col, req.loss_col)
    fit["entities"] = fit["entities"][: req.max_entities_returned]
    return fit


@router.post("/validate")
def validate(req: CredibilityRequest):
    return validate_out_of_sample(load_health_panel(), req.entity_col, req.exposure_col,
                                  req.loss_col, req.period_col)
