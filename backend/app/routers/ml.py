from fastapi import APIRouter, HTTPException
from app.data_loader import load_generic_insurance
from app.models.schemas import MLPredictRequest
from app.services import ml_service

router = APIRouter(prefix="/api/ml", tags=["machine learning (topic 9)"])
_state: dict = {}  # trained once, cached in memory


def _ensure_trained():
    if "report" not in _state:
        _state["report"], _state["glm"], _state["gbm"] = ml_service.train_and_evaluate(load_generic_insurance())


@router.get("/evaluate")
def evaluate():
    _ensure_trained()
    return _state["report"]


@router.post("/predict-severity")
def predict(req: MLPredictRequest):
    _ensure_trained()
    if req.model not in ("gbm", "glm"):
        raise HTTPException(status_code=400, detail="model must be 'gbm' or 'glm'")
    rec = req.model_dump(exclude={"model"})
    return {"model": req.model,
            "predicted_severity": ml_service.predict_single(_state[req.model], rec, log_target=(req.model == "gbm"))}
