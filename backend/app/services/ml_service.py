"""
Topic 9: ML claim severity. Gamma GLM (interpretable baseline) vs gradient boosting on
log-severity, evaluated with RMSE/MAE and the actuarial Gini / Lorenz curve.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import TweedieRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

CATEGORICAL = ["Gender", "Marital_Status", "Education", "Occupation"]
NUMERIC = ["Age", "Income"]
TARGET = "Claim_Amount"
_trapz = getattr(np, "trapezoid", None) or np.trapz  # numpy 1.x / 2.x compatibility


def _pipeline(model) -> Pipeline:
    prep = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
                              ("num", "passthrough", NUMERIC)])
    return Pipeline([("prep", prep), ("model", model)])


def _lorenz(y_true, y_pred):
    y = y_true[np.argsort(y_pred)]
    return np.insert(np.cumsum(y) / y.sum(), 0, 0)


def gini_index(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Normalized Gini: area between Lorenz curve (sorted by prediction) and diagonal,
    divided by the perfect-ranking value. 0 = no discrimination, 1 = perfect ranking."""
    x = np.linspace(0, 1, len(y_true) + 1)
    raw = 1 - 2 * _trapz(_lorenz(y_true, y_pred), x)
    best = 1 - 2 * _trapz(_lorenz(y_true, y_true), x)
    return float(raw / best) if best else 0.0


def lorenz_curve(y_true, y_pred, n_points: int = 20) -> dict:
    c = _lorenz(y_true, y_pred)
    idx = np.linspace(0, len(y_true), n_points).astype(int)
    return {"cumulative_share_of_policies": (idx / len(y_true)).tolist(),
            "cumulative_share_of_losses": c[idx].tolist()}


def train_and_evaluate(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42):
    df = df.dropna(subset=CATEGORICAL + NUMERIC + [TARGET])
    df = df[df[TARGET] > 0]
    X, y = df[CATEGORICAL + NUMERIC], df[TARGET].to_numpy()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=seed)

    glm = _pipeline(TweedieRegressor(power=2, link="log", max_iter=2000)).fit(Xtr, ytr)
    gbm = _pipeline(HistGradientBoostingRegressor(max_iter=300, max_depth=4, learning_rate=0.05,
                                                  random_state=seed)).fit(Xtr, np.log(ytr))
    glm_pred, gbm_pred = glm.predict(Xte), np.exp(gbm.predict(Xte))

    def m(p):
        return {"rmse": float(np.sqrt(mean_squared_error(yte, p))),
                "mae": float(mean_absolute_error(yte, p)), "gini": gini_index(yte, p)}

    report = {"n_train": len(Xtr), "n_test": len(Xte), "glm_gamma": m(glm_pred), "gbm": m(gbm_pred),
              "lorenz_curve_gbm": lorenz_curve(yte, gbm_pred), "lorenz_curve_glm": lorenz_curve(yte, glm_pred)}
    return report, glm, gbm


def predict_single(pipeline, record: dict, log_target: bool = False) -> float:
    pred = pipeline.predict(pd.DataFrame([record])[CATEGORICAL + NUMERIC])[0]
    return float(np.exp(pred)) if log_target else float(pred)
