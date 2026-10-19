"""
Topics 2 (Continuous Distributions) + 3 (MLE) for SEVERITY.
Fits Gamma, Lognormal, Pareto (Lomax) and Weibull by maximum likelihood.
"""
from dataclasses import dataclass, asdict
import numpy as np
from scipy import stats

DISTRIBUTIONS = {
    "gamma": stats.gamma,
    "lognormal": stats.lognorm,
    # scipy's `pareto` is Type I (needs a positive threshold, degenerates with floc=0).
    # The actuarial "Pareto severity" supported on [0, inf) is Lomax (Pareto Type II).
    "pareto": stats.lomax,
    "weibull": stats.weibull_min,
}


@dataclass
class FitResult:
    distribution: str
    params: dict
    scipy_params: list
    log_likelihood: float
    aic: float
    bic: float
    ks_statistic: float
    ks_pvalue: float


def _human_params(name: str, p: tuple) -> dict:
    shape, loc, scale = p
    if name == "gamma":
        return {"shape_alpha": float(shape), "loc": loc, "scale_beta": float(scale)}
    if name == "lognormal":
        return {"sigma": float(shape), "loc": loc, "mu": float(np.log(scale))}
    if name == "pareto":
        return {"shape_alpha": float(shape), "loc": loc, "scale_lambda": float(scale)}
    if name == "weibull":
        return {"shape_k": float(shape), "loc": loc, "scale_lambda": float(scale)}
    raise ValueError(name)


def fit_one(name: str, data: np.ndarray) -> FitResult:
    """MLE fit with loc fixed at 0 (standard for severity; avoids unstable 3-param fits)."""
    dist = DISTRIBUTIONS[name]
    params = dist.fit(data, floc=0)
    log_lik = float(np.sum(dist.logpdf(data, *params)))
    k = len(params) - 1  # loc is fixed
    n = len(data)
    ks_stat, ks_p = stats.kstest(data, dist.name, args=params)
    return FitResult(
        distribution=name,
        params=_human_params(name, params),
        scipy_params=[float(x) for x in params],
        log_likelihood=log_lik,
        aic=2 * k - 2 * log_lik,
        bic=k * np.log(n) - 2 * log_lik,
        ks_statistic=float(ks_stat),
        ks_pvalue=float(ks_p),
    )


def fit_all(data: np.ndarray) -> dict:
    results = {name: fit_one(name, data) for name in DISTRIBUTIONS}
    ranked = sorted(results.values(), key=lambda r: r.aic)
    return {
        "n_observations": len(data),
        "sample_mean": float(np.mean(data)),
        "sample_std": float(np.std(data, ddof=1)),
        "sample_skewness": float(stats.skew(data)),
        "fits": [asdict(r) for r in ranked],
        "best_fit": ranked[0].distribution,
    }


def tail_comparison(data: np.ndarray, fits: dict, quantiles=(0.5, 0.9, 0.95, 0.99, 0.995)) -> dict:
    """Empirical vs fitted quantiles: shows which family tracks the tail."""
    out = {"quantiles": list(quantiles), "empirical": np.quantile(data, quantiles).tolist()}
    for fit in fits["fits"]:
        dist = DISTRIBUTIONS[fit["distribution"]]
        out[fit["distribution"]] = dist.ppf(quantiles, *fit["scipy_params"]).tolist()
    return out
