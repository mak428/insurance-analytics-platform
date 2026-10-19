"""
Topic 6: Numerical integration for stop-loss premiums.
E[(S-d)+] = integral_d^inf (1 - F_S(x)) dx.
Route 1: Panjer recursion (exact up to lattice width). Route 2: adaptive quadrature
on a moment-matched approximation. Cross-check against Monte Carlo.
"""
import numpy as np
from scipy import integrate, stats
from app.services.severity_service import DISTRIBUTIONS


def discretize_severity(dist_name: str, scipy_params: list, h: float, n_points: int) -> np.ndarray:
    """Method of rounding: mass at k*h = P(k*h - h/2 < X <= k*h + h/2)."""
    dist = DISTRIBUTIONS[dist_name]
    edges = (np.arange(n_points + 1) - 0.5) * h
    edges[0] = 0
    probs = np.diff(dist.cdf(edges, *scipy_params))
    probs[-1] += max(1.0 - probs.sum(), 0.0)  # tail beyond the lattice collapses into last bucket
    return probs


def panjer_recursion(expected_claims: float, severity_probs: np.ndarray, extend: int = 6) -> np.ndarray:
    """Compound Poisson (a=0, b=lambda). Returns aggregate pmf on the same lattice width."""
    m = len(severity_probs)
    n = m * max(extend, 3)
    f = np.zeros(n)
    f[:m] = severity_probs
    g = np.zeros(n)
    g[0] = np.exp(expected_claims * (f[0] - 1))
    for k in range(1, n):
        j = np.arange(1, min(k, m - 1) + 1)
        g[k] = np.sum(expected_claims * j / k * f[j] * g[k - j])
    return g / g.sum()


def stop_loss_from_aggregate_pmf(g: np.ndarray, h: float, retention: float) -> float:
    levels = np.arange(len(g)) * h
    return float(np.sum(g * np.clip(levels - retention, 0, None)))


def stop_loss_via_quadrature(mean: float, std: float, retention: float, distribution: str = "lognormal") -> dict:
    """Integrate the survival function of a moment-matched lognormal/normal with scipy quad."""
    if distribution == "lognormal":
        s2 = np.log(1 + (std / mean) ** 2)
        dist = stats.lognorm(s=np.sqrt(s2), scale=np.exp(np.log(mean) - s2 / 2))
    else:
        dist = stats.norm(mean, std)
    val, err = integrate.quad(lambda x: 1 - dist.cdf(x), retention, mean + 15 * std, limit=200)
    return {"stop_loss_premium": float(val), "quad_abs_error": float(err),
            "approximating_distribution": distribution}
