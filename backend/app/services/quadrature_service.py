"""
Topic 7: Quadrature for E[g(X)] without closed form, vs Monte Carlo integration.
Limited expected value: E[min(X, L)] = integral_0^L (1 - F(x)) dx.
"""
import numpy as np
from scipy import integrate
from app.services.severity_service import DISTRIBUTIONS


def _gauss(f, limit: float, n: int) -> float:
    nodes, weights = np.polynomial.legendre.leggauss(n)
    x = 0.5 * (nodes + 1) * limit
    return float(np.sum(0.5 * limit * weights * f(x)))


def limited_expected_value(dist_name: str, scipy_params: list, limit: float) -> dict:
    dist = DISTRIBUTIONS[dist_name]
    surv = lambda x: 1 - dist.cdf(x, *scipy_params)
    quad_val, quad_err = integrate.quad(surv, 0, limit, limit=200)
    gl = _gauss(surv, limit, 64)
    return {
        "limit": limit,
        "adaptive_quadrature": {"value": float(quad_val), "abs_error": float(quad_err)},
        "gauss_legendre_64node": {"value": gl, "abs_diff_from_adaptive": abs(gl - quad_val)},
    }


def convergence_study(dist_name: str, scipy_params: list, limit: float,
                      node_counts=(4, 8, 16, 32, 64), mc_sizes=(100, 1_000, 10_000, 100_000),
                      seed: int = 42) -> dict:
    dist = DISTRIBUTIONS[dist_name]
    surv = lambda x: 1 - dist.cdf(x, *scipy_params)
    ref, _ = integrate.quad(surv, 0, limit, limit=200)
    rng = np.random.default_rng(seed)
    return {
        "reference_value": float(ref),
        "quadrature": {"node_counts": list(node_counts),
                       "abs_errors": [abs(_gauss(surv, limit, n) - ref) for n in node_counts]},
        "monte_carlo": {"sample_sizes": list(mc_sizes),
                        "abs_errors": [abs(limit * float(np.mean(surv(rng.uniform(0, limit, n)))) - ref)
                                       for n in mc_sizes]},
    }
