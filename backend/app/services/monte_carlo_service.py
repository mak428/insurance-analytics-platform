"""Topic 5: Monte Carlo simulation of S = sum X_i, N ~ Poisson(lambda*exposure)."""
import numpy as np
from app.services.severity_service import DISTRIBUTIONS


def _sum_by_trial(severities: np.ndarray, counts: np.ndarray) -> np.ndarray:
    totals = np.zeros(len(counts))
    if len(severities) > 0:
        np.add.at(totals, np.repeat(np.arange(len(counts)), counts), severities)
    return totals


def simulate_aggregate_loss(lam: float, exposure: float, severity_dist: str, scipy_params: list,
                            n_trials: int = 10_000, seed: int = 42, batch_size: int = 500) -> dict:
    """Batched so a large portfolio (~30k claims/yr) never needs all draws in memory at once."""
    rng = np.random.default_rng(seed)
    dist = DISTRIBUTIONS[severity_dist]
    mean_claims = lam * exposure
    agg = np.empty(n_trials)
    for start in range(0, n_trials, batch_size):
        b = min(batch_size, n_trials - start)
        counts = rng.poisson(mean_claims, size=b)
        sev = dist.rvs(*scipy_params, size=int(counts.sum()), random_state=rng)
        agg[start:start + b] = _sum_by_trial(sev, counts)

    var95, var99 = np.quantile(agg, [0.95, 0.99])
    hist, edges = np.histogram(agg, bins=60)
    return {
        "n_trials": n_trials,
        "mean": float(agg.mean()),
        "std": float(agg.std()),
        "var_95": float(var95),
        "var_99": float(var99),
        "tvar_95": float(agg[agg >= var95].mean()),
        "tvar_99": float(agg[agg >= var99].mean()),
        "histogram": {"counts": hist.tolist(), "bin_edges": edges.tolist()},
        "percentiles": {str(p): float(np.percentile(agg, p)) for p in [1, 5, 25, 50, 75, 95, 99]},
    }
