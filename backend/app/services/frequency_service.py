"""
Topic 3 (MLE) for FREQUENCY and Topic 1 (Compound Poisson) aggregate-loss moments.
N_i ~ Poisson(lambda * Exposure_i), or NegBin if overdispersed.
"""
from dataclasses import dataclass
import numpy as np
from scipy import stats
from scipy.optimize import minimize


@dataclass
class PoissonFitResult:
    lambda_hat: float
    log_likelihood: float
    standard_error: float
    dispersion_ratio: float  # ~1 Poisson ok, >1 overdispersed, <1 underdispersed


def fit_poisson_mle(claim_counts: np.ndarray, exposures: np.ndarray) -> PoissonFitResult:
    """Closed-form MLE lambda = sum(n)/sum(e), verified against numerical optimization."""
    n, e = claim_counts, exposures
    closed = float(n.sum() / e.sum())

    def nll(x):
        lam = np.exp(x[0])
        return -np.sum(n * np.log(lam * e + 1e-300) - lam * e)

    opt = minimize(nll, x0=[np.log(closed)], method="BFGS")
    assert abs(np.exp(opt.x[0]) - closed) / closed < 1e-3, "numeric MLE disagrees with closed form"

    se = float(np.sqrt(closed / e.sum()))  # inverse Fisher information
    ll = float(np.sum(stats.poisson.logpmf(n, mu=closed * e)))
    expected = closed * e
    disp = float(np.sum((n - expected) ** 2 / expected) / (len(n) - 1))  # Pearson dispersion
    return PoissonFitResult(closed, ll, se, disp)


@dataclass
class NegBinFitResult:
    r_hat: float
    mean_rate: float
    log_likelihood: float
    aic: float


def fit_negbin_mle(claim_counts: np.ndarray, exposures: np.ndarray) -> NegBinFitResult:
    """NegBin with mean = rate*exposure and size r, fit by Nelder-Mead."""
    n, e = claim_counts, exposures

    def nll(params):
        r, rate = np.exp(params[0]), np.exp(params[1])
        mu = rate * e
        return -np.sum(stats.nbinom.logpmf(n, r, r / (r + mu)))

    opt = minimize(nll, x0=[np.log(5.0), np.log(n.sum() / e.sum())], method="Nelder-Mead")
    ll = -float(opt.fun)
    return NegBinFitResult(float(np.exp(opt.x[0])), float(np.exp(opt.x[1])), ll, 4 - 2 * ll)


def compound_poisson_moments(lam: float, total_exposure: float, sev_mean: float, sev_m2: float) -> dict:
    """E[S] = lam*e*E[X];  Var(S) = lam*e*E[X^2]."""
    k = lam * total_exposure
    es, vs = k * sev_mean, k * sev_m2
    return {
        "expected_claim_count": k,
        "expected_aggregate_loss": es,
        "variance_aggregate_loss": vs,
        "std_aggregate_loss": float(np.sqrt(vs)),
        "coefficient_of_variation": float(np.sqrt(vs) / es) if es > 0 else None,
    }
