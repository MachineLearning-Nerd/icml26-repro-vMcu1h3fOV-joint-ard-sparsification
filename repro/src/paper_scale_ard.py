"""Numerically stable weight-space EM for paper-scale joint ARD.

Here ``noise_var`` is the paper's per-sample variance lambda.  The older toy
module stores its reciprocal precision; both parameterizations are equivalent,
but the variance form follows Appendix C's notation directly.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.linalg import cho_factor, cho_solve


@dataclass
class SBLFit:
    coef: np.ndarray
    covariance_diag: np.ndarray
    gamma: np.ndarray
    noise_var: np.ndarray
    iterations: int
    converged: bool


def effective_support(relevance: np.ndarray) -> float:
    """Exponentiated Shannon entropy divided by vector length, in [1/n, 1]."""
    values = np.maximum(np.asarray(relevance, float), 1e-300)
    p = values / values.sum()
    return float(np.exp(-np.sum(p * np.log(p))) / len(p))


def fit_sbl(
    X: np.ndarray,
    y: np.ndarray,
    *,
    heteroscedastic: bool,
    max_iter: int = 80,
    warmup: int = 50,
    noise_update_every: int = 3,
    damping: float = 0.35,
    tol: float = 1e-5,
) -> SBLFit:
    """Fit Appendix-C EM with warm-up, damping, clipping, and robust stopping.

    The safeguards implement the numerical recommendations in Appendix D.1,
    including its stated 50--300-step model-only warm-up range.
    No observations or features are pruned, so dimensions and ESS remain fully
    auditable.  ``heteroscedastic=False`` performs the homoscedastic ablation.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, d = X.shape
    if y.shape != (n,) or not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("finite X[n,d] and y[n] required")
    gamma = np.ones(d)
    base_var = max(float(np.var(y)) * 0.1, 1e-3)
    noise_var = np.full(n, base_var)
    patience = 0
    converged = False
    coef = np.zeros(d)
    covariance_diag = np.ones(d)

    for iteration in range(1, max_iter + 1):
        weights = 1.0 / noise_var
        precision = np.diag(gamma) + X.T @ (weights[:, None] * X)
        factor = cho_factor(precision, lower=True, check_finite=False)
        coef = cho_solve(factor, X.T @ (weights * y), check_finite=False)
        covariance = cho_solve(factor, np.eye(d), check_finite=False)
        covariance_diag = np.maximum(np.diag(covariance), 1e-12)

        gamma_target = 1.0 / np.maximum(coef * coef + covariance_diag, 1e-12)
        gamma_target = np.clip(gamma_target, 1e-8, 1e8)
        log_gamma_old = np.log(gamma)
        gamma = np.exp((1.0 - damping) * log_gamma_old + damping * np.log(gamma_target))

        update_noise = not heteroscedastic or (
            iteration > warmup and (iteration - warmup) % noise_update_every == 0
        )
        log_noise_old = np.log(noise_var)
        if update_noise:
            residual = y - X @ coef
            leverage_var = np.einsum("ij,jk,ik->i", X, covariance, X, optimize=True)
            target = np.maximum(residual * residual + leverage_var, 1e-8)
            if not heteroscedastic:
                target = np.full(n, float(np.mean(target)))
            # Appendix D.1 recommends clipping to prevent the n extra noise
            # parameters from explaining away residual structure.  A broad
            # two-decade window keeps real outliers separable without allowing
            # near-zero variances to dominate the entropy-based ESS metric.
            floor = max(float(np.median(target)) * 0.1, 1e-6)
            ceiling = max(float(np.median(target)) * 100.0, 10.0)
            target = np.clip(target, floor, ceiling)
            noise_var = np.exp((1.0 - damping) * log_noise_old + damping * np.log(target))

        delta_gamma = np.max(np.abs(np.log(gamma) - log_gamma_old)) / (
            1.0 + np.max(np.abs(np.log(gamma)))
        )
        delta_noise = np.max(np.abs(np.log(noise_var) - log_noise_old)) / (
            1.0 + np.max(np.abs(np.log(noise_var)))
        )
        if iteration > warmup and max(delta_gamma, delta_noise) < tol:
            patience += 1
            if patience >= 5:
                converged = True
                break
        else:
            patience = 0

    # Recompute the posterior at final hyperparameters, as required by Alg. 1.
    weights = 1.0 / noise_var
    precision = np.diag(gamma) + X.T @ (weights[:, None] * X)
    factor = cho_factor(precision, lower=True, check_finite=False)
    coef = cho_solve(factor, X.T @ (weights * y), check_finite=False)
    covariance_diag = np.diag(cho_solve(factor, np.eye(d), check_finite=False))
    return SBLFit(coef, covariance_diag, gamma, noise_var, iteration, converged)
