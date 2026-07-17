"""Joint Model-and-Data Sparsification via the Marginal Likelihood (arXiv
2605.29908, vMcu1h3fOV).

Sparse Bayesian Learning (SBL / ARD) for y = Φ θ with
    θ ~ N(0, Γ^{-1}),  Γ = diag(γ_j)        (feature precisions -> model sparsity)
    y | θ ~ N(Φθ, Λ^{-1}),  Λ = diag(λ_i)   (per-SAMPLE precisions -> data sparsity)

This is the standard SBL framework (Tipping 2001) extended with HETEROSCEDASTIC
per-sample noise λ (the paper's joint model+data ARD).  All quantities are
closed-form under Gaussian conjugacy:

  * marginal likelihood:   y ~ N(0, C),   C = Φ Γ^{-1} Φ^T + Λ^{-1}.
  * weight posterior:      θ | y ~ N(μ, Σ),  Σ^{-1} = Γ + Φ^T Λ Φ,  μ = Σ Φ^T Λ y.
  * EM/MacKay re-estimation (closed-form M-steps):
        γ_j^{new} = 1 / (Σ_jj + μ_j^2)                      (feature relevance)
        λ_i^{new} = 1 / ((y_i − Φ_i μ)^2 + Φ_i Σ Φ_i^T)     (sample relevance)
    each an ascent step on the marginal likelihood.
"""
from __future__ import annotations
import numpy as np


def marginal_loglik(Phi, y, gamma, lam):
    """log p(y | γ, λ) = Gaussian log-pdf of N(0, C), C = Φ Γ⁻¹ Φᵀ + Λ⁻¹."""
    n = len(y)
    C = Phi @ np.diag(1.0 / gamma) @ Phi.T + np.diag(1.0 / lam)
    sign, logdet = np.linalg.slogdet(C)
    Cinv_y = np.linalg.solve(C, y)
    return -0.5 * (n * np.log(2 * np.pi) + logdet + y @ Cinv_y)


def posterior(Phi, y, gamma, lam):
    """θ | y,γ,λ ~ N(μ, Σ):  Σ⁻¹ = Γ + ΦᵀΛΦ,  μ = Σ ΦᵀΛ y."""
    LamPhi = lam[:, None] * Phi                       # Λ Φ  (row-scaled)
    Prec = np.diag(gamma) + Phi.T @ LamPhi            # Γ + Φᵀ Λ Φ
    Sigma = np.linalg.inv(Prec)
    mu = Sigma @ (Phi.T @ (lam * y))                  # Φᵀ Λ y
    return mu, Sigma


def em_update(Phi, y, gamma, lam):
    """One closed-form EM/MacKay re-estimation step for (γ, λ)."""
    mu, Sigma = posterior(Phi, y, gamma, lam)
    gamma_new = 1.0 / (np.diag(Sigma) + mu ** 2)
    resid = y - Phi @ mu
    var_resid = resid ** 2 + np.einsum("ij,jk,ik->i", Phi, Sigma, Phi)  # Φ_i Σ Φ_iᵀ
    lam_new = 1.0 / var_resid
    return gamma_new, lam_new


def fit_joint(Phi, y, n_iter=300, tol=1e-7, feature_only=False, seed=0):
    """Run EM to convergence.  feature_only=True fixes λ=σ⁻² (homoscedastic) to
    reproduce the standard model-only ARD baseline."""
    d = Phi.shape[1]; n = len(y)
    rng = np.random.default_rng(seed)
    gamma = rng.uniform(0.5, 2.0, d)
    lam = np.ones(n) if feature_only else rng.uniform(0.5, 2.0, n)
    ll = marginal_loglik(Phi, y, gamma, lam)
    traj = [ll]
    for _ in range(n_iter):
        g_new, l_new = em_update(Phi, y, gamma, lam)
        gamma = np.clip(g_new, 1e-8, 1e8)
        if not feature_only:
            lam = np.clip(l_new, 1e-8, 1e8)
        ll_new = marginal_loglik(Phi, y, gamma, lam)
        traj.append(ll_new)
        if abs(ll_new - ll) < tol:
            break
        ll = ll_new
    return gamma, lam, traj


def predict(Phi_train, y, Phi_test, gamma, lam):
    """Posterior predictive mean for test features."""
    mu, _ = posterior(Phi_train, y, gamma, lam)
    return Phi_test @ mu
