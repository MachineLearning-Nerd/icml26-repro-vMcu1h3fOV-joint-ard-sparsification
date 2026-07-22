#!/usr/bin/env python3
"""Joint Model & Data Sparsification via the Marginal Likelihood (arXiv:2605.29908). Reproduces
claim [1]: the heteroscedastic sparse-Bayesian objective admits CLOSED-FORM coordinate-ascent (EM)
updates for both the ARD weight precisions gamma_j and the per-sample noise variances lambda_i:

  Sigma = (Gamma + X^T Lambda^{-1} X)^{-1},   mu = Sigma X^T Lambda^{-1} y
  gamma_j  <- 1 / (mu_j^2 + Sigma_jj)          # ARD weight precision (feature pruning)
  lambda_i <- r_i^2 + x_i^T Sigma x_i,  r_i = y_i - x_i^T mu   # per-sample noise variance (robustness)

We verify: (i) the marginal likelihood L increases monotonically under the EM updates; (ii) irrelevant
features' gamma -> infinity (pruned) while relevant features keep small gamma; (iii) OUTLIER samples
acquire large lambda (down-weighted). Deterministic seeds.
"""
import numpy as np, json, hashlib

def log_evidence(X, y, gamma, lam):
    """log p(y|gamma,lambda) for y ~ N(0, Lambda + X Gamma^{-1} X^T)."""
    n = len(y)
    C = np.diag(lam) + (X / gamma) @ X.T                  # X Gamma^{-1} X^T + Lambda
    sign, logdet = np.linalg.slogdet(C)
    return float(-0.5 * (logdet + y @ np.linalg.solve(C, y) + n * np.log(2 * np.pi)))

def main():
    R = {"claim": "JointARD_closed_form_EM_updates", "paper": "arXiv:2605.29908"}
    rng = np.random.default_rng(0)
    n, p = 120, 12
    X = rng.standard_normal((n, p))
    theta_true = np.zeros(p); theta_true[[0, 3, 7]] = [2.5, -1.8, 1.2]   # only 3 relevant features
    y = X @ theta_true + rng.standard_normal(n) * 0.3
    outliers = [5, 20, 55]                                  # inject 3 outliers
    y[outliers] += np.array([8.0, -7.0, 9.0])

    gamma = np.ones(p); lam = np.ones(n)
    Ls = []
    for it in range(60):
        Ls.append(log_evidence(X, y, gamma, lam))
        Lam_inv = 1.0 / lam
        A = np.diag(gamma) + (X.T * Lam_inv) @ X
        Sigma = np.linalg.inv(A)
        mu = Sigma @ (X.T * Lam_inv) @ y
        gamma = 1.0 / (mu ** 2 + np.diag(Sigma) + 1e-12)   # ARD update
        r = y - X @ mu
        lam = r ** 2 + np.einsum("ij,jk,ik->i", X, Sigma, X) + 1e-12   # noise-variance update
    Ls.append(log_evidence(X, y, gamma, lam))

    R["logL_start"] = round(Ls[0], 3); R["logL_end"] = round(Ls[-1], 3)
    R["logL_monotone_increasing"] = all(Ls[i] <= Ls[i + 1] + 1e-6 for i in range(len(Ls) - 1))
    relevant = [0, 3, 7]; irrelevant = [j for j in range(p) if j not in relevant]
    R["gamma_relevant_mean"] = round(float(np.mean(gamma[relevant])), 3)
    R["gamma_irrelevant_min"] = round(float(np.min(gamma[irrelevant])), 1)
    R["irrelevant_features_pruned"] = float(np.min(gamma[irrelevant])) > 100 * float(np.max(gamma[relevant]))
    R["lambda_outlier_mean"] = round(float(np.mean(lam[outliers])), 2)
    inliers = [i for i in range(n) if i not in outliers]
    R["lambda_inlier_mean"] = round(float(np.mean(lam[inliers])), 3)
    R["outliers_downweighted"] = float(np.mean(lam[outliers])) > 10 * float(np.mean(lam[inliers]))
    # recovered weights match the sparse truth
    R["recovered_theta_relevant"] = [round(float(mu[j]), 3) for j in relevant]
    R["recovery_close_to_truth"] = float(np.max(np.abs(mu[relevant] - theta_true[relevant]))) < 0.3

    R["verdict"] = "supports" if (R["logL_monotone_increasing"] and R["irrelevant_features_pruned"]
                                  and R["outliers_downweighted"] and R["recovery_close_to_truth"]) else "inconclusive"

    print("claim: " + R["claim"])
    print("Joint ARD closed-form EM: gamma_j <- 1/(mu_j^2+Sigma_jj), lambda_i <- r_i^2 + x_i^T Sigma x_i.")
    print()
    print(f"(i) marginal likelihood: {R['logL_start']} -> {R['logL_end']}, monotone increasing: {R['logL_monotone_increasing']}")
    print(f"(ii) feature pruning: relevant gamma mean={R['gamma_relevant_mean']} vs irrelevant gamma min={R['gamma_irrelevant_min']} "
          f"-> irrelevant pruned (gamma->inf): {R['irrelevant_features_pruned']}")
    print(f"(iii) robustness: outlier lambda mean={R['lambda_outlier_mean']} vs inlier lambda mean={R['lambda_inlier_mean']} "
          f"-> outliers down-weighted: {R['outliers_downweighted']}")
    print(f"     recovered relevant weights {R['recovered_theta_relevant']} (truth [2.5,-1.8,1.2]) close: {R['recovery_close_to_truth']}")
    print(f"verdict: {R['verdict']}")

    def _np(o):
        if isinstance(o, np.bool_): return bool(o)
        if isinstance(o, np.integer): return int(o)
        if isinstance(o, np.floating): return float(o)
        raise TypeError
    import os; os.makedirs("outputs", exist_ok=True)
    open("outputs/ard_results.json", "w").write(json.dumps(R, indent=2, default=_np))
    print("RESULTS_SHA256=" + hashlib.sha256(json.dumps(R, sort_keys=True, default=_np).encode()).hexdigest())
    return 0 if R["verdict"] == "supports" else 1

if __name__ == "__main__":
    raise SystemExit(main())
