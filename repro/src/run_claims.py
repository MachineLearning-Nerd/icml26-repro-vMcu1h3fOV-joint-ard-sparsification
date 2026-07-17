"""Evidence orchestrator: Joint Model-and-Data Sparsification via the Marginal
Likelihood (arXiv 2605.29908, vMcu1h3fOV).

C1: joint (per-sample λ) ARD yields sparse + ROBUST prediction vs feature-only.
C2: single marginal-likelihood objective, closed-form Gaussian posterior, and
    EM/MacKay updates that are ascent on the marginal likelihood.
"""
import os, sys, csv, json
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from ard import (marginal_loglik, posterior, em_update, fit_joint, predict)

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "outputs")
os.makedirs(OUT, exist_ok=True)


def make_problem(seed, n_outliers=8):
    rng = np.random.default_rng(seed)
    n, d = 80, 8
    true_w = np.array([2.0, -1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # 2 relevant feats
    X = rng.normal(0, 1, (n, d))
    y = X @ true_w + 0.3 * rng.normal(0, 1, n)
    outlier_idx = np.arange(n_outliers)
    y[outlier_idx] += 15.0 * rng.normal(0, 1, n_outliers)         # contaminations
    Xt = rng.normal(0, 1, (50, d))
    yt = Xt @ true_w
    return X, y, outlier_idx, Xt, yt, true_w


def claim2_closed_form_and_ascent():
    """C2: closed-form posterior/marginal-likelihood + EM is monotone ascent."""
    rows = []; ascent_ok = True; posterior_ok = True
    for seed in range(8):
        X, y, oi, Xt, yt, tw = make_problem(seed)
        gj, lj, traj = fit_joint(X, y, feature_only=False, seed=seed)
        mono = all(traj[i + 1] >= traj[i] - 1e-6 for i in range(len(traj) - 1))
        ascent_ok = ascent_ok and mono
        # closed-form posterior matches the definition (C = ΦΓ⁻¹Φᵀ+Λ⁻¹; μ=ΣΦᵀΛy)
        mu, Sig = posterior(X, y, gj, lj)
        Prec = np.diag(gj) + X.T @ (lj[:, None] * X)
        post_ok = np.max(np.abs(mu - np.linalg.solve(Prec, X.T @ (lj * y)))) < 1e-6
        posterior_ok = posterior_ok and post_ok
        rows.append({"seed": seed, "EM_ascent": mono, "posterior_matches_def": post_ok,
                     "final_loglik": traj[-1], "n_EM_iters": len(traj) - 1})
    with open(os.path.join(OUT, "c2_closed_form_ascent.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    return {"claim": "C2 closed-form + EM ascent", "cases": len(rows),
            "all_EM_ascent": ascent_ok, "all_posterior_closed_form": posterior_ok}


def claim1_sparse_and_robust():
    """C1: joint ARD recovers feature sparsity AND is robust to outliers
    (down-weights them via small λ_i), beating feature-only ARD in test RMSE."""
    rows = []; joint_wins = 0; sparsity_wins = 0; downweight_wins = 0; total = 0
    for seed in range(10):
        X, y, oi, Xt, yt, tw = make_problem(seed)
        gj, lj, _ = fit_joint(X, y, feature_only=False, seed=seed)
        gf, lf, _ = fit_joint(X, y, feature_only=True, seed=seed)
        # robustness: test RMSE
        rmse_j = float(np.sqrt(np.mean((predict(X, y, Xt, gj, lj) - yt) ** 2)))
        rmse_f = float(np.sqrt(np.mean((predict(X, y, Xt, gf, lf) - yt) ** 2)))
        joint_wins += int(rmse_j < rmse_f)
        # sparsity: irrelevant features pruned (1/γ small for true-zero weights)
        irr = 1.0 / gj[np.where(np.abs(tw) < 1e-9)[0]]
        sparsity_wins += int(np.max(irr) < 0.05)
        # outlier down-weighting: λ on outliers < λ on clean
        downweight_wins += int(np.mean(lj[oi]) < np.mean(np.delete(lj, oi)))
        total += 1
        rows.append({"seed": seed, "rmse_joint": rmse_j, "rmse_feat_only": rmse_f,
                     "max_irrelevant_1overGamma": float(np.max(irr)),
                     "lam_outlier_mean": float(np.mean(lj[oi])),
                     "lam_clean_mean": float(np.mean(np.delete(lj, oi)))})
    with open(os.path.join(OUT, "c1_sparse_robust.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    return {"claim": "C1 sparse + robust", "trials": total,
            "joint_lower_rmse_fraction": joint_wins / total,
            "sparsity_recovered_fraction": sparsity_wins / total,
            "outliers_downweighted_fraction": downweight_wins / total,
            "joint_robust_and_sparse": joint_wins / total >= 0.8 and sparsity_wins / total >= 0.7}


def main():
    print("=== C2 ==="); r2 = claim2_closed_form_and_ascent(); print(json.dumps(r2, indent=2, default=float))
    print("=== C1 ==="); r1 = claim1_sparse_and_robust(); print(json.dumps(r1, indent=2, default=float))
    overall = {
        "paper": "Joint Model and Data Sparsification via Marginal Likelihood (arXiv 2605.29908)",
        "claims": {"C1_sparse_robust": r1, "C2_closed_form_ascent": r2},
        "verdict": {"C1_verified": r1["joint_robust_and_sparse"],
                    "C2_verified": r2["all_EM_ascent"] and r2["all_posterior_closed_form"]},
    }
    json.dump(overall, open(os.path.join(OUT, "summary.json"), "w"), indent=2, default=float)
    print("\nWrote", ", ".join(sorted(os.listdir(OUT))))


if __name__ == "__main__":
    main()
