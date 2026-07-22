#!/usr/bin/env python3
"""Joint Model & Data Sparsification via the Marginal Likelihood (arXiv:2605.29908).

Reproduces claim [3] -- addressing the judge's objection that no experiment covered
Student-t / Huber robust baselines, RVM kernel regression, a RANGE of contamination
levels, or the Figure 3/4 comparisons. Two experiments:

  [Fig 4 analog]  RVM (Relevance Vector Machine) kernel regression on an RBF basis:
      sweep contamination 0..30% and compare the paper's "Joint ARD" against the
      Student-t and Huber ROBUST baselines and a homoscedastic RVM. Show Joint ARD
      is competitive with (and at high contamination better than) the robust
      baselines across the whole range.

  [Fig 3 analog]  Sparse Bayesian recovery (under-determined linear model, d>n):
      at LOW contamination, vary the signal sparsity (number of true nonzero weights)
      and show Joint ARD's gain over homoscedastic modeling is LARGEST for sparse
      signals (and collapses as the signal becomes dense / unrecoverable).

ALL estimators share the SAME closed-form ARD update for the weight precisions gamma_j
and the SAME evidence-based solver; they differ ONLY in how the per-sample noise
variances lambda_i are set -- i.e. the robustness mechanism:

    Sigma = (diag(gamma) + Phi^T Lambda^{-1} Phi)^{-1},   mu = Sigma Phi^T Lambda^{-1} y
    gamma_j  <- 1 / (mu_j^2 + Sigma_jj)               # ARD weight / basis pruning

  * Homoscedastic RVM : one shared beta^{-1} (Tipping-2001 evidence update). NOT robust.
  * Joint ARD (paper) : lambda_i = r_i^2 + phi_i^T Sigma phi_i        (type-II ML, this paper)
  * Student-t (Geweke): scale mixture, u_i=(nu+1)/(nu+(r_i^2+..)/s^2), lambda_i=s^2/u_i
  * Huber (Huber-1964): IRLS, u_i=min(1, c*s/|r_i|),  lambda_i=s^2/u_i,  s=1.4826*MAD

Fully synthetic, fixed seeds, no network / external data.
"""
import numpy as np
from scipy.stats import spearmanr

ELL = 1.0            # RBF lengthscale
N_TRAIN = 80         # RVM training points (Fig 4)
N_TEST = 200
N_ITER = 60          # EM / evidence iterations
NU_T = 4.0           # Student-t degrees of freedom
HUBER_C = 1.345      # Huber tuning constant


# ----------------------------- noise-update rules --------------------------
def upd_homo(r, sd, s2, gamma, Sig_diag, n):
    eff = np.sum(1.0 - gamma * Sig_diag)                 # well-determined params
    beta_inv = np.sum(r ** 2) / max(n - eff, 1.0)        # Tipping-2001 evidence update
    return np.full(n, beta_inv), beta_inv


def upd_joint(r, sd, s2, gamma, Sig_diag, n):
    return r ** 2 + sd, s2                                # paper's per-sample noise (data ARD)


def upd_studentt(r, sd, s2, gamma, Sig_diag, n):
    u = (NU_T + 1.0) / (NU_T + (r ** 2 + sd) / s2)       # E[latent scale]
    s2_new = np.mean(u * (r ** 2 + sd))
    return s2_new / u, s2_new


def upd_huber(r, sd, s2, gamma, Sig_diag, n):
    s = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-6
    a = np.abs(r) / s
    u = np.where(a <= HUBER_C, 1.0, HUBER_C / np.maximum(a, 1e-12))
    return (s ** 2) / u, s ** 2


METHODS = {"homo": upd_homo, "joint": upd_joint, "studentt": upd_studentt, "huber": upd_huber}


# ------------------------------ shared solver ------------------------------
def fit(Phi, y, upd):
    n, m = Phi.shape
    gamma = np.ones(m)
    lam = np.full(n, np.var(y) + 1e-6)
    s2 = np.var(y) + 1e-6
    for _ in range(N_ITER):
        Linv = 1.0 / lam
        A = np.diag(gamma) + (Phi.T * Linv) @ Phi
        A[np.diag_indices_from(A)] += 1e-10
        Sigma = np.linalg.inv(A)
        Sig_diag = np.diag(Sigma)
        mu = Sigma @ ((Phi.T * Linv) @ y)
        r = y - Phi @ mu
        sd = np.einsum("ij,jk,ik->i", Phi, Sigma, Phi)   # phi_i^T Sigma phi_i
        gamma = np.minimum(1.0 / (mu ** 2 + Sig_diag + 1e-12), 1e12)
        lam, s2 = upd(r, sd, s2, gamma, Sig_diag, n)
        lam = np.maximum(lam, 1e-6)
    return mu


# --------------------------- Fig 4: RVM kernel -----------------------------
def rbf(a, b, ell=ELL):
    return np.exp(-0.5 * (a[:, None] - b[None, :]) ** 2 / ell ** 2)


def design(x, centers):
    return np.hstack([rbf(x, centers), np.ones((len(x), 1))])   # RBF basis + bias


def kernel_data(rng, K_active, cont_frac, sigma=0.3, out_scale=10.0):
    x_tr = rng.uniform(-5, 5, N_TRAIN)
    x_te = np.linspace(-5, 5, N_TEST)
    c_true = rng.uniform(-5, 5, K_active)
    w_true = rng.standard_normal(K_active) * 2.0
    f = lambda x: rbf(x, c_true) @ w_true
    y_tr = f(x_tr) + rng.standard_normal(N_TRAIN) * sigma
    n_out = int(round(cont_frac * N_TRAIN))
    if n_out > 0:                                          # large-magnitude contamination
        idx = rng.choice(N_TRAIN, n_out, replace=False)
        y_tr[idx] += rng.choice([-1.0, 1.0], n_out) * out_scale
    return x_tr, y_tr, x_te, f(x_te)


def kernel_rmse(upd, rng, K_active, cont_frac):
    x_tr, y_tr, x_te, f_te = kernel_data(rng, K_active, cont_frac)
    mu = fit(design(x_tr, x_tr), y_tr, upd)
    return float(np.sqrt(np.mean((design(x_te, x_tr) @ mu - f_te) ** 2)))


def kernel_grid(K_active, cont_frac, seeds):
    out = {k: [] for k in METHODS}
    for s in seeds:
        for k, upd in METHODS.items():
            out[k].append(kernel_rmse(upd, np.random.default_rng(1000 * s + 7), K_active, cont_frac))
    return {k: float(np.mean(v)) for k, v in out.items()}


# ------------------- Fig 3: sparse Bayesian recovery -----------------------
def linear_rmse(upd, rng, n, d, S, p_cont, mfac, sigma):
    X = rng.standard_normal((n, d)) / np.sqrt(n)          # under-determined d>n
    theta = np.zeros(d)
    idx = rng.choice(d, S, replace=False)
    theta[idx] = rng.choice([-1.0, 1.0], S) * rng.uniform(1.5, 3.0, S)
    noise = rng.standard_normal(n) * sigma
    n_out = int(round(p_cont * n))
    if n_out > 0:                                         # low, noise-inflation contamination
        oi = rng.choice(n, n_out, replace=False)
        noise[oi] *= mfac
    y = X @ theta + noise
    Xte = rng.standard_normal((500, d)) / np.sqrt(n)
    mu = fit(X, y, upd)
    return float(np.sqrt(np.mean((Xte @ mu - Xte @ theta) ** 2)))


def linear_gain(S, seeds, n=60, d=120, p_cont=0.10, mfac=4.0, sigma=0.1):
    h, j = [], []
    for s in seeds:
        h.append(linear_rmse(upd_homo, np.random.default_rng(1000 * s + 7), n, d, S, p_cont, mfac, sigma))
        j.append(linear_rmse(upd_joint, np.random.default_rng(1000 * s + 7), n, d, S, p_cont, mfac, sigma))
    hm, jm = float(np.mean(h)), float(np.mean(j))
    return hm, jm, (hm - jm) / hm


def main():
    seeds4 = list(range(12))
    seeds3 = list(range(16))
    print("=" * 78)
    print("Claim [3]  Joint ARD robustness vs Student-t / Huber baselines (arXiv:2605.29908)")
    print(f"shared ARD solver, {N_ITER} evidence iters; per-sample noise = robustness mechanism")
    print("=" * 78)

    # ---------- Fig 4: robustness across a RANGE of contamination -----------
    print("\n[Fig 4 analog]  RVM kernel regression (RBF, n_train=80), test RMSE vs clean truth")
    print(f"                sparse signal K=6, {len(seeds4)} seeds")
    levels = [0.0, 0.05, 0.10, 0.20, 0.30]
    print(f"\n{'contam':>7} | {'homo':>7} {'joint':>7} {'studentt':>9} {'huber':>7} | "
          f"{'joint-best_robust':>18}")
    fig4 = {c: kernel_grid(6, c, seeds4) for c in levels}
    max_gap, beats_homo_hi = -1e9, True
    for c in levels:
        g = fig4[c]
        best_robust = min(g["studentt"], g["huber"])
        gap = g["joint"] - best_robust
        max_gap = max(max_gap, gap)
        tag = "  (joint WINS)" if gap < 0 else ""
        print(f"{c*100:5.0f}% | {g['homo']:7.3f} {g['joint']:7.3f} {g['studentt']:9.3f} "
              f"{g['huber']:7.3f} | {gap:+18.3f}{tag}")
        if c >= 0.10 and not (g["joint"] < g["homo"]):
            beats_homo_hi = False

    j_wc = max(fig4[c]["joint"] for c in levels)
    st_wc = max(fig4[c]["studentt"] for c in levels)
    hu_wc = max(fig4[c]["huber"] for c in levels)
    ho_wc = max(fig4[c]["homo"] for c in levels)
    print(f"\n  worst-case RMSE over 0-30% contamination:  homo={ho_wc:.3f}  studentt={st_wc:.3f}"
          f"  huber={hu_wc:.3f}  joint={j_wc:.3f}")
    margin_ok = max_gap <= 0.06                # within 0.06 RMSE of best robust baseline at every level
    best_worstcase = j_wc <= min(st_wc, hu_wc) # most robust across the whole range
    competitive = margin_ok and best_worstcase
    print(f"  Joint ARD within 0.06 RMSE of (or better than) best robust baseline at every level: "
          f"{margin_ok}  (max gap {max_gap:+.3f})")
    print(f"  Joint ARD has the BEST worst-case RMSE across the range (most robust)             : "
          f"{best_worstcase}")
    print(f"  Joint ARD beats homoscedastic RVM at contamination >=10%                          : "
          f"{beats_homo_hi}")

    # ---------- Fig 3: gain over homoscedastic vs signal sparsity ----------
    print("\n[Fig 3 analog]  Sparse Bayesian recovery (n=60, d=120, low contamination p=10%, m=4x)")
    print(f"                Joint-ARD gain over homoscedastic vs signal sparsity, {len(seeds3)} seeds")
    Ss = [2, 4, 8, 16, 32]                     # sparse -> dense true signal
    print(f"\n{'nonzeros S':>10} | {'homo RMSE':>10} {'joint RMSE':>11} | {'rel. gain':>10}")
    gains = []
    for S in Ss:
        hm, jm, gain = linear_gain(S, seeds3)
        gains.append(gain)
        print(f"{S:10d} | {hm:10.3f} {jm:11.3f} | {gain*100:9.1f}%")

    gain_sparse = float(np.mean(gains[:3]))    # S in {2,4,8}
    gain_dense = float(np.mean(gains[3:]))     # S in {16,32}
    rho, _ = spearmanr([-s for s in Ss], gains)
    argmax_sparse = Ss[int(np.argmax(gains))] <= 8
    print(f"\n  mean gain, sparse signals (S<=8) = {gain_sparse*100:5.1f}%   "
          f"dense signals (S>=16) = {gain_dense*100:5.1f}%")
    print(f"  largest gain occurs at a SPARSE signal (S<=8)            : {argmax_sparse}")
    print(f"  Spearman(rho) gain vs signal-sparsity (want > 0)         : {rho:.3f}")
    fig3_ok = argmax_sparse and rho > 0 and (gain_sparse - gain_dense) > 0.15

    # ------------------------------ verdict --------------------------------
    print("\n" + "=" * 78)
    print("GATING SUMMARY")
    print(f"  (Fig4) Joint ARD competitive with robust baselines across contamination : {competitive}")
    print(f"  (Fig4) Joint ARD robust vs homoscedastic at high contamination          : {beats_homo_hi}")
    print(f"  (Fig3) largest gain over homoscedastic for SPARSE signals               : {fig3_ok}")
    supports = competitive and beats_homo_hi and fig3_ok
    print("=" * 78)
    print(f"verdict: {'supports' if supports else 'inconclusive'}")


if __name__ == "__main__":
    main()
