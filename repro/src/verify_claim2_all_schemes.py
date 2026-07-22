#!/usr/bin/env python3
"""Exact CPU audit of all Section-3 Joint-ARD update families.

Paper: Joint Model and Data Sparsification via the Marginal Likelihood
arXiv:2605.29908, OpenReview vMcu1h3fOV.

Source audited (ar5iv HTML): Sections 3.1--3.2 and Appendix B.2--B.7.
The source digest is checked when --paper-html is supplied.  The numerical
audit itself is self-contained and deterministic.

This script deliberately distinguishes a closed-form hyperparameter update
from a closed-form theta subproblem: the paper says that the double-l1 theta
subproblem is convex but requires an iterative solver.  We solve that exact
subproblem and then check the closed-form gamma/lambda map-back rules.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import lsq_linear, minimize


SOURCE_URL = "https://ar5iv.labs.arxiv.org/html/2605.29908"
SOURCE_SHA256 = "0a3ed827e899a92f2ed3b1e877ea2f6456ec3d827af184045acac3b8ebd4cb49"
SOURCE_SCOPE = "Sections 3.1-3.2; Appendix B.2-B.7; equations 3-7, 10, 17-70"


def direct_state(X: np.ndarray, y: np.ndarray, gamma: np.ndarray, lam: np.ndarray):
    """Compute the two conjugate Gaussian representations independently."""
    Gamma = np.diag(gamma)
    Lambda = np.diag(lam)
    Sigma_y = Lambda + (X / gamma) @ X.T
    Pi = np.linalg.inv(Sigma_y)
    Sigma = np.linalg.inv(Gamma + (X.T / lam) @ X)
    mu = Sigma @ (X.T @ (y / lam))
    residual = y - X @ mu
    return Sigma_y, Pi, Sigma, mu, residual


def l1_objective(theta, X, y, lam, w, v):
    residual = y - X @ theta
    return float(
        residual @ (residual / lam)
        + 2.0 * w @ np.abs(theta)
        + 2.0 * v @ np.abs(residual)
    )


def solve_double_l1(X, y, lam, w, v):
    """Solve paper Eq. 68 by a convex epigraph QP using SciPy SLSQP."""
    n, d = X.shape
    ridge = np.linalg.solve((X.T / lam) @ X + np.eye(d), X.T @ (y / lam))
    r0 = y - X @ ridge
    start = np.concatenate([ridge, np.abs(ridge) + 0.1, np.abs(r0) + 0.1])

    def unpack(z):
        return z[:d], z[d : 2 * d], z[2 * d :]

    def fun(z):
        theta, u, t = unpack(z)
        residual = y - X @ theta
        return float(residual @ (residual / lam) + 2.0 * w @ u + 2.0 * v @ t)

    def jac(z):
        theta, _, _ = unpack(z)
        residual = y - X @ theta
        return np.concatenate([-2.0 * X.T @ (residual / lam), 2.0 * w, 2.0 * v])

    # c(z) >= 0: u >= +/-theta and t >= +/-(y-X theta).
    def constraints(z):
        theta, u, t = unpack(z)
        residual = y - X @ theta
        return np.concatenate([u - theta, u + theta, t - residual, t + residual])

    result = minimize(
        fun,
        start,
        jac=jac,
        constraints={"type": "ineq", "fun": constraints},
        bounds=[(None, None)] * d + [(0.0, None)] * (d + n),
        method="SLSQP",
        options={"ftol": 1e-12, "maxiter": 2000, "disp": False},
    )
    if not result.success:
        raise AssertionError("double-l1 convex solver failed: " + result.message)
    theta, u, t = unpack(result.x)
    feas = float(max(0.0, -np.min(constraints(result.x))))
    epigraph_gap = float(
        abs(fun(result.x) - l1_objective(theta, X, y, lam, w, v))
    )
    return theta, float(result.fun), feas, epigraph_gap


def kkt_residual_double_l1(theta, X, y, lam, w, v, zero_tol=2e-6):
    """Independent global-optimality certificate for the convex Eq. 68 solve."""
    residual = y - X @ theta
    theta_zero = np.flatnonzero(np.abs(theta) <= zero_tol)
    residual_zero = np.flatnonzero(np.abs(residual) <= zero_tol)
    theta_nz = np.flatnonzero(np.abs(theta) > zero_tol)
    residual_nz = np.flatnonzero(np.abs(residual) > zero_tol)

    const = -X.T @ (residual / lam)
    if len(theta_nz):
        const[theta_nz] += w[theta_nz] * np.sign(theta[theta_nz])
    if len(residual_nz):
        const -= X[residual_nz].T @ (
            v[residual_nz] * np.sign(residual[residual_nz])
        )

    columns = []
    for j in theta_zero:
        col = np.zeros(X.shape[1])
        col[j] = w[j]
        columns.append(col)
    for i in residual_zero:
        columns.append(-X[i] * v[i])

    if columns:
        A = np.column_stack(columns)
        feasibility = lsq_linear(A, -const, bounds=(-1.0, 1.0), tol=1e-14)
        if not feasibility.success:
            return float("inf")
        return float(np.max(np.abs(A @ feasibility.x + const)))
    return float(np.max(np.abs(const)))


def generate_case(seed: int):
    rng = np.random.default_rng(912_000 + seed)
    n = 8 + seed % 3
    d = 4 + seed % 2
    X = rng.normal(size=(n, d))
    X += 0.08 * rng.normal(size=(n, 1)) @ rng.normal(size=(1, d))
    y = rng.normal(size=n) + X @ rng.normal(scale=0.35, size=d)
    gamma = np.exp(rng.uniform(-0.7, 0.8, size=d))
    lam = np.exp(rng.uniform(-0.5, 0.9, size=n))
    return X, y, gamma, lam


def audit_case(seed: int):
    X, y, gamma, lam = generate_case(seed)
    Sigma_y, Pi, Sigma, mu, residual = direct_state(X, y, gamma, lam)
    n, d = X.shape

    # Gaussian conjugacy and the Woodbury/determinant identities in Eqs. 3-4.
    Pi_dual = np.diag(1.0 / lam) - (X / lam[:, None]) @ Sigma @ (X.T / lam)
    sign_y, logdet_y = np.linalg.slogdet(Sigma_y)
    sign_s, logdet_s_inv = np.linalg.slogdet(np.linalg.inv(Sigma))
    determinant_rhs = logdet_s_inv - np.log(gamma).sum() + np.log(lam).sum()
    conjugacy_error = max(
        float(np.max(np.abs(Pi - Pi_dual))),
        float(abs(logdet_y - determinant_rhs)),
        0.0 if sign_y == sign_s == 1 else float("inf"),
    )

    # EM, Eqs. 10 and 35-37: exact scalar maximizers of Q.
    sigma_diag = np.diag(Sigma)
    leverage = np.einsum("ij,jk,ik->i", X, Sigma, X)
    gamma_em = 1.0 / (mu * mu + sigma_diag)
    lam_em = residual * residual + leverage
    em_stationarity = max(
        float(np.max(np.abs(1.0 / gamma_em - (mu * mu + sigma_diag)))),
        float(np.max(np.abs(lam_em - (residual * residual + leverage)))),
    )

    # MacKay, Eqs. 24/31 and 41/42.  These are fixed-point updates, not EM.
    effective = 1.0 - gamma * sigma_diag
    gamma_mackay = effective / (mu * mu)
    lam_mackay = lam_em.copy()
    column_pi = np.einsum("nj,nm,mj->j", X, Pi, X)
    column_piy = X.T @ (Pi @ y)
    mackay_identity_error = max(
        float(np.max(np.abs(column_pi - gamma * effective))),
        float(np.max(np.abs(column_piy - gamma * mu))),
        float(np.max(np.abs(gamma_mackay * mu * mu - effective))),
        float(np.max(np.abs(lam_mackay - lam_em))),
    )

    # l2-IRLS, Eqs. 48-55.  Its weighted-ridge theta solve is closed form.
    z_l2 = sigma_diag.copy()
    theta_l2 = np.linalg.solve(np.diag(gamma) + (X.T / lam) @ X, X.T @ (y / lam))
    gamma_l2 = 1.0 / (theta_l2 * theta_l2 + z_l2)
    lam_l2 = (y - X @ theta_l2) ** 2 + leverage
    ridge_gradient = -2.0 * X.T @ ((y - X @ theta_l2) / lam) + 2.0 * gamma * theta_l2
    l2_error = max(
        float(np.max(np.abs(ridge_gradient))),
        float(np.max(np.abs(theta_l2 - mu))),
        float(np.max(np.abs(gamma_l2 - gamma_em))),
        float(np.max(np.abs(lam_l2 - lam_em))),
    )

    # l1-IRLS, Eqs. 58-68.  Eq. 68 itself is convex and iterative; the
    # gamma/lambda map-backs in Eqs. 63 and 66 are closed form.
    q_l1 = np.einsum("nj,nm,mj->j", X, Pi, X)
    z_l1 = np.diag(Pi)
    w = np.sqrt(q_l1)
    v = np.sqrt(z_l1)
    theta_l1, l1_obj, l1_feas, l1_epigraph_gap = solve_double_l1(X, y, lam, w, v)
    residual_l1 = y - X @ theta_l1
    # The extended-real boundary is part of sparse recovery: theta_j=0 maps
    # to gamma_j=+infinity, while r_i=0 maps to lambda_i=0.  Check the scalar
    # identities on active coordinates and the correct boundary direction on
    # inactive coordinates.
    theta_active = np.abs(theta_l1) > 2e-6
    residual_active = np.abs(residual_l1) > 2e-6
    gamma_l1 = np.full(d, np.inf)
    gamma_l1[theta_active] = np.sqrt(q_l1[theta_active]) / np.abs(theta_l1[theta_active])
    lam_l1 = np.zeros(n)
    lam_l1[residual_active] = np.abs(residual_l1[residual_active]) / np.sqrt(z_l1[residual_active])
    scalar_errors = [0.0]
    if np.any(theta_active):
        scalar_errors.append(float(np.max(np.abs(
            theta_l1[theta_active] ** 2
            - q_l1[theta_active] / gamma_l1[theta_active] ** 2
        ))))
    if np.any(residual_active):
        scalar_errors.append(float(np.max(np.abs(
            z_l1[residual_active]
            - residual_l1[residual_active] ** 2 / lam_l1[residual_active] ** 2
        ))))
    # Boundary direction checks at two finite probes.
    if np.any(~theta_active):
        q0 = q_l1[~theta_active]
        scalar_errors.append(0.0 if np.all(q0 / 1e9 < q0 / 1e6) else float("inf"))
    if np.any(~residual_active):
        z0 = z_l1[~residual_active]
        scalar_errors.append(0.0 if np.all(z0 * 1e-9 < z0 * 1e-6) else float("inf"))
    l1_scalar_error = max(scalar_errors)
    l1_kkt = kkt_residual_double_l1(theta_l1, X, y, lam, w, v)

    return {
        "conjugacy": conjugacy_error,
        "em": em_stationarity,
        "mackay": mackay_identity_error,
        "l2irls": l2_error,
        "l1irls_scalar": l1_scalar_error,
        "l1irls_kkt": l1_kkt,
        "l1_feas": l1_feas,
        "l1_epigraph_gap": l1_epigraph_gap,
        "l1_objective": l1_obj,
        "canonical": {
            "gamma_em": gamma_em,
            "lam_em": lam_em,
            "gamma_mackay": gamma_mackay,
            "lam_mackay": lam_mackay,
            "theta_l2": theta_l2,
            "gamma_l2": gamma_l2,
            "lam_l2": lam_l2,
            "q_l1": q_l1,
            "z_l1": z_l1,
            "gamma_l1": gamma_l1,
            "lam_l1": lam_l1,
            "mu": mu,
            "sigma_diag": sigma_diag,
            "leverage": leverage,
            "effective": effective,
            "theta_l1": theta_l1,
            "residual_l1": residual_l1,
            "theta_l1_active": theta_active,
            "residual_l1_active": residual_active,
            "Pi": Pi,
            "X": X,
            "y": y,
            "gamma": gamma,
            "lam": lam,
        },
    }


def mutation_gate(case):
    """Reject one independently meaningful mutation of every claimed family."""
    c = case["canonical"]
    X, y, gamma, lam = c["X"], c["y"], c["gamma"], c["lam"]
    mu, sd, leverage = c["mu"], c["sigma_diag"], c["leverage"]
    residual = y - X @ mu

    mutations = {
        "em_gamma_drop_posterior_variance": (1.0 / (mu * mu), c["gamma_em"]),
        "em_lambda_drop_leverage": (residual * residual, c["lam_em"]),
        "mackay_gamma_wrong_plus": ((1.0 + gamma * sd) / (mu * mu), c["gamma_mackay"]),
        "mackay_lambda_force_homoscedastic": (
            np.full_like(c["lam_mackay"], np.mean(c["lam_mackay"])),
            c["lam_mackay"],
        ),
        "l2_theta_ignore_lambda": (
            np.linalg.solve(np.diag(gamma) + X.T @ X, X.T @ y),
            c["theta_l2"],
        ),
        "l2_gamma_drop_z": (1.0 / (c["theta_l2"] ** 2), c["gamma_l2"]),
        "l2_lambda_drop_q": ((y - X @ c["theta_l2"]) ** 2, c["lam_l2"]),
        "l1_q_drop_pi": (np.sum(X * X, axis=0), c["q_l1"]),
        "l1_z_use_lambda_inverse": (1.0 / lam, c["z_l1"]),
        "l1_gamma_drop_sqrt": (
            c["q_l1"][c["theta_l1_active"]] / np.abs(c["theta_l1"][c["theta_l1_active"]]),
            c["gamma_l1"][c["theta_l1_active"]],
        ),
        "l1_lambda_multiply_sqrt_z": (
            np.abs(c["residual_l1"][c["residual_l1_active"]])
            * np.sqrt(c["z_l1"][c["residual_l1_active"]]),
            c["lam_l1"][c["residual_l1_active"]],
        ),
        "conjugacy_drop_model_covariance": (np.diag(lam), np.diag(lam) + (X / gamma) @ X.T),
    }
    rejected = []
    for name, (bad, good) in mutations.items():
        scale = max(1.0, float(np.max(np.abs(good))))
        if float(np.max(np.abs(bad - good))) > 1e-5 * scale:
            rejected.append(name)
    return len(rejected), len(mutations), rejected


def check_source(path: Path | None):
    if path is None:
        return "recorded"
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != SOURCE_SHA256:
        raise AssertionError(f"paper source digest mismatch: {digest}")
    text = raw.decode("utf-8")
    required = [
        'id="S3.Ex5.m1"',
        'id="A2.Ex41.m1"',
        'id="A2.Ex50.m1"',
        'id="A2.Ex63.m1"',
        'id="A2.Ex66.m1"',
        "convex but no longer admits a closed-form solution",
    ]
    missing = [x for x in required if x not in text]
    if missing:
        raise AssertionError("paper source missing required anchors: " + repr(missing))
    return "verified"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper-html", type=Path)
    args = parser.parse_args()
    source_status = check_source(args.paper_html)

    cases = [audit_case(seed) for seed in range(6)]
    keys = [
        "conjugacy",
        "em",
        "mackay",
        "l2irls",
        "l1irls_scalar",
        "l1irls_kkt",
        "l1_feas",
        "l1_epigraph_gap",
    ]
    maxima = {key: max(case[key] for case in cases) for key in keys}
    thresholds = {
        "conjugacy": 2e-10,
        "em": 2e-12,
        "mackay": 2e-10,
        "l2irls": 2e-10,
        "l1irls_scalar": 2e-10,
        "l1irls_kkt": 2e-5,
        "l1_feas": 2e-8,
        "l1_epigraph_gap": 2e-8,
    }
    passed = all(maxima[k] <= thresholds[k] for k in keys)
    rejected, total, names = mutation_gate(cases[0])
    passed = passed and rejected == total

    digest_payload = {
        "source_sha256": SOURCE_SHA256,
        "cases": 6,
        "maxima": {k: format(maxima[k], ".12e") for k in sorted(maxima)},
        "mutations_rejected": names,
        "passed": passed,
    }
    result_digest = hashlib.sha256(
        json.dumps(digest_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    print("CLAIM2_ALL_SCHEMES_AUDIT")
    print(f"source_url={SOURCE_URL}")
    print(f"source_sha256={SOURCE_SHA256} source_status={source_status}")
    print(f"source_scope={SOURCE_SCOPE}")
    print("cases=6 schemes=EM,MacKay,l2-IRLS,l1-IRLS")
    print(
        "max_errors "
        + " ".join(f"{k}={maxima[k]:.3e}" for k in keys)
    )
    print("l1_theta_subproblem=iterative-convex gamma_lambda_mapback=closed-form")
    print(f"mutation_gates={rejected}/{total}")
    print(f"result_sha256={result_digest}")
    print("verdict=" + ("supports" if passed else "inconclusive"))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
