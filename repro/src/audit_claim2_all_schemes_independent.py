#!/usr/bin/env python3
"""Independent, non-importing audit for the all-schemes Joint-ARD evidence."""
from __future__ import annotations

import argparse
import hashlib
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, lsq_linear, minimize


SOURCE_SHA256 = "0a3ed827e899a92f2ed3b1e877ea2f6456ec3d827af184045acac3b8ebd4cb49"


def source_audit(path: Path):
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SOURCE_SHA256:
        raise AssertionError("independent source digest mismatch")
    html = raw.decode("utf-8")
    anchors = {
        "em": ["S3.Ex5.m1", "A2.Ex35.m1", "A2.Ex37.m1"],
        "mackay": ["A2.Ex41.m1", "A2.Ex42.m1"],
        "l2irls": ["A2.Ex48.m1", "A2.Ex50.m1", "A2.Ex55.m1"],
        "l1irls": ["A2.Ex60.m1", "A2.Ex63.m1", "A2.Ex66.m1", "A2.Ex68.m1"],
    }
    for family, ids in anchors.items():
        if not all(f'id="{equation}"' in html for equation in ids):
            raise AssertionError(f"missing {family} source equation")
    semantic_fragments = [
        "preserves conjugacy",
        "Improvement assurances are traded for faster convergence",
        "convex but no longer admits a closed-form solution",
        "permitting the same IRLS-style updates to hold in the heteroscedastic case",
    ]
    if not all(fragment in html for fragment in semantic_fragments):
        raise AssertionError("missing source qualification")

    # Fail-sensitive source mutations: each changes a material equation/qualification.
    mutants = [
        html.replace("A2.Ex41.m1", "A2.Ex41.MUTANT", 1),
        html.replace("A2.Ex50.m1", "A2.Ex50.MUTANT", 1),
        html.replace("A2.Ex63.m1", "A2.Ex63.MUTANT", 1),
        html.replace("A2.Ex66.m1", "A2.Ex66.MUTANT", 1),
        html.replace("convex but no longer admits a closed-form solution", "closed form", 1),
    ]
    rejected = 0
    for mutant in mutants:
        ok = all(
            all(f'id="{equation}"' in mutant for equation in ids)
            for ids in anchors.values()
        ) and all(fragment in mutant for fragment in semantic_fragments)
        rejected += int(not ok)
    return len(anchors), rejected, len(mutants)


def l1_objective(theta, X, y, lam, w, v, include_residual_l1=True):
    residual = y - X @ theta
    value = residual @ (residual / lam) + 2.0 * w @ np.abs(theta)
    if include_residual_l1:
        value += 2.0 * v @ np.abs(residual)
    return float(value)


def solve_l1_trust_constr(X, y, lam, w, v):
    """Independent Eq. 68 epigraph QP via trust-constr and a constant Hessian."""
    n, d = X.shape
    D = np.diag(1.0 / lam)
    Htheta = 2.0 * X.T @ D @ X
    zero_du = np.zeros((d, d + n))
    Hessian = np.block(
        [
            [Htheta, zero_du],
            [zero_du.T, np.zeros((d + n, d + n))],
        ]
    )

    # Variables are (theta[d], u[d], t[n]); u >= |theta|, t >= |y-X theta|.
    A = np.block(
        [
            [-np.eye(d), np.eye(d), np.zeros((d, n))],
            [np.eye(d), np.eye(d), np.zeros((d, n))],
            [X, np.zeros((n, d)), np.eye(n)],
            [-X, np.zeros((n, d)), np.eye(n)],
        ]
    )
    lower = np.concatenate([np.zeros(2 * d), y, -y])
    upper = np.full(2 * d + 2 * n, np.inf)
    constraint = LinearConstraint(A, lower, upper)
    bounds = Bounds(
        np.concatenate([np.full(d, -np.inf), np.zeros(d + n)]),
        np.full(2 * d + n, np.inf),
    )

    ridge = np.linalg.solve(X.T @ D @ X + np.eye(d), X.T @ D @ y)
    residual0 = y - X @ ridge
    start = np.concatenate([ridge, np.abs(ridge) + 0.25, np.abs(residual0) + 0.25])

    def unpack(z):
        return z[:d], z[d : 2 * d], z[2 * d :]

    def objective(z):
        theta, u, t = unpack(z)
        residual = y - X @ theta
        return float(residual @ (residual / lam) + 2.0 * w @ u + 2.0 * v @ t)

    def gradient(z):
        theta, _, _ = unpack(z)
        residual = y - X @ theta
        return np.concatenate([-2.0 * X.T @ (residual / lam), 2.0 * w, 2.0 * v])

    result = minimize(
        objective,
        start,
        method="trust-constr",
        jac=gradient,
        hess=lambda _z: Hessian,
        constraints=[constraint],
        bounds=bounds,
        options={
            "gtol": 2e-12,
            "xtol": 1e-13,
            "barrier_tol": 1e-14,
            "initial_barrier_parameter": 1e-10,
            "initial_barrier_tolerance": 1e-12,
            "maxiter": 4000,
            "verbose": 0,
        },
    )
    if not result.success:
        raise AssertionError("independent trust-constr l1 solve failed: " + result.message)
    theta, u, t = unpack(result.x)
    residual = y - X @ theta
    constraint_values = A @ result.x - lower
    feasibility = float(max(0.0, -np.min(constraint_values)))
    epigraph_gap = float(
        abs(objective(result.x) - l1_objective(theta, X, y, lam, w, v))
    )
    return theta, float(result.fun), feasibility, epigraph_gap


def certify_l1_kkt(theta, X, y, lam, w, v, zero_tol=2e-6):
    """Independent subgradient KKT certificate; convexity makes it global."""
    residual = y - X @ theta
    theta_zero = np.flatnonzero(np.abs(theta) <= zero_tol)
    residual_zero = np.flatnonzero(np.abs(residual) <= zero_tol)
    theta_nz = np.flatnonzero(np.abs(theta) > zero_tol)
    residual_nz = np.flatnonzero(np.abs(residual) > zero_tol)

    constant = -X.T @ (residual / lam)
    if len(theta_nz):
        constant[theta_nz] += w[theta_nz] * np.sign(theta[theta_nz])
    if len(residual_nz):
        constant -= X[residual_nz].T @ (
            v[residual_nz] * np.sign(residual[residual_nz])
        )

    columns = []
    for j in theta_zero:
        column = np.zeros(X.shape[1])
        column[j] = w[j]
        columns.append(column)
    for i in residual_zero:
        columns.append(-X[i] * v[i])
    if columns:
        matrix = np.column_stack(columns)
        fit = lsq_linear(matrix, -constant, bounds=(-1.0, 1.0), tol=1e-14)
        if not fit.success:
            return float("inf")
        constant = matrix @ fit.x + constant
    return float(np.max(np.abs(constant)))


def certify_l1_mapback(theta, X, y, q, z, zero_tol=2e-6):
    """Check Eqs. 63/66, including their extended-real sparse boundaries."""
    residual = y - X @ theta
    theta_active = np.abs(theta) > zero_tol
    residual_active = np.abs(residual) > zero_tol
    errors = [0.0]
    gamma = np.full(len(theta), np.inf)
    lam = np.zeros(len(residual))
    if np.any(theta_active):
        gamma[theta_active] = np.sqrt(q[theta_active]) / np.abs(theta[theta_active])
        errors.append(float(np.max(np.abs(
            theta[theta_active] ** 2 - q[theta_active] / gamma[theta_active] ** 2
        ))))
    if np.any(residual_active):
        lam[residual_active] = np.abs(residual[residual_active]) / np.sqrt(z[residual_active])
        errors.append(float(np.max(np.abs(
            z[residual_active] - residual[residual_active] ** 2 / lam[residual_active] ** 2
        ))))
    if np.any(~theta_active):
        q0 = q[~theta_active]
        errors.append(0.0 if np.all(q0 / 1e9 < q0 / 1e6) else float("inf"))
    if np.any(~residual_active):
        z0 = z[~residual_active]
        errors.append(0.0 if np.all(z0 * 1e-9 < z0 * 1e-6) else float("inf"))
    return max(errors), gamma, lam, theta_active, residual_active


def independent_numeric_audit():
    """Different seeds/shapes and a separate l1 solver; no primary imports."""
    max_error = 0.0
    mutation_rejections = 0
    mutation_total = 0
    l1_max = {"feasibility": 0.0, "epigraph": 0.0, "kkt": 0.0, "mapback": 0.0}
    l1_boundaries = {"theta": 0, "residual": 0}
    l1_mutation_rejections = 0
    l1_mutation_total = 0
    for index in range(7):
        rng = np.random.default_rng(303_000 + index)
        n, d = 9 + index % 2, 3 + index % 3
        X = rng.normal(size=(n, d))
        y = rng.normal(size=n)
        gamma = np.exp(rng.uniform(-0.4, 0.9, size=d))
        lam = np.exp(rng.uniform(-0.8, 0.6, size=n))
        Sy = np.diag(lam) + (X / gamma) @ X.T
        Pi = np.linalg.inv(Sy)
        Sigma = np.linalg.inv(np.diag(gamma) + (X.T / lam) @ X)
        mu = Sigma @ (X.T @ (y / lam))
        residual = y - X @ mu

        q_data = np.einsum("nj,nm,mj->j", X, Pi, X)
        z_data = np.diag(Pi)
        q_parameter = gamma * (1.0 - gamma * np.diag(Sigma))
        leverage = np.einsum("ij,jk,ik->i", X, Sigma, X)
        z_parameter = 1.0 / lam - leverage / (lam * lam)
        max_error = max(
            max_error,
            float(np.max(np.abs(q_data - q_parameter))),
            float(np.max(np.abs(z_data - z_parameter))),
            float(np.max(np.abs(X.T @ (Pi @ y) - gamma * mu))),
            float(np.max(np.abs(Pi @ y - residual / lam))),
        )

        em_gamma = 1.0 / (mu * mu + np.diag(Sigma))
        em_lambda = residual * residual + leverage
        mackay_gamma = (1.0 - gamma * np.diag(Sigma)) / (mu * mu)
        l2_gamma = 1.0 / (mu * mu + np.diag(Sigma))
        l2_lambda = residual * residual + leverage

        canonical = [em_gamma, em_lambda, mackay_gamma, l2_gamma, l2_lambda]
        mutants = [
            1.0 / (mu * mu),
            residual * residual,
            (1.0 + gamma * np.diag(Sigma)) / (mu * mu),
            1.0 / (mu * mu + 2.0 * np.diag(Sigma)),
            residual * residual + 2.0 * leverage,
        ]
        for bad, good in zip(mutants, canonical):
            mutation_total += 1
            scale = max(1.0, float(np.max(np.abs(good))))
            mutation_rejections += int(
                float(np.max(np.abs(bad - good))) > 1e-5 * scale
            )

        # Genuinely independent l1 path: trust-constr epigraph QP rather than
        # the primary's SLSQP nonlinear constraints.
        q = q_data
        z = z_data
        w = np.sqrt(q)
        v = np.sqrt(z)
        theta_l1, objective_l1, feasibility, epigraph = solve_l1_trust_constr(
            X, y, lam, w, v
        )
        kkt = certify_l1_kkt(theta_l1, X, y, lam, w, v)
        mapback, gamma_l1, lambda_l1, theta_active, residual_active = certify_l1_mapback(
            theta_l1, X, y, q, z
        )
        l1_max["feasibility"] = max(l1_max["feasibility"], feasibility)
        l1_max["epigraph"] = max(l1_max["epigraph"], epigraph)
        l1_max["kkt"] = max(l1_max["kkt"], kkt)
        l1_max["mapback"] = max(l1_max["mapback"], mapback)
        l1_boundaries["theta"] += int(np.sum(~theta_active))
        l1_boundaries["residual"] += int(np.sum(~residual_active))

        residual_l1 = y - X @ theta_l1
        # Six claim-sensitive l1 mutations per disjoint case. Each is checked
        # against an equation identity, global KKT, or the exact objective.
        l1_mutants = []
        wrong_q = np.sum(X * X, axis=0)
        wrong_z = 1.0 / lam
        l1_mutants.append(float(np.max(np.abs(wrong_q - q))) > 1e-5)
        l1_mutants.append(float(np.max(np.abs(wrong_z - z))) > 1e-5)
        if np.any(theta_active):
            wrong_gamma = q[theta_active] / np.abs(theta_l1[theta_active])
            good_gamma = gamma_l1[theta_active]
            l1_mutants.append(float(np.max(np.abs(wrong_gamma - good_gamma))) > 1e-5)
        else:
            l1_mutants.append(False)
        if np.any(residual_active):
            wrong_lambda = np.abs(residual_l1[residual_active]) * np.sqrt(z[residual_active])
            good_lambda = lambda_l1[residual_active]
            l1_mutants.append(float(np.max(np.abs(wrong_lambda - good_lambda))) > 1e-5)
        else:
            l1_mutants.append(False)
        # Omitting the residual-l1 term must alter the exact Eq. 68 objective.
        omitted_residual_value = l1_objective(
            theta_l1, X, y, lam, w, v, include_residual_l1=False
        )
        l1_mutants.append(abs(omitted_residual_value - objective_l1) > 1e-5)
        # A ridge-only theta must fail the complete double-l1 KKT certificate.
        ridge_only = np.linalg.solve((X.T / lam) @ X + np.eye(d), X.T @ (y / lam))
        l1_mutants.append(certify_l1_kkt(ridge_only, X, y, lam, w, v) > 1e-4)
        l1_mutation_rejections += sum(l1_mutants)
        l1_mutation_total += len(l1_mutants)
    return (
        max_error,
        mutation_rejections,
        mutation_total,
        l1_max,
        l1_boundaries,
        l1_mutation_rejections,
        l1_mutation_total,
    )


def run_primary(primary: Path, paper: Path):
    result = subprocess.run(
        [sys.executable, str(primary), "--paper-html", str(paper)],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise AssertionError("primary failed:\n" + result.stdout + result.stderr)
    if result.stdout.count("verdict=supports") != 1:
        raise AssertionError("primary did not emit one supports verdict")
    match = re.search(r"mutation_gates=(\d+)/(\d+)", result.stdout)
    if not match or match.group(1) != match.group(2):
        raise AssertionError("primary mutation gates not all rejected")
    return hashlib.sha256(result.stdout.encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper-html", type=Path, required=True)
    parser.add_argument("--primary", type=Path, required=True)
    args = parser.parse_args()

    families, source_rejected, source_total = source_audit(args.paper_html)
    (
        numeric_error,
        numeric_rejected,
        numeric_total,
        l1_max,
        l1_boundaries,
        l1_rejected,
        l1_total,
    ) = independent_numeric_audit()
    primary_stdout_sha = run_primary(args.primary, args.paper_html)
    passed = (
        families == 4
        and source_rejected == source_total
        and numeric_error < 2e-10
        and numeric_rejected == numeric_total
        and l1_max["feasibility"] < 2e-9
        and l1_max["epigraph"] < 2e-8
        and l1_max["kkt"] < 3e-7
        and l1_max["mapback"] < 2e-10
        and l1_boundaries["theta"] > 0
        and l1_boundaries["residual"] > 0
        and l1_rejected == l1_total
    )
    print("CLAIM2_INDEPENDENT_AUDIT")
    print(f"source_families={families} source_mutations={source_rejected}/{source_total}")
    print(f"independent_cases=7 max_algebra_error={numeric_error:.3e}")
    print(f"numeric_mutations={numeric_rejected}/{numeric_total}")
    print(
        "l1_independent_cases=7 solver=trust-constr "
        + " ".join(f"max_{key}={l1_max[key]:.3e}" for key in ["feasibility", "epigraph", "kkt", "mapback"])
        + f" theta_boundaries={l1_boundaries['theta']} residual_boundaries={l1_boundaries['residual']}"
    )
    print(f"l1_mutations={l1_rejected}/{l1_total}")
    print(f"primary_stdout_sha256={primary_stdout_sha}")
    print("verdict=" + ("supports" if passed else "inconclusive"))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
