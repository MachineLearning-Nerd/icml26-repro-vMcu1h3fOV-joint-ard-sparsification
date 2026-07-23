#!/usr/bin/env python3
"""Paper-protocol Table 3 reproduction for Energy, Carbon, and Protein.

This experiment deliberately uses the documented Appendix-D.1 endpoint
configured below.  It does not tune against the published answers.
"""
from __future__ import annotations

import hashlib
import json
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path
import sys

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parent))
from claim1_datasets import load_route_dataset
from claim1_paper_scale import _prepare_tabular
from paper_scale_ard import effective_support


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "claim3_table3"
SEEDS = tuple(range(10))
PAPER_REDUCTIONS = {"Energy": 32.07, "Carbon": 69.91, "Protein": 45.12}
PAPER_SAMPLE_ESS = {"Energy": 91.2, "Carbon": 92.0, "Protein": 90.2}
ROUTES = {"Energy": 5, "Carbon": 6, "Protein": 7}
SCHEMES = {"Energy": "em", "Carbon": "em", "Protein": "mackay"}


@dataclass(frozen=True)
class Protocol:
    damping: float = 0.005
    warmup: int = 300
    noise_update_every: int = 5
    clip_min: float = 1e-3
    clip_max: float = 1e2
    init: float = 0.1
    tolerance: float = 1e-6
    patience: int = 5
    max_iter: int = 5000


CONFIG = Protocol()


@dataclass
class Fit:
    coef: np.ndarray
    gamma: np.ndarray
    noise_var: np.ndarray
    iterations: int
    converged: bool
    stationarity_error: float


def fit(
    X: np.ndarray,
    y: np.ndarray,
    *,
    heteroscedastic: bool,
    scheme: str,
    config: Protocol = CONFIG,
) -> Fit:
    """Algorithms 1/2 with the arithmetic EMA and safeguards in Appendix D.1."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, d = X.shape
    gamma = np.full(d, config.init)
    noise_var = np.full(n, config.init)
    xtx, xty = X.T @ X, X.T @ y
    patience_count = 0
    converged = False
    stationarity_error = float("inf")

    for iteration in range(1, config.max_iter + 1):
        weights = 1.0 / noise_var
        if heteroscedastic:
            gram = X.T @ (weights[:, None] * X)
            rhs = X.T @ (weights * y)
        else:
            gram = weights[0] * xtx
            rhs = weights[0] * xty
        factor = cho_factor(np.diag(gamma) + gram, lower=True, check_finite=False)
        coef = cho_solve(factor, rhs, check_finite=False)
        covariance = cho_solve(factor, np.eye(d), check_finite=False)
        covariance_diag = np.maximum(np.diag(covariance), 0.0)

        if scheme == "em":
            gamma_new = 1.0 / np.maximum(coef**2 + covariance_diag, config.clip_min)
        elif scheme == "mackay":
            degrees = np.clip(1.0 - gamma * covariance_diag, config.clip_min, 1.0)
            gamma_new = degrees / np.maximum(coef**2, config.clip_min)
        else:
            raise ValueError(f"unknown scheme: {scheme}")
        gamma_new = np.clip(gamma_new, config.clip_min, config.clip_max)
        old_gamma = gamma.copy()
        gamma = (1.0 - config.damping) * gamma + config.damping * gamma_new

        old_noise = noise_var.copy()
        update_noise = (
            not heteroscedastic
            or (
                iteration > config.warmup
                and (iteration - config.warmup) % config.noise_update_every == 0
            )
        )
        if update_noise:
            residual = y - X @ coef
            if heteroscedastic:
                leverage = np.einsum(
                    "ij,jk,ik->i", X, covariance, X, optimize=True
                )
                noise_new = residual**2 + leverage
            else:
                mean_leverage = float(np.sum(covariance * xtx) / n)
                noise_new = np.full(n, np.mean(residual**2) + mean_leverage)
            noise_new = np.clip(noise_new, config.clip_min, config.clip_max)
            noise_var = (
                (1.0 - config.damping) * noise_var
                + config.damping * noise_new
            )

        gamma_delta = np.max(np.abs(np.log(gamma) - np.log(old_gamma))) / (
            1.0 + np.max(np.abs(np.log(gamma)))
        )
        noise_delta = np.max(np.abs(np.log(noise_var) - np.log(old_noise))) / (
            1.0 + np.max(np.abs(np.log(noise_var)))
        )
        if iteration > config.warmup and max(gamma_delta, noise_delta) < config.tolerance:
            patience_count += 1
            if patience_count >= config.patience:
                converged = True
                break
        else:
            patience_count = 0

    # Algorithm 1/2 requires a final posterior recomputation.
    weights = 1.0 / noise_var
    if heteroscedastic:
        gram = X.T @ (weights[:, None] * X)
        rhs = X.T @ (weights * y)
    else:
        gram = weights[0] * xtx
        rhs = weights[0] * xty
    factor = cho_factor(np.diag(gamma) + gram, lower=True, check_finite=False)
    coef = cho_solve(factor, rhs, check_finite=False)
    covariance_diag = np.diag(
        cho_solve(factor, np.eye(d), check_finite=False)
    )
    if scheme == "em":
        target = 1.0 / np.maximum(coef**2 + covariance_diag, config.clip_min)
        stationarity_error = float(
            np.max(np.abs(np.log(gamma) - np.log(np.clip(target, config.clip_min, config.clip_max))))
        )
    else:
        degrees = np.clip(1.0 - gamma * covariance_diag, config.clip_min, 1.0)
        stationarity_error = float(
            np.max(np.abs(gamma * coef**2 - degrees))
        )
    return Fit(coef, gamma, noise_var, iteration, converged, stationarity_error)


def rmse(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.sqrt(np.mean((prediction - target) ** 2)))


def bootstrap_reduction_ci(joint: np.ndarray, model: np.ndarray) -> tuple[float, float]:
    rng = np.random.default_rng(260529908)
    indices = rng.integers(0, len(joint), size=(20_000, len(joint)))
    jmeans = joint[indices].mean(axis=1)
    mmeans = model[indices].mean(axis=1)
    reductions = 100.0 * (mmeans - jmeans) / mmeans
    return tuple(map(float, np.quantile(reductions, [0.025, 0.975])))


def bootstrap_mean_ci(values: np.ndarray) -> tuple[float, float]:
    rng = np.random.default_rng(260529908)
    indices = rng.integers(0, len(values), size=(20_000, len(values)))
    return tuple(map(float, np.quantile(values[indices].mean(axis=1), [0.025, 0.975])))


def run_dataset(name: str) -> tuple[list[dict], dict]:
    route = ROUTES[name]
    scheme = SCHEMES[name]
    X, y, source = load_route_dataset(route)
    rows: list[dict] = []
    for seed in SEEDS:
        phi, phi_test, target, y_test, scaler, rff_gamma = _prepare_tabular(X, y, seed)
        rng = np.random.default_rng(seed + 1000 * route + 10)
        contaminated = target.copy()
        k = int(np.floor(0.1 * len(target)))
        outliers = rng.choice(len(target), k, replace=False)
        signs = rng.choice(np.array([-1.0, 1.0]), k)
        contaminated[outliers] += 3.0 * signs * rng.normal(1.0, 0.25, k)

        started = time.monotonic()
        joint = fit(phi, contaminated, heteroscedastic=True, scheme=scheme)
        model = fit(phi, contaminated, heteroscedastic=False, scheme=scheme)
        elapsed = time.monotonic() - started

        def original_scale(values: np.ndarray) -> np.ndarray:
            return scaler.inverse_transform(values.reshape(-1, 1)).ravel()

        joint_rmse = rmse(original_scale(phi_test @ joint.coef), y_test)
        model_rmse = rmse(original_scale(phi_test @ model.coef), y_test)
        row = {
            "dataset": name,
            "seed": seed,
            "scheme": scheme,
            "joint_rmse": joint_rmse,
            "model_only_rmse": model_rmse,
            "paired_reduction_percent": 100.0 * (model_rmse - joint_rmse) / model_rmse,
            "weight_ess_percent": 100.0 * effective_support(1.0 / joint.gamma),
            "sample_ess_percent": 100.0 * effective_support(1.0 / joint.noise_var),
            "joint_iterations": joint.iterations,
            "model_iterations": model.iterations,
            "joint_converged": joint.converged,
            "model_converged": model.converged,
            "joint_stationarity_error": joint.stationarity_error,
            "model_stationarity_error": model.stationarity_error,
            "rff_gamma": rff_gamma,
            "runtime_seconds": elapsed,
        }
        rows.append(row)
        print("CLAIM3_ROW " + json.dumps(row, sort_keys=True), flush=True)

    joint_values = np.array([row["joint_rmse"] for row in rows])
    model_values = np.array([row["model_only_rmse"] for row in rows])
    observed = float(100.0 * (model_values.mean() - joint_values.mean()) / model_values.mean())
    ci_low, ci_high = bootstrap_reduction_ci(joint_values, model_values)
    target = PAPER_REDUCTIONS[name]
    sample_ess = np.array([row["sample_ess_percent"] for row in rows])
    ess_low, ess_high = bootstrap_mean_ci(sample_ess)
    ess_target = PAPER_SAMPLE_ESS[name]
    summary = {
        "dataset": name,
        "scheme": scheme,
        "paper_reduction_percent": target,
        "observed_reduction_percent": observed,
        "bootstrap_95ci_percent": [ci_low, ci_high],
        "paper_value_in_bootstrap_ci": bool(ci_low <= target <= ci_high),
        "paper_sample_ess_percent": ess_target,
        "sample_ess_bootstrap_95ci_percent": [ess_low, ess_high],
        "paper_sample_ess_in_bootstrap_ci": bool(ess_low <= ess_target <= ess_high),
        "mean_joint_rmse": float(joint_values.mean()),
        "mean_model_only_rmse": float(model_values.mean()),
        "joint_wins": int(np.sum(joint_values < model_values)),
        "mean_weight_ess_percent": float(np.mean([row["weight_ess_percent"] for row in rows])),
        "mean_sample_ess_percent": float(np.mean([row["sample_ess_percent"] for row in rows])),
        "all_joint_converged": bool(all(row["joint_converged"] for row in rows)),
        "all_model_converged": bool(all(row["model_converged"] for row in rows)),
        "all_finite": bool(np.isfinite(
            [[row["joint_rmse"], row["model_only_rmse"]] for row in rows]
        ).all()),
        "source": source,
    }
    print("CLAIM3_DATASET_SUMMARY " + json.dumps(summary, sort_keys=True), flush=True)
    return rows, summary


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    all_rows: list[dict] = []
    summaries = []
    for name in ("Energy", "Carbon", "Protein"):
        rows, summary = run_dataset(name)
        all_rows.extend(rows)
        summaries.append(summary)
    payload = {
        "paper": "arXiv:2605.29908",
        "claim": 3,
        "protocol": asdict(CONFIG),
        "seeds": list(SEEDS),
        "paper_source_sha256": "0a3ed827e899a92f2ed3b1e877ea2f6456ec3d827af184045acac3b8ebd4cb49",
        "git_sha": __import__("subprocess").check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "cpu": platform.processor(),
        "elapsed_seconds": time.monotonic() - started,
        "rows": all_rows,
        "summaries": summaries,
        "integrity_pass": bool(
            len(all_rows) == 30
            and all(summary["all_finite"] for summary in summaries)
            and all(0.0 < row["sample_ess_percent"] <= 100.0 for row in all_rows)
        ),
        "assessment": (
            "VERIFIED"
            if all(
                summary["paper_value_in_bootstrap_ci"]
                and summary["paper_sample_ess_in_bootstrap_ci"]
                and summary["all_joint_converged"]
                and summary["all_model_converged"]
                and 2.0 <= summary["mean_weight_ess_percent"] <= 45.0
                for summary in summaries
            )
            else "BLOCKED"
        ),
        "assessment_rule": (
            "VERIFIED only if every paper reduction and sample-ESS value lies inside "
            "its paired-seed 95% bootstrap interval, mean weight ESS is 2--45%, all "
            "fits meet the paper stopping rule, and integrity passes; otherwise "
            "BLOCKED because unreleased RFF/bandwidth/safeguard choices remain."
        ),
    }
    output = OUT / "results.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    payload["results_sha256"] = hashlib.sha256(output.read_bytes()).hexdigest()
    print("CLAIM3_FINAL " + json.dumps(payload, sort_keys=True), flush=True)
    return 0 if payload["integrity_pass"] else 2


if __name__ == "__main__":
    with threadpool_limits(limits=8):
        raise SystemExit(main())
