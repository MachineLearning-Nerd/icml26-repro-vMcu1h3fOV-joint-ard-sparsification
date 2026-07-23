#!/usr/bin/env python3
"""Faithful Figure 3 grids and real-Boston Figure 4 reproduction."""
from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import json
import platform
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parent))
from claim1_datasets import load_route_dataset
from claim3_table3 import Protocol, fit as fit_paper_sbl
from verify_joint_ard_robust_scale import fit, upd_studentt


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "claim4_faithful"
FIG3_INPUT = ROOT / "outputs" / "claim1_paper_scale" / "route01_synthetic.csv"
SEEDS = tuple(range(20))
LEVELS = (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30)
PAPER_L2_RMSE = {
    0.0: 3.303,
    0.05: 3.521,
    0.10: 3.713,
    0.15: 4.126,
    0.20: 4.284,
    0.25: 4.679,
    0.30: 5.216,
}
NONINFERIORITY_RMSE = 0.5
HUBER_THRESHOLD = 1.35
HUBER_RIDGE = 1e-4
HUBER_TOLERANCE = 1e-8
HUBER_MAX_ITER = 500
RVM_PROTOCOL = Protocol(
    damping=0.02,
    warmup=50,
    noise_update_every=2,
    clip_min=1e-6,
    clip_max=1e6,
    init=0.1,
    tolerance=1e-6,
    patience=5,
    max_iter=1500,
)


@dataclass(frozen=True)
class HuberIRLSFit:
    coef: np.ndarray
    intercept: float
    iterations: int
    converged: bool
    relative_step: float
    stationarity_error: float
    objective: float


def rmse(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.sqrt(np.mean((prediction - target) ** 2)))


def rbf_design(
    X_train: np.ndarray, X_test: np.ndarray, X_centers: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    """RVM basis with all n covariates as centers, matching paper d=n."""
    center_squared = np.sum(
        (X_centers[:, None, :] - X_centers[None, :, :]) ** 2, axis=2
    )
    positive = center_squared[np.triu_indices_from(center_squared, k=1)]
    positive = positive[positive > 1e-12]
    lengthscale = float(np.sqrt(np.median(positive)))
    train_squared = np.sum(
        (X_train[:, None, :] - X_centers[None, :, :]) ** 2, axis=2
    )
    test_squared = np.sum(
        (X_test[:, None, :] - X_centers[None, :, :]) ** 2, axis=2
    )
    return (
        np.exp(-train_squared / (2.0 * lengthscale**2)),
        np.exp(-test_squared / (2.0 * lengthscale**2)),
        lengthscale,
    )


def fit_huber_irls(
    design: np.ndarray,
    target: np.ndarray,
    threshold: float = HUBER_THRESHOLD,
    ridge: float = HUBER_RIDGE,
    tolerance: float = HUBER_TOLERANCE,
    max_iter: int = HUBER_MAX_ITER,
) -> HuberIRLSFit:
    """Solve convex ridge-Huber regression with an unpenalized intercept."""
    augmented = np.column_stack([design, np.ones(len(design))])
    penalty = np.zeros(augmented.shape[1])
    penalty[:-1] = ridge

    def solve(weights: np.ndarray) -> np.ndarray:
        gram = (augmented.T * weights) @ augmented
        gram.flat[:: gram.shape[0] + 1] += penalty
        rhs = augmented.T @ (weights * target)
        factor = cho_factor(gram, lower=True, check_finite=False)
        return cho_solve(factor, rhs, check_finite=False)

    def diagnostics(parameters: np.ndarray) -> tuple[float, float]:
        residual = target - augmented @ parameters
        absolute = np.abs(residual)
        loss = np.where(
            absolute <= threshold,
            0.5 * residual**2,
            threshold * (absolute - 0.5 * threshold),
        )
        objective = float(
            np.sum(loss) + 0.5 * ridge * np.dot(parameters[:-1], parameters[:-1])
        )
        gradient = -(augmented.T @ np.clip(residual, -threshold, threshold))
        gradient[:-1] += ridge * parameters[:-1]
        scale = 1.0 + np.linalg.norm(augmented.T @ target, ord=np.inf)
        stationarity = float(np.linalg.norm(gradient, ord=np.inf) / scale)
        return objective, stationarity

    parameters = solve(np.ones(len(target)))
    relative_step = float("inf")
    objective, stationarity = diagnostics(parameters)
    converged = False
    iterations = 0
    for iterations in range(1, max_iter + 1):
        residual = target - augmented @ parameters
        weights = np.minimum(
            1.0, threshold / np.maximum(np.abs(residual), np.finfo(float).eps)
        )
        updated = solve(weights)
        relative_step = float(
            np.linalg.norm(updated - parameters) /
            (1.0 + np.linalg.norm(parameters))
        )
        parameters = updated
        objective, stationarity = diagnostics(parameters)
        if relative_step <= tolerance and stationarity <= tolerance:
            converged = True
            break
    return HuberIRLSFit(
        coef=parameters[:-1],
        intercept=float(parameters[-1]),
        iterations=iterations,
        converged=converged,
        relative_step=relative_step,
        stationarity_error=stationarity,
        objective=objective,
    )


def top_k_recall(scores: np.ndarray, truth: np.ndarray) -> float:
    if len(truth) == 0:
        return float("nan")
    selected = np.argpartition(scores, -len(truth))[-len(truth):]
    return float(len(set(map(int, selected)).intersection(map(int, truth))) / len(truth))


def one_trial(X: np.ndarray, y: np.ndarray, source: dict, seed: int, level: float) -> dict:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )
    x_scaler = StandardScaler().fit(X_train)
    X_train = x_scaler.transform(X_train)
    X_test = x_scaler.transform(X_test)
    X_centers = x_scaler.transform(X)
    y_scaler = StandardScaler().fit(y_train.reshape(-1, 1))
    y_clean = y_scaler.transform(y_train.reshape(-1, 1)).ravel()
    phi_train, phi_test, lengthscale = rbf_design(
        X_train, X_test, X_centers
    )

    rng = np.random.default_rng(seed + 260_529_080 + int(level * 100))
    contaminated = y_clean.copy()
    n_outliers = int(np.floor(level * len(contaminated)))
    outliers = (
        rng.choice(len(contaminated), n_outliers, replace=False)
        if n_outliers
        else np.array([], dtype=int)
    )
    if n_outliers:
        signs = rng.choice(np.array([-1.0, 1.0]), n_outliers)
        contaminated[outliers] += 3.0 * signs * rng.normal(1.0, 0.25, n_outliers)

    predictions = {}
    noise = {}
    fitted = {}
    for name, heteroscedastic in (
        ("homoscedastic", False),
        ("joint", True),
    ):
        fitted[name] = fit_paper_sbl(
            phi_train,
            contaminated,
            heteroscedastic=heteroscedastic,
            scheme="em",
            config=RVM_PROTOCOL,
        )
        predictions[name] = phi_test @ fitted[name].coef
        noise[name] = fitted[name].noise_var
    student_coef, student_noise = fit(
        phi_train, contaminated, upd_studentt, return_lam=True
    )
    predictions["student_t"] = phi_test @ student_coef
    noise["student_t"] = student_noise
    huber = fit_huber_irls(phi_train, contaminated)
    predictions["huber"] = phi_test @ huber.coef + huber.intercept

    def original(values: np.ndarray) -> np.ndarray:
        return y_scaler.inverse_transform(values.reshape(-1, 1)).ravel()

    row = {
        "dataset": "Boston",
        "dataset_sha256": source["canonical_matrix_sha256"],
        "seed": seed,
        "contamination_fraction": level,
        "original_rows": len(X),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "kernel_basis_columns": phi_train.shape[1],
        "lengthscale": lengthscale,
        "rmse_homoscedastic": rmse(original(predictions["homoscedastic"]), y_test),
        "rmse_joint": rmse(original(predictions["joint"]), y_test),
        "rmse_student_t": rmse(original(predictions["student_t"]), y_test),
        "rmse_huber": rmse(original(predictions["huber"]), y_test),
        "joint_outlier_recall": top_k_recall(noise["joint"], outliers),
        "homoscedastic_noise_cv": float(
            np.std(noise["homoscedastic"]) / np.mean(noise["homoscedastic"])
        ),
        "joint_iterations": fitted["joint"].iterations,
        "joint_converged": fitted["joint"].converged,
        "homoscedastic_iterations": fitted["homoscedastic"].iterations,
        "homoscedastic_converged": fitted["homoscedastic"].converged,
        "huber_iterations": huber.iterations,
        "huber_converged": huber.converged,
        "huber_relative_step": huber.relative_step,
        "huber_stationarity_error": huber.stationarity_error,
        "huber_objective": huber.objective,
    }
    print("CLAIM4_BOSTON_ROW " + json.dumps(row, sort_keys=True), flush=True)
    return row


def bootstrap_gap_ci(rows: list[dict], method: str) -> tuple[float, float]:
    joint = np.array([row["rmse_joint"] for row in rows])
    other = np.array([row[f"rmse_{method}"] for row in rows])
    rng = np.random.default_rng(260_529_084)
    indices = rng.integers(0, len(rows), size=(20_000, len(rows)))
    gaps = (joint[indices] - other[indices]).mean(axis=1)
    return tuple(map(float, np.quantile(gaps, [0.025, 0.975])))


def summarize_boston(rows: list[dict]) -> list[dict]:
    summaries = []
    for level in LEVELS:
        group = [row for row in rows if row["contamination_fraction"] == level]
        means = {
            method: float(np.mean([row[f"rmse_{method}"] for row in group]))
            for method in ("homoscedastic", "joint", "student_t", "huber")
        }
        st_ci = bootstrap_gap_ci(group, "student_t")
        huber_ci = bootstrap_gap_ci(group, "huber")
        recall = [row["joint_outlier_recall"] for row in group]
        summary = {
            "contamination_fraction": level,
            "trials": len(group),
            "mean_rmse": means,
            "paper_l2_irls_rmse": PAPER_L2_RMSE[level],
            "joint_minus_student_t_95ci": list(st_ci),
            "joint_minus_huber_95ci": list(huber_ci),
            "joint_noninferior_student_t": bool(st_ci[1] <= NONINFERIORITY_RMSE),
            "joint_noninferior_huber": bool(huber_ci[1] <= NONINFERIORITY_RMSE),
            "joint_beats_homoscedastic": bool(
                means["joint"] < means["homoscedastic"]
            ),
            "mean_joint_outlier_recall": (
                float(np.nanmean(recall)) if level else None
            ),
        }
        print("CLAIM4_BOSTON_SUMMARY " + json.dumps(summary, sort_keys=True), flush=True)
        summaries.append(summary)
    return summaries


def load_fig3() -> tuple[list[dict], dict]:
    if not FIG3_INPUT.exists():
        raise FileNotFoundError(
            "Figure-3 input must be regenerated earlier in the fixed campaign"
        )
    with FIG3_INPUT.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    numeric = {
        "seed",
        "support_fraction",
        "sigma",
        "contamination_fraction",
        "noise_multiplier",
        "weight_recall_joint",
        "weight_recall_model_only",
        "outlier_recall_joint",
    }
    for row in rows:
        for key in numeric:
            row[key] = float(row[key])

    weight = [row for row in rows if row["grid"] == "weight_grid"]
    data = [row for row in rows if row["grid"] == "data_grid"]
    weight_by_support = {}
    for support in (0.1, 0.2, 0.4):
        group = [row for row in weight if row["support_fraction"] == support]
        weight_by_support[str(support)] = float(np.mean([
            row["weight_recall_joint"] - row["weight_recall_model_only"]
            for row in group
        ]))
    data_by_contamination = {}
    for level in (0.05, 0.1, 0.2):
        group = [row for row in data if row["contamination_fraction"] == level]
        data_by_contamination[str(level)] = float(np.mean([
            row["outlier_recall_joint"] - row["contamination_fraction"]
            for row in group
        ]))
    summary = {
        "rows": len(rows),
        "n": 500,
        "d": 50,
        "trials_per_cell": 10,
        "weight_cells": 9,
        "data_cells": 9,
        "weight_recovery_gain_by_support": weight_by_support,
        "outlier_recovery_gain_by_contamination": data_by_contamination,
        "largest_weight_gain_at_high_sparsity": bool(
            np.mean([weight_by_support["0.1"], weight_by_support["0.2"]])
            >= weight_by_support["0.4"]
        ),
        "largest_outlier_gain_at_low_contamination": bool(
            np.mean([
                data_by_contamination["0.05"],
                data_by_contamination["0.1"],
            ]) >= data_by_contamination["0.2"]
        ),
        "category_definition": {
            "high_weight_sparsity": "support fraction 0.1 or 0.2 versus 0.4",
            "low_contamination": "rho 0.05 or 0.1 versus 0.2",
        },
    }
    print("CLAIM4_FIG3_SUMMARY " + json.dumps(summary, sort_keys=True), flush=True)
    return rows, summary


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    fig3_rows, fig3 = load_fig3()
    X, y, source = load_route_dataset(2)
    boston_rows = [
        one_trial(X, y, source, seed, level)
        for level in LEVELS
        for seed in SEEDS
    ]
    summaries = summarize_boston(boston_rows)
    write_csv(OUT / "boston_rvm_raw.csv", boston_rows)

    high_contamination = [
        item for item in summaries if item["contamination_fraction"] >= 0.10
    ]
    mechanism_rows = [
        row for row in boston_rows if row["contamination_fraction"] >= 0.10
    ]
    gates = {
        "real_boston_exact_schema": bool(
            source["dataset"] == "Boston"
            and source["rows"] == 506
            and all(row["train_rows"] == 404 for row in boston_rows)
            and all(row["test_rows"] == 102 for row in boston_rows)
        ),
        "twenty_trials_seven_levels": bool(
            len(boston_rows) == 140
            and all(item["trials"] == 20 for item in summaries)
        ),
        "joint_competitive_student_t": bool(
            all(item["joint_noninferior_student_t"] for item in summaries)
        ),
        "joint_competitive_huber": bool(
            all(item["joint_noninferior_huber"] for item in summaries)
        ),
        "joint_beats_homoscedastic_at_10pct_plus": bool(
            all(item["joint_beats_homoscedastic"] for item in high_contamination)
        ),
        "joint_outlier_recall_above_chance": bool(
            np.mean([
                row["joint_outlier_recall"]
                - row["contamination_fraction"]
                for row in mechanism_rows
            ]) >= 0.20
        ),
        "homoscedastic_negative_control_equal_noise": bool(
            max(row["homoscedastic_noise_cv"] for row in boston_rows) < 1e-12
        ),
        "huber_converged": bool(
            all(row["huber_converged"] for row in boston_rows)
        ),
        "fig3_exact_grid": bool(fig3["rows"] == 180),
        "fig3_sparse_low_contamination_pattern": bool(
            fig3["largest_weight_gain_at_high_sparsity"]
            and fig3["largest_outlier_gain_at_low_contamination"]
        ),
    }
    verified = all(gates.values())
    payload = {
        "paper": "arXiv:2605.29908",
        "claim": 4,
        "git_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "cpu": platform.processor(),
        "paper_source_sha256": "0a3ed827e899a92f2ed3b1e877ea2f6456ec3d827af184045acac3b8ebd4cb49",
        "protocol": {
            "dataset": source,
            "seeds": list(SEEDS),
            "levels": list(LEVELS),
            "test_fraction": 0.2,
            "kernel": "RBF RVM; all 506 covariates are centers (d=n), median scalar lengthscale, no embedded bias",
            "noninferiority_margin_rmse": NONINFERIORITY_RMSE,
            "huber_optimizer": {
                "solver": "convex IRLS with fixed standardized-target threshold",
                "threshold": HUBER_THRESHOLD,
                "ridge": HUBER_RIDGE,
                "tolerance": HUBER_TOLERANCE,
                "max_iter": HUBER_MAX_ITER,
                "intercept_penalized": False,
            },
            "rvm_optimizer": {
                "scheme": "l2-IRLS/EM-equivalent closed form",
                **RVM_PROTOCOL.__dict__,
            },
            "contamination": "paper Appendix D.2: 3*s*N(1,0.25^2)",
        },
        "fig3": fig3,
        "fig4": summaries,
        "gates": gates,
        "assessment": "VERIFIED" if verified else "BLOCKED",
        "limitations": [
            "The paper does not release its split seeds or fixed scalar RBF lengthscale.",
            "Following the paper's explicit d=n=506 statement uses all test covariates as transductive RBF centers, but never test targets; the scaler is fit on training covariates only.",
            "The paper used scikit-learn HuberRegressor, which did not terminate after 5,000 iterations on this rank-deficient design; this run solves a fixed-threshold convex Huber loss by IRLS and verifies its gradient stationarity.",
        ],
        "elapsed_seconds": time.monotonic() - started,
    }
    result_path = OUT / "results.json"
    result_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    payload["raw_csv_sha256"] = hashlib.sha256(
        (OUT / "boston_rvm_raw.csv").read_bytes()
    ).hexdigest()
    payload["results_sha256"] = hashlib.sha256(result_path.read_bytes()).hexdigest()
    print("CLAIM4_FINAL " + json.dumps(payload, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    with threadpool_limits(limits=8):
        raise SystemExit(main())
