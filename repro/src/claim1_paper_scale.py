"""Execute the frozen exactly-ten paper-scale approaches for Claim 1."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.stats import binomtest, wilcoxon
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import HuberRegressor, RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from claim1_datasets import load_route_dataset
from paper_scale_ard import effective_support, fit_sbl


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "claim1_paper_scale"
SEEDS = tuple(range(10))
ROUTES = tuple(range(1, 11))
DATASET_NAMES = {
    2: "Boston", 3: "Yacht", 4: "Concrete", 5: "Energy", 6: "Carbon",
    7: "Protein", 8: "Power", 9: "Kin8nm", 10: "Elevators",
}


def _rmse(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.sqrt(np.mean((prediction - target) ** 2)))


def _top_k_recall(scores: np.ndarray, truth: np.ndarray) -> float:
    k = len(truth)
    if k == 0:
        return float("nan")
    selected = np.argpartition(scores, -k)[-k:]
    return float(len(set(selected).intersection(map(int, truth))) / k)


def _rff_design(X_train: np.ndarray, X_test: np.ndarray, seed: int):
    scaler = StandardScaler().fit(X_train)
    train_scaled = scaler.transform(X_train)
    test_scaled = scaler.transform(X_test)
    rng = np.random.default_rng(seed + 70_000)
    probe = train_scaled[rng.choice(len(train_scaled), min(len(train_scaled), 512), replace=False)]
    pairs = rng.integers(0, len(probe), size=(2048, 2))
    squared = np.sum((probe[pairs[:, 0]] - probe[pairs[:, 1]]) ** 2, axis=1)
    positive = squared[squared > 1e-12]
    gamma = 1.0 / max(float(np.median(positive)), 1e-6)
    rff = RBFSampler(gamma=gamma, n_components=256, random_state=seed)
    phi_train = rff.fit_transform(train_scaled)
    phi_test = rff.transform(test_scaled)
    # Section 5.2 standardizes the input features before the RFF mapping.  It
    # does not specify a second column-wise standardization of the correlated
    # RFF design, so preserve the RBF approximation's native scaling.
    return phi_train, phi_test, gamma


def _prepare_tabular(X: np.ndarray, y: np.ndarray, seed: int):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )
    rng = np.random.default_rng(seed + 50_000)
    if len(X_train) > 2000:
        chosen = rng.choice(len(X_train), 2000, replace=False)
        X_train, y_train = X_train[chosen], y_train[chosen]
    y_scaler = StandardScaler().fit(y_train.reshape(-1, 1))
    y_train_std = y_scaler.transform(y_train.reshape(-1, 1)).ravel()
    phi_train, phi_test, gamma = _rff_design(X_train, X_test, seed)
    return phi_train, phi_test, y_train_std, y_test, y_scaler, gamma


def _fit_tabular_condition(route: int, seed: int, contamination: float, X, y) -> dict:
    phi, phi_test, target, y_test, y_scaler, rff_gamma = _prepare_tabular(X, y, seed)
    rng = np.random.default_rng(seed + 1000 * route + int(100 * contamination))
    contaminated = target.copy()
    k = int(np.floor(contamination * len(target)))
    outliers = rng.choice(len(target), k, replace=False) if k else np.array([], dtype=int)
    if k:
        signs = rng.choice(np.array([-1.0, 1.0]), k)
        magnitudes = rng.normal(1.0, 0.25, k)
        contaminated[outliers] += 3.0 * signs * magnitudes

    joint = fit_sbl(phi, contaminated, heteroscedastic=True)
    model_only = fit_sbl(phi, contaminated, heteroscedastic=False)
    ridge = RidgeCV(alphas=np.logspace(-4, 4, 17)).fit(phi, contaminated)
    # RFF columns are standardized and numerous relative to Boston/Yacht;
    # alpha=100 prevents the nominally robust baseline from interpolating the
    # contaminated training set (a failure seen with sklearn's 1e-4 default).
    huber = HuberRegressor(epsilon=1.35, alpha=100.0, max_iter=500).fit(phi, contaminated)

    def original_scale(pred):
        return y_scaler.inverse_transform(np.asarray(pred).reshape(-1, 1)).ravel()

    return {
        "route": route, "dataset": DATASET_NAMES[route], "seed": seed,
        "condition": "contaminated_10pct" if contamination else "clean",
        "original_rows": len(X), "train_rows": len(phi), "test_rows": len(phi_test),
        "raw_features": X.shape[1], "rff_features": phi.shape[1], "rff_gamma": rff_gamma,
        "outliers": k,
        "rmse_joint": _rmse(original_scale(phi_test @ joint.coef), y_test),
        "rmse_model_only": _rmse(original_scale(phi_test @ model_only.coef), y_test),
        "rmse_ridge": _rmse(original_scale(ridge.predict(phi_test)), y_test),
        "rmse_huber": _rmse(original_scale(huber.predict(phi_test)), y_test),
        "weight_ess_joint": effective_support(1.0 / joint.gamma),
        "data_ess_joint": effective_support(1.0 / joint.noise_var),
        "outlier_recall_joint": _top_k_recall(joint.noise_var, outliers),
        "joint_iterations": joint.iterations, "joint_converged": joint.converged,
        "model_only_iterations": model_only.iterations,
    }


def run_tabular_route(route: int) -> tuple[list[dict], dict, dict]:
    X, y, source = load_route_dataset(route)
    rows = []
    for seed in SEEDS:
        for contamination in (0.0, 0.1):
            rows.append(_fit_tabular_condition(route, seed, contamination, X, y))
    dirty = [r for r in rows if r["condition"] == "contaminated_10pct"]
    gains = np.array([r["rmse_model_only"] - r["rmse_joint"] for r in dirty])
    wins = int(np.sum(gains > 0))
    p_value = float(binomtest(wins, len(dirty), 0.5, alternative="greater").pvalue)
    summary = {
        "route": route, "approach": DATASET_NAMES[route], "state": "complete",
        "trials": len(SEEDS), "conditions": 2, "result_rows": len(rows),
        "mean_rmse_joint_contaminated": float(np.mean([r["rmse_joint"] for r in dirty])),
        "mean_rmse_model_only_contaminated": float(np.mean([r["rmse_model_only"] for r in dirty])),
        "joint_wins": wins, "one_sided_sign_test_p": p_value,
        "mean_outlier_recall": float(np.mean([r["outlier_recall_joint"] for r in dirty])),
        "chance_outlier_recall": 0.1,
        "mean_weight_ess": float(np.mean([r["weight_ess_joint"] for r in dirty])),
        "mean_data_ess": float(np.mean([r["data_ess_joint"] for r in dirty])),
        "robustness_supported": bool(np.mean(gains) > 0 and wins >= 6),
        "outlier_signal_supported": bool(np.mean([r["outlier_recall_joint"] for r in dirty]) > 0.2),
        "sparsity_supported": bool(np.mean([r["weight_ess_joint"] for r in dirty]) < 0.8),
    }
    summary["claim_supported"] = bool(
        summary["robustness_supported"] and summary["outlier_signal_supported"]
        and summary["sparsity_supported"]
    )
    return rows, summary, source


def _synthetic_problem(seed, support_fraction, sigma, rho, multiplier):
    rng = np.random.default_rng(seed)
    n, d, ntest = 500, 50, 1000
    support = rng.choice(d, int(round(support_fraction * d)), replace=False)
    coef = np.zeros(d)
    coef[support] = rng.normal(size=len(support))
    X = rng.normal(size=(n, d))
    Xtest = rng.normal(size=(ntest, d))
    outliers = rng.choice(n, int(np.floor(rho * n)), replace=False)
    scales = np.full(n, sigma)
    scales[outliers] *= multiplier
    y = X @ coef + rng.normal(size=n) * scales
    ytest = Xtest @ coef + rng.normal(size=ntest) * sigma
    return X, y, Xtest, ytest, support, outliers


def run_synthetic_route() -> tuple[list[dict], dict]:
    rows = []
    # Appendix D.2's two heatmaps, frozen to explicit three-by-three grids.
    grids = []
    for support_fraction in (0.1, 0.2, 0.4):
        for sigma in (0.1, 0.2, 0.5):
            grids.append(("weight_grid", support_fraction, sigma, 0.2, 10.0))
    for rho in (0.05, 0.1, 0.2):
        for multiplier in (2.0, 5.0, 10.0):
            grids.append(("data_grid", 0.2, 0.2, rho, multiplier))
    for grid, support_fraction, sigma, rho, multiplier in grids:
        for seed in SEEDS:
            X, y, Xtest, ytest, support, outliers = _synthetic_problem(
                seed, support_fraction, sigma, rho, multiplier
            )
            joint = fit_sbl(X, y, heteroscedastic=True, max_iter=80, warmup=50)
            model_only = fit_sbl(X, y, heteroscedastic=False, max_iter=80, warmup=0)
            rows.append({
                "route": 1, "dataset": "synthetic", "grid": grid, "seed": seed,
                "n": 500, "d": 50, "ntest": 1000,
                "support_fraction": support_fraction, "sigma": sigma,
                "contamination_fraction": rho, "noise_multiplier": multiplier,
                "rmse_joint": _rmse(Xtest @ joint.coef, ytest),
                "rmse_model_only": _rmse(Xtest @ model_only.coef, ytest),
                "weight_recall_joint": _top_k_recall(1.0 / joint.gamma, support),
                "outlier_recall_joint": _top_k_recall(joint.noise_var, outliers),
                "weight_ess_joint": effective_support(1.0 / joint.gamma),
                "data_ess_joint": effective_support(1.0 / joint.noise_var),
            })
    gains = np.array([r["rmse_model_only"] - r["rmse_joint"] for r in rows])
    try:
        paired_p = float(wilcoxon(gains, alternative="greater").pvalue)
    except ValueError:
        paired_p = 1.0
    summary = {
        "route": 1, "approach": "paper-scale synthetic grids", "state": "complete",
        "trials_per_cell": len(SEEDS), "grid_cells": len(grids), "result_rows": len(rows),
        "n": 500, "d": 50, "ntest": 1000,
        "mean_weight_recall": float(np.mean([r["weight_recall_joint"] for r in rows if r["grid"] == "weight_grid"])),
        "mean_outlier_recall": float(np.mean([r["outlier_recall_joint"] for r in rows if r["grid"] == "data_grid"])),
        "mean_rmse_gain": float(np.mean(gains)), "paired_wilcoxon_p": paired_p,
        "claim_supported": bool(np.mean(gains) > 0),
    }
    return rows, summary


def _write_csv(path: Path, rows: list[dict]):
    if not rows:
        raise ValueError("refusing to write an empty result")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def execute(routes=ROUTES):
    routes = tuple(routes)
    if routes != ROUTES:
        raise ValueError(f"final execution requires routes {ROUTES}, got {routes}")
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.time()
    synthetic_rows, synthetic_summary = run_synthetic_route()
    _write_csv(OUT / "route01_synthetic.csv", synthetic_rows)
    summaries = [synthetic_summary]
    sources = []
    for route in range(2, 11):
        rows, summary, source = run_tabular_route(route)
        _write_csv(OUT / f"route{route:02d}_{DATASET_NAMES[route].lower()}.csv", rows)
        summaries.append(summary)
        sources.append(source)
        print(json.dumps(summary, sort_keys=True), flush=True)

    route_numbers = [s["route"] for s in summaries]
    if route_numbers != list(ROUTES) or len(set(route_numbers)) != 10 or max(route_numbers) > 10:
        raise RuntimeError(f"exactly-ten invariant failed: {route_numbers}")
    result_files = sorted(OUT.glob("route*.csv"))
    overall = {
        "paper": "Joint Model and Data Sparsification via the Marginal Likelihood",
        "openreview_id": "vMcu1h3fOV", "claim": 1,
        "approaches_executed": len(summaries), "approach_numbers": route_numbers,
        "approaches_supported": int(sum(s["claim_supported"] for s in summaries)),
        "approaches_adverse_or_mixed": int(sum(not s["claim_supported"] for s in summaries)),
        "exactly_ten_invariant": True, "route_11_executed": False,
        "total_result_rows": int(sum(s["result_rows"] for s in summaries)),
        "elapsed_seconds": time.time() - started,
        "source_datasets": sources, "approaches": summaries,
        "result_file_sha256": {p.name: _sha256(p) for p in result_files},
    }
    (OUT / "validation.json").write_text(json.dumps(overall, indent=2, sort_keys=True) + "\n")
    print(json.dumps(overall, indent=2, sort_keys=True))
    return overall


def main():
    parser = argparse.ArgumentParser()
    parser.parse_args()
    execute()


if __name__ == "__main__":
    main()
