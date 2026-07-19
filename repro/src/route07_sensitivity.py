"""Retain the adverse Protein result and test paper-authorized EM schedules.

This is a sensitivity subcheck inside frozen approach 7, not another approach.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from claim1_datasets import load_route_dataset
from claim1_paper_scale import OUT, _prepare_tabular, _rmse
from paper_scale_ard import effective_support, fit_sbl


def main():
    X, y, source = load_route_dataset(7)
    phi, phi_test, target, y_test, y_scaler, _ = _prepare_tabular(X, y, 0)
    rng = np.random.default_rng(7010)
    count = int(0.1 * len(target))
    outliers = rng.choice(len(target), count, replace=False)
    contaminated = target.copy()
    contaminated[outliers] += (
        3.0 * rng.choice(np.array([-1.0, 1.0]), count) * rng.normal(1.0, 0.25, count)
    )

    def rmse(fit):
        pred = y_scaler.inverse_transform((phi_test @ fit.coef).reshape(-1, 1)).ravel()
        return _rmse(pred, y_test)

    baseline = fit_sbl(phi, contaminated, heteroscedastic=False, max_iter=90, warmup=0)
    schedules = []
    for max_iter, warmup, update_every in ((80, 50, 3), (90, 70, 5), (120, 100, 5)):
        fit = fit_sbl(
            phi, contaminated, heteroscedastic=True, max_iter=max_iter,
            warmup=warmup, noise_update_every=update_every,
        )
        selected = np.argpartition(fit.noise_var, -count)[-count:]
        schedules.append({
            "max_iter": max_iter, "warmup": warmup,
            "noise_update_every": update_every, "rmse": rmse(fit),
            "data_ess": effective_support(1.0 / fit.noise_var),
            "weight_ess": effective_support(1.0 / fit.gamma),
            "outlier_recall": len(set(selected).intersection(map(int, outliers))) / count,
            "converged": fit.converged,
        })
    result = {
        "route": 7, "approach": "Protein", "kind": "within-route sensitivity",
        "adds_approach": False, "seed": 0, "model_only_rmse": rmse(baseline),
        "source": source, "schedules": schedules,
        "conclusion": "Adverse predictive result persists across Appendix-D.1 schedules.",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "route07_sensitivity.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
