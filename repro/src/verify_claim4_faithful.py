#!/usr/bin/env python3
"""Independent fail-closed checker for the Claim 4 evidence bundle."""
from __future__ import annotations

import copy
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "claim4_faithful"
EXPECTED_LEVELS = {0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30}
EXPECTED_DATASET_SHA = "fbad0e4a24f9e92621c27514aaeb6cac06f355038b2c4529315e66a670713b10"


def read_rows() -> list[dict]:
    with (OUT / "boston_rvm_raw.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    numeric = {
        "seed",
        "contamination_fraction",
        "original_rows",
        "train_rows",
        "test_rows",
        "kernel_basis_columns",
        "lengthscale",
        "rmse_homoscedastic",
        "rmse_joint",
        "rmse_student_t",
        "rmse_huber",
        "joint_outlier_recall",
        "homoscedastic_noise_cv",
        "huber_iterations",
    }
    for row in rows:
        for key in numeric:
            row[key] = float(row[key])
    return rows


def validate(payload: dict, rows: list[dict]) -> list[str]:
    errors = []
    levels = {row["contamination_fraction"] for row in rows}
    if len(rows) != 140 or levels != EXPECTED_LEVELS:
        errors.append("not exactly 20x7 Boston rows")
    if {row["dataset"] for row in rows} != {"Boston"}:
        errors.append("dataset name mismatch")
    if {row["dataset_sha256"] for row in rows} != {EXPECTED_DATASET_SHA}:
        errors.append("dataset hash mismatch")
    if any(
        row["original_rows"] != 506
        or row["train_rows"] != 404
        or row["test_rows"] != 102
        or row["kernel_basis_columns"] != 405
        for row in rows
    ):
        errors.append("schema or leakage invariant mismatch")
    if any(int(row["seed"]) not in range(20) for row in rows):
        errors.append("seed outside 0..19")
    if any(
        sum(
            row["contamination_fraction"] == level and int(row["seed"]) == seed
            for row in rows
        ) != 1
        for level in EXPECTED_LEVELS
        for seed in range(20)
    ):
        errors.append("missing or duplicate level/seed")
    if any(float(value) <= 0 for row in rows for key, value in row.items() if key.startswith("rmse_")):
        errors.append("nonpositive RMSE")
    if max(row["homoscedastic_noise_cv"] for row in rows) >= 1e-12:
        errors.append("homoscedastic negative control has unequal noise")
    if payload.get("assessment") != "VERIFIED" or not all(payload.get("gates", {}).values()):
        errors.append("claim contract is not VERIFIED")

    summaries = {
        item["contamination_fraction"]: item for item in payload.get("fig4", [])
    }
    for level in EXPECTED_LEVELS:
        group = [row for row in rows if row["contamination_fraction"] == level]
        if level not in summaries or len(group) != 20:
            errors.append(f"summary missing for {level}")
            continue
        for method in ("homoscedastic", "joint", "student_t", "huber"):
            observed = sum(row[f"rmse_{method}"] for row in group) / len(group)
            recorded = summaries[level]["mean_rmse"][method]
            if abs(observed - recorded) > 1e-12:
                errors.append(f"summary mismatch {level}/{method}")
    return errors


def main() -> int:
    payload = json.loads((OUT / "results.json").read_text())
    rows = read_rows()
    errors = validate(payload, rows)

    mutations = []
    missing_level = [row for row in rows if row["contamination_fraction"] != 0.15]
    mutations.append(bool(validate(payload, missing_level)))
    wrong_hash = copy.deepcopy(rows)
    wrong_hash[0]["dataset_sha256"] = "0" * 64
    mutations.append(bool(validate(payload, wrong_hash)))
    blocked = copy.deepcopy(payload)
    blocked["assessment"] = "BLOCKED"
    mutations.append(bool(validate(blocked, rows)))
    mutation_rejections = sum(mutations)

    print(json.dumps({
        "checker": "independent Claim 4 contract",
        "rows": len(rows),
        "levels": sorted(EXPECTED_LEVELS),
        "errors": errors,
        "mutation_rejections": mutation_rejections,
        "mutations_total": 3,
        "pass": not errors and mutation_rejections == 3,
    }, sort_keys=True))
    return 0 if not errors and mutation_rejections == 3 else 2


if __name__ == "__main__":
    raise SystemExit(main())
