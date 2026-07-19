"""Independent fail-closed verifier for the frozen exactly-ten C1 evidence."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "claim1_paper_scale"
EXPECTED = [
    "route01_synthetic.csv", "route02_boston.csv", "route03_yacht.csv",
    "route04_concrete.csv", "route05_energy.csv", "route06_carbon.csv",
    "route07_protein.csv", "route08_power.csv", "route09_kin8nm.csv",
    "route10_elevators.csv",
]


def _rows(path: Path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _sha(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _close(a, b, tolerance=1e-12):
    if abs(float(a) - float(b)) > tolerance * max(1.0, abs(float(b))):
        raise AssertionError(f"{a} != {b}")


def verify():
    validation = json.loads((OUT / "validation.json").read_text())
    assert validation["approaches_executed"] == 10
    assert validation["approach_numbers"] == list(range(1, 11))
    assert validation["exactly_ten_invariant"] is True
    assert validation["route_11_executed"] is False
    assert sorted(p.name for p in OUT.glob("route*.csv")) == EXPECTED

    ledger = (ROOT / "CLAIM1_APPROACH_LEDGER.md").read_text()
    ledger_routes = [int(x) for x in re.findall(r"^\| (\d+) \|", ledger, flags=re.MULTILINE)]
    assert ledger_routes == list(range(1, 11))
    assert "route numbered 11 or higher" in ledger

    all_rows = []
    for number, filename in enumerate(EXPECTED, 1):
        path = OUT / filename
        assert validation["result_file_sha256"][filename] == _sha(path)
        rows = _rows(path)
        assert {int(r["route"]) for r in rows} == {number}
        if number == 1:
            assert len(rows) == 180
            assert {int(r["seed"]) for r in rows} == set(range(10))
            assert {(int(r["n"]), int(r["d"]), int(r["ntest"])) for r in rows} == {(500, 50, 1000)}
            assert {r["grid"] for r in rows} == {"weight_grid", "data_grid"}
        else:
            assert len(rows) == 20
            assert {int(r["seed"]) for r in rows} == set(range(10))
            assert {r["condition"] for r in rows} == {"clean", "contaminated_10pct"}
            assert {int(r["rff_features"]) for r in rows} == {256}
            assert max(int(r["train_rows"]) for r in rows) <= 2000
        all_rows.extend(rows)
    assert len(all_rows) == 360
    assert {int(r["route"]) for r in all_rows} == set(range(1, 11))

    by_route = {int(a["route"]): a for a in validation["approaches"]}
    for route in range(2, 11):
        dirty = [r for r in all_rows if int(r["route"]) == route and r.get("condition") == "contaminated_10pct"]
        wins = sum(float(r["rmse_joint"]) < float(r["rmse_model_only"]) for r in dirty)
        assert wins == by_route[route]["joint_wins"]
        _close(sum(float(r["rmse_joint"]) for r in dirty) / 10, by_route[route]["mean_rmse_joint_contaminated"])
        _close(sum(float(r["rmse_model_only"]) for r in dirty) / 10, by_route[route]["mean_rmse_model_only_contaminated"])
        _close(sum(float(r["outlier_recall_joint"]) for r in dirty) / 10, by_route[route]["mean_outlier_recall"])

    supported = sum(bool(a["claim_supported"]) for a in validation["approaches"])
    assert supported == validation["approaches_supported"] == 9
    assert validation["approaches_adverse_or_mixed"] == 1
    sensitivity = json.loads((OUT / "route07_sensitivity.json").read_text())
    assert sensitivity["route"] == 7 and sensitivity["adds_approach"] is False
    assert len(sensitivity["schedules"]) == 3
    assert all(s["rmse"] > sensitivity["model_only_rmse"] for s in sensitivity["schedules"])
    print(
        "CLAIM1_VERIFY_PASS approaches=10 routes=1..10 rows=360 "
        "supported=9 adverse=1 route11=false sensitivity_route=7"
    )
    return True


if __name__ == "__main__":
    verify()
