# Claim 1 — Paper-scale exactly 10


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_d61c56a736a3", "created_at": "2026-07-19T00:21:20+00:00", "title": "Exactly-ten paper-scale report", "pinned": true, "pinned_at": "2026-07-19T00:21:21+00:00"}
-->
# Claim 1 paper-scale repair — exactly 10 approaches

## Outcome

This repair materially supports Claim 1 at paper scale, with an important
boundary case. Exactly ten pre-registered approaches were executed: the
paper's synthetic experiment and all nine tabular benchmarks. Nine approaches
support joint model-and-data ARD; Protein is adverse and is retained. No
eleventh approach was created.

Across the nine real benchmarks, joint ARD beats its homoscedastic model-only
ablation in 80/90 independently seeded 10%-contaminated trials (one-sided sign
test p=5.26e-15). Mean top-10%-noise outlier recall is 95.38% versus 10% chance,
and mean effective weight support is 16.88%, demonstrating simultaneous data
and model sparsification. The mean relative RMSE reduction is 31.94%, although
this aggregate must be read alongside the adverse Protein result.

The synthetic route uses n=500, d=50, ntest=1,000, 18 frozen grid cells, and ten
trials per cell. It obtains 96.67% mean weight-support recall, 61.50% mean
outlier-support recall, and a positive mean joint-vs-homoscedastic RMSE gain of
0.0593 (paired Wilcoxon p=2.41e-29).

## Exactly-ten results

| Route | Approach | Joint RMSE | Model-only RMSE | Wins | Outlier recall | Weight ESS | Result |
|---:|---|---:|---:|---:|---:|---:|---|
| 1 | Synthetic grids | — | — | paired gain 0.0593 | 61.50% | — | supports |
| 2 | Boston | 3.7975 | 5.1315 | 10/10 | 95.00% | 26.78% | supports |
| 3 | Yacht | 1.7612 | 6.7154 | 10/10 | 100.00% | 19.57% | supports |
| 4 | Concrete | 7.1162 | 8.4918 | 10/10 | 97.44% | 16.84% | supports |
| 5 | Energy | 0.8347 | 3.3536 | 10/10 | 100.00% | 7.50% | supports |
| 6 | Carbon | 0.00326 | 0.04312 | 10/10 | 100.00% | 6.50% | supports |
| 7 | Protein | 6.1554 | 5.1090 | 0/10 | 81.05% | 18.63% | **adverse** |
| 8 | Power | 4.2575 | 4.6553 | 10/10 | 98.60% | 23.81% | supports |
| 9 | Kin8nm | 0.14368 | 0.15535 | 10/10 | 91.90% | 18.94% | supports |
| 10 | Elevators | 0.003084 | 0.003400 | 10/10 | 94.45% | 13.32% | supports |

RMSE values are means over the ten contaminated trials in original target
units. The synthetic design has multiple scale/noise cells, so a single RMSE
mean would be misleading; its paired gain and recovery metrics are reported.

Machine invariant: `approaches_executed=10`, route numbers 1–10 exactly once,
`approaches_supported=9`, `approaches_adverse_or_mixed=1`, and
`route_11_executed=false`. Baselines, conditions, seeds, grids, and sensitivity
checks are subchecks within those routes.

## Protocol and primary sources

- Paper: arXiv 2605.29908v1, Sections 5.1–5.2 and Appendix D.2.
- Cited author repository: `alextimans/robust-sbl` pinned at
  `58908b880b1367dd417b34160b8e324c54b44e42`. At audit time it contains a
  README and license but no executable experiments, despite the paper's code
  availability statement. The repair is therefore clean-room and equation
  faithful.
- Synthetic: Gaussian X; sparse Gaussian weights; n=500, d=50, ntest=1,000;
  the paper's fixed heatmap conditions; explicit three-by-three weight and data
  grids; ten seeds.
- Tabular: 20% uncontaminated test split, standardized targets, 256 random
  Fourier features, clean and 10% response-contamination conditions, amplitude
  a=3, ten seeds, and random 2,000-row training cap for larger data.
- Each contaminated target receives the paper's signed Rademacher perturbation
  with magnitude distributed N(1,0.25²). Test targets remain untouched.
- Comparators are the joint EM estimator, its homoscedastic model-ARD ablation,
  RidgeCV, and Huber regression. All share the same split and RFF design.

The nine source schemas match Appendix D.2 exactly: Boston 506×13, Yacht
308×6, Concrete 1,030×8, Energy 768×8, Carbon 10,721×5, Protein 45,730×9,
Power 9,568×4, Kin8nm 8,192×8, and Elevators 16,599×18. UCI archives are
SHA-256 pinned; OpenML datasets are ID/version/MD5 pinned. Canonical numeric
matrix hashes appear in `outputs/claim1_paper_scale/validation.json`.

## Adversarial and negative checks

- Route 11 is rejected by code and unit test.
- The verifier independently reads every CSV, checks route/seed/condition
  coverage, recomputes mean RMSE, wins and recall, verifies every CSV hash, and
  checks the Markdown ledger independently of `validation.json`.
- Protein is not dropped or relabeled. It identifies 81.05% of injected
  outliers and stays sparse, but joint prediction is worse in all ten trials.
  Three Appendix-D.1 warm-up/update schedules at seed 0 also remain adverse
  (joint RMSE 5.58–5.89 vs model-only 5.22), so the limitation is robust to
  that local schedule sensitivity.
- Dataset source digests, raw dimensions, clean conditions, contaminated
  conditions, and all 360 decisive rows are retained.

## Limitations

The public author repository did not provide the claimed experiment code, so
exact optimizer hyperparameters and RFF bandwidth choices could not be copied.
This repair uses a median-distance RFF bandwidth and the paper's EM equations
with its recommended warm-up, damping, clipping, update cadence, and final
posterior recomputation. The fixed 80-step budget did not meet the stringent
log-parameter convergence flag; raw rows disclose this, and the route-7
120-step sensitivity remained adverse. The results therefore establish strong
finite-budget evidence, not exact parity with the unreleased implementation.

The Huber baseline emitted iteration-cap warnings on part of the Carbon run;
it is ancillary, and none of the claim-support decisions use Huber. The
decisive comparator is the same EM implementation with per-sample variance
enabled versus disabled.

## Reproduction and artifacts

```bash
uv venv --python 3.12 .venv
uv pip install -r requirements.txt
.venv/bin/python -m pytest repro/tests -q
.venv/bin/python repro/src/claim1_paper_scale.py
.venv/bin/python repro/src/route07_sensitivity.py
.venv/bin/python repro/src/verify_claim1_paper_scale.py
```

The decisive run took 1,445.58 CPU-wall seconds and produced 360 raw rows.
`validation.json` SHA-256 is
`26e587d679b888dcbc3b856c25606a2eabee476252a5f373ac500f04c4d2c6d9`;
the retained Protein sensitivity JSON SHA-256 is
`b800289ffdf93a0d93037075347cf544faf7711d27002d1d6e7e340b2e3fff88`.
The verifier terminates with:

```text
CLAIM1_VERIFY_PASS approaches=10 routes=1..10 rows=360 supported=9 adverse=1 route11=false sensitivity_route=7
```


---
<!-- trackio-cell
{"type": "code", "id": "cell_696d390e40f6", "created_at": "2026-07-19T00:21:21+00:00", "title": "Independent fail-closed verifier", "command": [".venv/bin/python", "repro/src/verify_claim1_paper_scale.py"], "exit_code": 0, "duration_s": 0.052}
-->
````bash
$ .venv/bin/python repro/src/verify_claim1_paper_scale.py
````

exit 0 · 0.1s


````python title=verify_claim1_paper_scale.py
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

````


````output
CLAIM1_VERIFY_PASS approaches=10 routes=1..10 rows=360 supported=9 adverse=1 route11=false sensitivity_route=7

````


---
<!-- trackio-cell
{"type": "code", "id": "cell_57173cfb72d6", "created_at": "2026-07-19T00:22:12+00:00", "title": "Deterministic raw evidence bundle", "command": [".venv/bin/python", "repro/src/bundle_claim1_artifacts.py"], "exit_code": 0, "duration_s": 0.046}
-->
````bash
$ .venv/bin/python repro/src/bundle_claim1_artifacts.py
````

exit 0 · 0.0s


````python title=bundle_claim1_artifacts.py
"""Build a deterministic, self-verifying archive for the public logbook."""
from __future__ import annotations

import hashlib
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / "outputs" / "claim1_paper_scale_bundle.zip"
FILES = [
    ROOT / "CLAIM1_APPROACH_LEDGER.md",
    ROOT / "CLAIM1_PAPER_SCALE_REPORT.md",
    ROOT / "README.md",
    ROOT / "requirements.txt",
    *sorted((ROOT / "outputs" / "claim1_paper_scale").glob("*")),
    ROOT / "repro" / "src" / "claim1_datasets.py",
    ROOT / "repro" / "src" / "claim1_paper_scale.py",
    ROOT / "repro" / "src" / "paper_scale_ard.py",
    ROOT / "repro" / "src" / "route07_sensitivity.py",
    ROOT / "repro" / "src" / "verify_claim1_paper_scale.py",
    ROOT / "repro" / "tests" / "test_claim1_datasets.py",
    ROOT / "repro" / "tests" / "test_claim1_outputs.py",
    ROOT / "repro" / "tests" / "test_paper_scale_ard.py",
]


def main():
    rows = []
    for path in FILES:
        if not path.is_file():
            raise FileNotFoundError(path)
        relative = path.relative_to(ROOT).as_posix()
        rows.append((relative, path.read_bytes()))
    manifest = "".join(
        f"{hashlib.sha256(payload).hexdigest()}  {name}\n" for name, payload in rows
    ).encode()
    DEST.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(DEST, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, payload in rows + [("MANIFEST.sha256", manifest)]:
            info = zipfile.ZipInfo(name, date_time=(2026, 7, 19, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, payload)
    print(
        f"BUNDLE_PASS files={len(rows) + 1} bytes={DEST.stat().st_size} "
        f"sha256={hashlib.sha256(DEST.read_bytes()).hexdigest()} path={DEST.relative_to(ROOT)}"
    )


if __name__ == "__main__":
    main()

````


````output
BUNDLE_PASS files=25 bytes=49915 sha256=d9d6e9df7f791137f7ce30d7519ffdcff2989a104196452690ed481c588308ae path=outputs/claim1_paper_scale_bundle.zip

````


---
<!-- trackio-cell
{"type": "code", "id": "cell_53ec5ce0c955", "created_at": "2026-07-19T00:22:53+00:00", "title": "Agent-readable complete evidence bundle", "command": [".venv/bin/python", "repro/src/bundle_claim1_artifacts.py"], "exit_code": 0, "duration_s": 0.049}
-->
````bash
$ .venv/bin/python repro/src/bundle_claim1_artifacts.py
````

exit 0 · 0.0s


````python title=bundle_claim1_artifacts.py
"""Build a deterministic, self-verifying archive for the public logbook."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / "outputs" / "claim1_paper_scale_bundle.zip"
DEST_JSON = ROOT / "outputs" / "claim1_paper_scale_bundle.json"
FILES = [
    ROOT / "CLAIM1_APPROACH_LEDGER.md",
    ROOT / "CLAIM1_PAPER_SCALE_REPORT.md",
    ROOT / "README.md",
    ROOT / "requirements.txt",
    *sorted((ROOT / "outputs" / "claim1_paper_scale").glob("*")),
    ROOT / "repro" / "src" / "claim1_datasets.py",
    ROOT / "repro" / "src" / "claim1_paper_scale.py",
    ROOT / "repro" / "src" / "paper_scale_ard.py",
    ROOT / "repro" / "src" / "route07_sensitivity.py",
    ROOT / "repro" / "src" / "verify_claim1_paper_scale.py",
    ROOT / "repro" / "tests" / "test_claim1_datasets.py",
    ROOT / "repro" / "tests" / "test_claim1_outputs.py",
    ROOT / "repro" / "tests" / "test_paper_scale_ard.py",
]


def main():
    rows = []
    for path in FILES:
        if not path.is_file():
            raise FileNotFoundError(path)
        relative = path.relative_to(ROOT).as_posix()
        rows.append((relative, path.read_bytes()))
    manifest = "".join(
        f"{hashlib.sha256(payload).hexdigest()}  {name}\n" for name, payload in rows
    ).encode()
    DEST.parent.mkdir(parents=True, exist_ok=True)
    json_bundle = {
        "schema": "vMcu1h3fOV-claim1-exactly-ten-v1",
        "files": [
            {"path": name, "sha256": hashlib.sha256(payload).hexdigest(),
             "content": payload.decode("utf-8")}
            for name, payload in rows
        ],
        "manifest_sha256": hashlib.sha256(manifest).hexdigest(),
    }
    DEST_JSON.write_text(json.dumps(json_bundle, separators=(",", ":")) + "\n")
    with zipfile.ZipFile(DEST, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, payload in rows + [("MANIFEST.sha256", manifest)]:
            info = zipfile.ZipInfo(name, date_time=(2026, 7, 19, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, payload)
    print(
        f"BUNDLE_PASS files={len(rows) + 1} bytes={DEST.stat().st_size} "
        f"sha256={hashlib.sha256(DEST.read_bytes()).hexdigest()} path={DEST.relative_to(ROOT)} "
        f"json_bytes={DEST_JSON.stat().st_size} "
        f"json_sha256={hashlib.sha256(DEST_JSON.read_bytes()).hexdigest()}"
    )


if __name__ == "__main__":
    main()

````


````output
BUNDLE_PASS files=25 bytes=49915 sha256=d9d6e9df7f791137f7ce30d7519ffdcff2989a104196452690ed481c588308ae path=outputs/claim1_paper_scale_bundle.zip json_bytes=129247 json_sha256=b9a4bab50ceff5861b38219557c781333cad3ccdd6de92a11e0bb36c976aabb5

````


---
<!-- trackio-cell
{"type": "artifact", "id": "cell_fef2fce1f92a", "created_at": "2026-07-19T00:23:02+00:00", "title": "Complete raw evidence (360 rows + sources)", "artifact": "outputs/claim1_paper_scale_bundle.json", "artifact_type": "dataset"}
-->
**📦 Artifact** `outputs/claim1_paper_scale_bundle.json` · dataset

trackio-artifact://outputs/claim1_paper_scale_bundle.json


---
<!-- trackio-cell
{"type": "artifact", "id": "cell_1007625b9d70", "created_at": "2026-07-19T00:23:09+00:00", "title": "Deterministic evidence archive with SHA manifest", "artifact": "outputs/claim1_paper_scale_bundle.zip", "artifact_type": "dataset"}
-->
**📦 Artifact** `outputs/claim1_paper_scale_bundle.zip` · dataset

trackio-artifact://outputs/claim1_paper_scale_bundle.zip


---
<!-- trackio-cell
{"type": "code", "id": "cell_0dc3f5515f84", "created_at": "2026-07-19T00:24:20+00:00", "title": "Final deterministic evidence bundle", "command": [".venv/bin/python", "repro/src/bundle_claim1_artifacts.py"], "exit_code": 0, "duration_s": 0.05}
-->
````bash
$ .venv/bin/python repro/src/bundle_claim1_artifacts.py
````

exit 0 · 0.1s


````python title=bundle_claim1_artifacts.py
"""Build a deterministic, self-verifying archive for the public logbook."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / "outputs" / "claim1_paper_scale_bundle.zip"
DEST_JSON = ROOT / "outputs" / "claim1_paper_scale_bundle.json"
FILES = [
    ROOT / "CLAIM1_APPROACH_LEDGER.md",
    ROOT / "CLAIM1_PAPER_SCALE_REPORT.md",
    ROOT / "README.md",
    ROOT / "requirements.txt",
    *sorted((ROOT / "outputs" / "claim1_paper_scale").glob("*")),
    ROOT / "repro" / "src" / "claim1_datasets.py",
    ROOT / "repro" / "src" / "claim1_paper_scale.py",
    ROOT / "repro" / "src" / "paper_scale_ard.py",
    ROOT / "repro" / "src" / "route07_sensitivity.py",
    ROOT / "repro" / "src" / "verify_claim1_paper_scale.py",
    ROOT / "repro" / "src" / "claim1_publish_gate.py",
    ROOT / "repro" / "tests" / "test_claim1_datasets.py",
    ROOT / "repro" / "tests" / "test_claim1_outputs.py",
    ROOT / "repro" / "tests" / "test_paper_scale_ard.py",
]


def main():
    rows = []
    for path in FILES:
        if not path.is_file():
            raise FileNotFoundError(path)
        relative = path.relative_to(ROOT).as_posix()
        rows.append((relative, path.read_bytes()))
    manifest = "".join(
        f"{hashlib.sha256(payload).hexdigest()}  {name}\n" for name, payload in rows
    ).encode()
    DEST.parent.mkdir(parents=True, exist_ok=True)
    json_bundle = {
        "schema": "vMcu1h3fOV-claim1-exactly-ten-v1",
        "files": [
            {"path": name, "sha256": hashlib.sha256(payload).hexdigest(),
             "content": payload.decode("utf-8")}
            for name, payload in rows
        ],
        "manifest_sha256": hashlib.sha256(manifest).hexdigest(),
    }
    DEST_JSON.write_text(json.dumps(json_bundle, separators=(",", ":")) + "\n")
    with zipfile.ZipFile(DEST, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, payload in rows + [("MANIFEST.sha256", manifest)]:
            info = zipfile.ZipInfo(name, date_time=(2026, 7, 19, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, payload)
    print(
        f"BUNDLE_PASS files={len(rows) + 1} bytes={DEST.stat().st_size} "
        f"sha256={hashlib.sha256(DEST.read_bytes()).hexdigest()} path={DEST.relative_to(ROOT)} "
        f"json_bytes={DEST_JSON.stat().st_size} "
        f"json_sha256={hashlib.sha256(DEST_JSON.read_bytes()).hexdigest()}"
    )


if __name__ == "__main__":
    main()

````


````output
BUNDLE_PASS files=26 bytes=50881 sha256=9740437226d6bf7c1b11458bf131a3828ff86c0b815b50b43e2e988aa161bdec path=outputs/claim1_paper_scale_bundle.zip json_bytes=131190 json_sha256=0aabf55f1f614eb14026d4c659aa3b3b6951dcd6facc07591a9d09c8d5a50c92

````


---
<!-- trackio-cell
{"type": "code", "id": "cell_bb1da24d2775", "created_at": "2026-07-19T00:26:11+00:00", "title": "Final publication gate", "command": [".venv/bin/python", "repro/src/claim1_publish_gate.py"], "exit_code": 0, "duration_s": 0.039}
-->
````bash
$ .venv/bin/python repro/src/claim1_publish_gate.py
````

exit 0 · 0.0s


````python title=claim1_publish_gate.py
"""Fail closed before syncing the exactly-ten repair to the public Space."""
from __future__ import annotations

import glob
import json
from pathlib import Path
import re

from verify_claim1_paper_scale import verify


ROOT = Path(__file__).resolve().parents[2]


def main():
    assert verify()
    metadata = json.loads((ROOT / ".trackio" / "metadata.json").read_text())
    manifest = json.loads((ROOT / ".trackio" / "logbook" / "logbook.json").read_text())
    pages = "".join(
        Path(path).read_text()
        for path in glob.glob(
            str(ROOT / ".trackio" / "logbook" / "pages" / "**" / "*.md"),
            recursive=True,
        )
    )
    cells = [
        json.loads(raw)
        for raw in re.findall(r"<!-- trackio-cell\n(\{[^\n]+\})\n-->", pages)
    ]
    assert metadata["space_id"] == "DineshAI/vMcu1h3fOV"
    assert metadata["autosync"] is False
    assert metadata["private"] is False
    assert set(metadata["tags"]) >= {"icml2026-repro", "paper-vMcu1h3fOV"}
    assert sum(cell.get("pinned") is True for cell in cells) == 1
    bundle_artifacts = [
        cell for cell in cells
        if cell.get("type") == "artifact"
        and str(cell.get("artifact", "")).startswith("outputs/claim1_paper_scale_bundle")
    ]
    assert len(bundle_artifacts) == 2
    host_prefix = "/" + "home" + "/"
    assert host_prefix not in pages
    for text in (
        "approaches_executed=10", "routes=1..10", "rows=360",
        "supported=9", "adverse=1", "route11=false", "Protein",
    ):
        assert text in pages
    assert int(manifest["agent_view_tokens"]) >= 4000
    for name in ("claim1_paper_scale_bundle.json", "claim1_paper_scale_bundle.zip"):
        path = ROOT / "outputs" / name
        assert path.stat().st_size > 10_000
    print(
        "CLAIM1_PUBLISH_GATE_PASS approaches=10 rows=360 supported=9 adverse=1 "
        f"pins=1 artifacts=2 agent_tokens={manifest['agent_view_tokens']} autosync=false "
        "host_paths=0"
    )


if __name__ == "__main__":
    main()

````


````output
CLAIM1_VERIFY_PASS approaches=10 routes=1..10 rows=360 supported=9 adverse=1 route11=false sensitivity_route=7
CLAIM1_PUBLISH_GATE_PASS approaches=10 rows=360 supported=9 adverse=1 pins=1 artifacts=2 agent_tokens=4773 autosync=false host_paths=0

````


---
<!-- trackio-cell
{"type": "code", "id": "cell_7735b3701291", "created_at": "2026-07-19T00:26:11+00:00", "title": "Canonical final evidence bundle", "command": [".venv/bin/python", "repro/src/bundle_claim1_artifacts.py"], "exit_code": 0, "duration_s": 0.05}
-->
````bash
$ .venv/bin/python repro/src/bundle_claim1_artifacts.py
````

exit 0 · 0.1s


````python title=bundle_claim1_artifacts.py
"""Build a deterministic, self-verifying archive for the public logbook."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / "outputs" / "claim1_paper_scale_bundle.zip"
DEST_JSON = ROOT / "outputs" / "claim1_paper_scale_bundle.json"
FILES = [
    ROOT / "CLAIM1_APPROACH_LEDGER.md",
    ROOT / "CLAIM1_PAPER_SCALE_REPORT.md",
    ROOT / "README.md",
    ROOT / "requirements.txt",
    *sorted((ROOT / "outputs" / "claim1_paper_scale").glob("*")),
    ROOT / "repro" / "src" / "claim1_datasets.py",
    ROOT / "repro" / "src" / "claim1_paper_scale.py",
    ROOT / "repro" / "src" / "paper_scale_ard.py",
    ROOT / "repro" / "src" / "route07_sensitivity.py",
    ROOT / "repro" / "src" / "verify_claim1_paper_scale.py",
    ROOT / "repro" / "src" / "claim1_publish_gate.py",
    ROOT / "repro" / "tests" / "test_claim1_datasets.py",
    ROOT / "repro" / "tests" / "test_claim1_outputs.py",
    ROOT / "repro" / "tests" / "test_paper_scale_ard.py",
]


def main():
    rows = []
    for path in FILES:
        if not path.is_file():
            raise FileNotFoundError(path)
        relative = path.relative_to(ROOT).as_posix()
        rows.append((relative, path.read_bytes()))
    manifest = "".join(
        f"{hashlib.sha256(payload).hexdigest()}  {name}\n" for name, payload in rows
    ).encode()
    DEST.parent.mkdir(parents=True, exist_ok=True)
    json_bundle = {
        "schema": "vMcu1h3fOV-claim1-exactly-ten-v1",
        "files": [
            {"path": name, "sha256": hashlib.sha256(payload).hexdigest(),
             "content": payload.decode("utf-8")}
            for name, payload in rows
        ],
        "manifest_sha256": hashlib.sha256(manifest).hexdigest(),
    }
    DEST_JSON.write_text(json.dumps(json_bundle, separators=(",", ":")) + "\n")
    with zipfile.ZipFile(DEST, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, payload in rows + [("MANIFEST.sha256", manifest)]:
            info = zipfile.ZipInfo(name, date_time=(2026, 7, 19, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, payload)
    print(
        f"BUNDLE_PASS files={len(rows) + 1} bytes={DEST.stat().st_size} "
        f"sha256={hashlib.sha256(DEST.read_bytes()).hexdigest()} path={DEST.relative_to(ROOT)} "
        f"json_bytes={DEST_JSON.stat().st_size} "
        f"json_sha256={hashlib.sha256(DEST_JSON.read_bytes()).hexdigest()}"
    )


if __name__ == "__main__":
    main()

````


````output
BUNDLE_PASS files=26 bytes=51017 sha256=b33cf6fbe9e95ae97bcb1004bfe98c3703ec69db353a8cdb53e1238bde2eff53 path=outputs/claim1_paper_scale_bundle.zip json_bytes=131560 json_sha256=ca663941baaf5d84171b4cfd5167b3e6a901792148963340dcee8b6684c0dafe

````


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_3429839c1c9e", "created_at": "2026-07-19T00:26:19+00:00", "title": "Canonical bundle digests"}
-->
The two artifact cells above contain the same canonical 26-file evidence payload. ZIP SHA-256: b33cf6fbe9e95ae97bcb1004bfe98c3703ec69db353a8cdb53e1238bde2eff53. Agent-readable JSON SHA-256: ca663941baaf5d84171b4cfd5167b3e6a901792148963340dcee8b6684c0dafe. The JSON is 131,560 bytes; the ZIP is 51,017 bytes and includes MANIFEST.sha256.


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_63415861d4e5", "created_at": "2026-07-19T00:27:35+00:00", "title": "Direct public evidence links"}
-->
Trackio artifact-bucket creation is rate-limited, so the same canonical payload is also stored directly in this existing public Space. Agent-readable JSON: https://huggingface.co/spaces/DineshAI/vMcu1h3fOV/resolve/main/evidence/claim1_paper_scale_bundle.json . Deterministic ZIP with MANIFEST.sha256: https://huggingface.co/spaces/DineshAI/vMcu1h3fOV/resolve/main/evidence/claim1_paper_scale_bundle.zip . Source and raw CSVs are also pinned at GitHub commit b041cf1796c2fc494aaf05abbcb33379a1a047a7.
