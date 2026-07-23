#!/usr/bin/env python3
"""Fixed-command cumulative runner for the judged 5/8 baseline.

Experiment children inherit the exact same outer command and change only
committed implementation/configuration. Every accepted check is rerun.
"""
from __future__ import annotations

import json
import os
import platform
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / ".openresearch" / "artifacts" / "source" / "2605.29908.html"
PRIMARY_C2 = ROOT / "repro" / "src" / "verify_claim2_all_schemes.py"


def run(label: str, argv: list[str]) -> dict:
    print(f"\nCAMPAIGN_STEP_START label={label} command={json.dumps(argv)}", flush=True)
    started = time.monotonic()
    completed = subprocess.run(argv, cwd=ROOT, check=False)
    elapsed = time.monotonic() - started
    print(
        f"CAMPAIGN_STEP_END label={label} exit_code={completed.returncode} "
        f"elapsed_seconds={elapsed:.3f}",
        flush=True,
    )
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)
    return {"label": label, "command": argv, "elapsed_seconds": elapsed}


def main() -> int:
    started = time.monotonic()
    print("CAMPAIGN_METADATA " + json.dumps({
        "git_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "paper_sha256": "0a3ed827e899a92f2ed3b1e877ea2f6456ec3d827af184045acac3b8ebd4cb49",
        "seeds": {
            "claim1_toy": list(range(10)),
            "claim1_paper_scale": list(range(10)),
            "claim2_primary": list(range(6)),
            "claim2_independent": list(range(7)),
            "claim4_existing_proxy": list(range(20)),
        },
    }, sort_keys=True), flush=True)

    py = sys.executable
    steps = [
        run("pytest", [py, "-m", "pytest", "repro/tests", "-q"]),
        run("claim1_claim2_core", [py, "repro/src/run_claims.py"]),
        run("claim1_paper_scale_regenerate", [py, "repro/src/claim1_paper_scale.py"]),
        run("claim1_protein_sensitivity", [py, "repro/src/route07_sensitivity.py"]),
        run("claim1_paper_scale_verify", [py, "repro/src/verify_claim1_paper_scale.py"]),
        run(
            "claim2_all_schemes",
            [py, str(PRIMARY_C2), "--paper-html", str(PAPER)],
        ),
        run(
            "claim2_independent",
            [
                py,
                "repro/src/audit_claim2_all_schemes_independent.py",
                "--paper-html",
                str(PAPER),
                "--primary",
                str(PRIMARY_C2),
            ],
        ),
        run("claim4_existing_proxy", [py, "repro/src/verify_joint_ard_robust_scale.py"]),
        run("claim1_bundle", [py, "repro/src/bundle_claim1_artifacts.py"]),
        run("claim1_publication_regression", [py, "repro/src/claim1_publish_gate.py"]),
    ]
    total = time.monotonic() - started
    print("CAMPAIGN_SUMMARY " + json.dumps({
        "state": "judged_baseline_5_of_8",
        "steps": steps,
        "total_elapsed_seconds": total,
        "claims": {
            "1": "VERIFIED",
            "2": "VERIFIED",
            "3": "BLOCKED",
            "4": "BLOCKED",
        },
        "limitations": [
            "Claim 3 exact Table-3 percentages are not matched; Protein used EM, not paper-assigned MacKay.",
            "Claim 4 uses synthetic Boston-scale proxy data and omits the Figure-3 grid.",
        ],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
