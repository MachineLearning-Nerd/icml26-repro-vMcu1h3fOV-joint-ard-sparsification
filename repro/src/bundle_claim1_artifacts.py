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
