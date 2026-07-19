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
