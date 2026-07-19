from pathlib import Path
import sys

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))
from verify_claim1_paper_scale import verify


def test_committed_exactly_ten_outputs_fail_closed():
    assert verify() is True
