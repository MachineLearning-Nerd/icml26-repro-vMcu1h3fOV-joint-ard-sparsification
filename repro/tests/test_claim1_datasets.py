"""Fail-closed structural tests for the exactly-ten Claim 1 route ledger."""
from pathlib import Path
import sys

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))
from claim1_datasets import OPENML, UCI, load_route_dataset


def test_real_route_manifest_is_exactly_2_through_10():
    assert sorted(OPENML.keys() | UCI.keys()) == list(range(2, 11))


def test_route_11_is_rejected():
    with pytest.raises(ValueError, match="2..10"):
        load_route_dataset(11)


@pytest.mark.parametrize("route", range(2, 11))
def test_paper_scale_dataset_schema(route):
    X, y, meta = load_route_dataset(route)
    assert X.shape == (meta["rows"], meta["raw_features"])
    assert y.shape == (meta["rows"],)
    assert len(meta["canonical_matrix_sha256"]) == 64
