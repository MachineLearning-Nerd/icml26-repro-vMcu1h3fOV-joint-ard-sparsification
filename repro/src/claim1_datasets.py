"""Pinned paper-scale datasets for Claim 1 routes 2--10.

The route numbers and source versions are frozen in CLAIM1_APPROACH_LEDGER.md.
Downloads are cached outside version control and every returned numeric matrix
gets a canonical SHA-256 digest so a report identifies the exact data content.
"""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
import requests
from sklearn.datasets import fetch_openml


ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / ".cache" / "claim1_data"

OPENML = {
    2: ("Boston", 531, 506, 13, "cdd361fb886627eaa80c92f90d0610cc"),
    9: ("Kin8nm", 189, 8192, 8, "b36414070701ab8bc985104c3ebd9b77"),
    10: ("Elevators", 216, 16599, 18, "dff17dfd0e44dcd867a0097b26241027"),
}

UCI = {
    3: {
        "name": "Yacht", "id": 243, "rows": 308, "features": 6,
        "slug": "yacht+hydrodynamics", "sha256": "aa52b68f88c4bb552187a53ef4c5753fa178f6a36035a3771c5bc04e078487ac",
        "member": "yacht_hydrodynamics.data",
    },
    4: {
        "name": "Concrete", "id": 165, "rows": 1030, "features": 8,
        "slug": "concrete+compressive+strength", "sha256": "dad85d14de8aee4e07479daa774e6b569a313715b71a3b92c95a07cf91c2c9a7",
        "member": "Concrete_Data.xls",
    },
    5: {
        "name": "Energy", "id": 242, "rows": 768, "features": 8,
        "slug": "energy+efficiency", "sha256": "499441eee27929a4b00417f58fd8c63c9cc14b8a71520cd0dd27fcb626738351",
        "member": "ENB2012_data.xlsx",
    },
    6: {
        "name": "Carbon", "id": 448, "rows": 10721, "features": 5,
        "slug": "carbon+nanotubes", "sha256": "e7e6167ddf40fc2a3e6cdeacd14ae139c9e06e9b7541778e0c3bd9bb86091479",
        "member": "carbon_nanotubes.csv",
    },
    7: {
        "name": "Protein", "id": 265, "rows": 45730, "features": 9,
        "slug": "physicochemical+properties+of+protein+tertiary+structure",
        "sha256": "ee6536c8cc415dc50d66bd247af7c9bc95c187ab8a4ea0f1f546b127206c1100",
        "member": "CASP.csv",
    },
    8: {
        "name": "Power", "id": 294, "rows": 9568, "features": 4,
        "slug": "combined+cycle+power+plant", "sha256": "cc7b2a4977c0a44e8221c91d9a7e5746b3c68186cff7e5c61c70af6432b98c7a",
        "member": "CCPP/Folds5x2_pp.xlsx",
    },
}


def _canonical_sha(X: np.ndarray, y: np.ndarray) -> str:
    h = hashlib.sha256()
    for a in (np.asarray(X, dtype="<f8"), np.asarray(y, dtype="<f8")):
        h.update(json.dumps(list(a.shape), separators=(",", ":")).encode())
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def _download_uci(spec: dict) -> bytes:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"uci-{spec['id']}.zip"
    if not path.exists():
        url = f"https://archive.ics.uci.edu/static/public/{spec['id']}/{spec['slug']}.zip"
        response = requests.get(url, timeout=120)
        response.raise_for_status()
        path.write_bytes(response.content)
    payload = path.read_bytes()
    actual = hashlib.sha256(payload).hexdigest()
    if actual != spec["sha256"]:
        raise RuntimeError(f"UCI {spec['id']} archive hash {actual} != pinned {spec['sha256']}")
    return payload


def _load_uci(route: int) -> tuple[np.ndarray, np.ndarray, dict]:
    spec = UCI[route]
    with zipfile.ZipFile(io.BytesIO(_download_uci(spec))) as archive:
        raw = archive.read(spec["member"])
    if route == 3:
        a = np.loadtxt(io.BytesIO(raw))
        X, y = a[:, :6], a[:, 6]
    elif route == 4:
        frame = pd.read_excel(io.BytesIO(raw), engine="xlrd")
        X, y = frame.iloc[:, :8].to_numpy(), frame.iloc[:, 8].to_numpy()
    elif route == 5:
        frame = pd.read_excel(io.BytesIO(raw), engine="openpyxl")
        frame = frame.dropna(how="all")
        X, y = frame.iloc[:, :8].to_numpy(), frame.iloc[:, 8].to_numpy()
    elif route == 6:
        frame = pd.read_csv(io.BytesIO(raw), sep=";", decimal=",")
        X, y = frame.iloc[:, :5].to_numpy(), frame.iloc[:, 5].to_numpy()
    elif route == 7:
        frame = pd.read_csv(io.BytesIO(raw))
        X, y = frame.iloc[:, 1:10].to_numpy(), frame.iloc[:, 0].to_numpy()
    elif route == 8:
        frame = pd.read_excel(io.BytesIO(raw), engine="openpyxl")
        X, y = frame.iloc[:, :4].to_numpy(), frame.iloc[:, 4].to_numpy()
    else:  # pragma: no cover - guarded by caller
        raise ValueError(route)
    meta = {
        "source": "UCI Machine Learning Repository",
        "source_id": spec["id"],
        "source_archive_sha256": spec["sha256"],
    }
    return np.asarray(X, float), np.asarray(y, float), meta


def _load_openml(route: int) -> tuple[np.ndarray, np.ndarray, dict]:
    name, did, _, _, md5 = OPENML[route]
    bunch = fetch_openml(data_id=did, as_frame=False, parser="auto")
    actual_md5 = bunch.details.get("md5_checksum")
    if actual_md5 != md5:
        raise RuntimeError(f"OpenML {did} md5 {actual_md5} != pinned {md5}")
    return np.asarray(bunch.data, float), np.asarray(bunch.target, float), {
        "source": "OpenML", "source_id": did, "source_version": 1,
        "source_md5": md5, "name": name,
    }


def load_route_dataset(route: int) -> tuple[np.ndarray, np.ndarray, dict]:
    """Load one frozen real-data route and validate its paper-stated schema."""
    if route in OPENML:
        X, y, meta = _load_openml(route)
        name, _, rows, features, _ = OPENML[route]
    elif route in UCI:
        X, y, meta = _load_uci(route)
        spec = UCI[route]
        name, rows, features = spec["name"], spec["rows"], spec["features"]
    else:
        raise ValueError(f"route must be one of 2..10, got {route}")
    if X.shape != (rows, features) or y.shape != (rows,):
        raise RuntimeError(f"{name} schema {(X.shape, y.shape)} != {((rows, features), (rows,))}")
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise RuntimeError(f"{name} contains non-finite values")
    meta.update({
        "route": route, "dataset": name, "rows": rows, "raw_features": features,
        "canonical_matrix_sha256": _canonical_sha(X, y),
    })
    return X, y, meta


if __name__ == "__main__":
    for number in range(2, 11):
        X_, y_, metadata = load_route_dataset(number)
        print(json.dumps(metadata, sort_keys=True))
