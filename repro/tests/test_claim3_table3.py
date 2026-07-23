import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from claim3_table3 import Protocol, fit


def test_mackay_and_em_are_finite_on_small_problem():
    rng = np.random.default_rng(11)
    X = rng.normal(size=(30, 5))
    y = X @ np.array([1.0, 0.0, -0.5, 0.0, 0.2]) + rng.normal(scale=0.1, size=30)
    config = Protocol(
        damping=0.02,
        warmup=2,
        noise_update_every=2,
        tolerance=1e-4,
        patience=2,
        max_iter=30,
    )
    for scheme in ("em", "mackay"):
        for heteroscedastic in (False, True):
            result = fit(X, y, heteroscedastic=heteroscedastic, scheme=scheme, config=config)
            assert np.isfinite(result.coef).all()
            assert np.isfinite(result.gamma).all()
            assert np.isfinite(result.noise_var).all()
            assert (result.gamma > 0).all()
            assert (result.noise_var > 0).all()
