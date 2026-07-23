import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from claim4_faithful import fit_huber_irls, rbf_design


def test_rbf_design_uses_all_declared_centers_without_bias():
    train = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    test = np.array([[2.0, 2.0], [3.0, 3.0]])
    centers = np.vstack([train, test])
    phi_train, phi_test, lengthscale = rbf_design(train, test, centers)
    assert phi_train.shape == (3, 5)
    assert phi_test.shape == (2, 5)
    assert lengthscale > 0
    assert np.allclose(np.diag(phi_train[:, :3]), 1.0)
    assert np.allclose(np.diag(phi_test[:, 3:]), 1.0)
    assert not np.allclose(phi_train[:, -1], 1.0)


def test_huber_irls_converges_and_downweights_a_large_outlier():
    x = np.linspace(-1.0, 1.0, 31)
    design = x[:, None]
    clean = 1.0 + 2.0 * x
    contaminated = clean.copy()
    contaminated[0] += 20.0
    fitted = fit_huber_irls(design, contaminated)
    assert fitted.converged
    assert fitted.iterations < 500
    assert fitted.relative_step <= 1e-8
    assert fitted.stationarity_error <= 1e-8
    assert abs(fitted.coef[0] - 2.0) < 0.2
    assert abs(fitted.intercept - 1.0) < 0.2
