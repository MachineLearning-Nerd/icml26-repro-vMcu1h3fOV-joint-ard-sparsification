import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from claim4_faithful import rbf_design


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
