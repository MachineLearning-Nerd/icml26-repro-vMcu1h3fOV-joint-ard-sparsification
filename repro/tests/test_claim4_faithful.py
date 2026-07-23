import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from claim4_faithful import rbf_design


def test_rbf_design_uses_training_centers_only():
    train = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    test = np.array([[2.0, 2.0], [3.0, 3.0]])
    phi_train, phi_test, lengthscale = rbf_design(train, test)
    assert phi_train.shape == (3, 4)
    assert phi_test.shape == (2, 4)
    assert lengthscale > 0
    assert np.allclose(phi_train[:, -1], 1.0)
    assert np.allclose(phi_test[:, -1], 1.0)
