import numpy as np

from paper_scale_ard import effective_support, fit_sbl


def test_effective_support_limits():
    assert np.isclose(effective_support(np.ones(10)), 1.0)
    concentrated = effective_support(np.array([1.0] + [1e-300] * 9))
    assert np.isclose(concentrated, 0.1)


def test_joint_ard_recovers_obvious_outliers_and_sparse_signal():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(220, 16))
    truth = np.zeros(16); truth[:3] = [2.0, -1.5, 0.8]
    y = X @ truth + rng.normal(scale=0.15, size=len(X))
    outliers = np.arange(20)
    y[outliers] += rng.normal(scale=8.0, size=len(outliers))
    fit = fit_sbl(X, y, heteroscedastic=True, max_iter=100, warmup=20)
    top_outliers = np.argpartition(fit.noise_var, -20)[-20:]
    assert len(set(top_outliers).intersection(outliers)) >= 14
    top_weights = np.argpartition(1.0 / fit.gamma, -3)[-3:]
    assert set(top_weights) == {0, 1, 2}
