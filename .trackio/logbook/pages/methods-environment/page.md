# Methods & environment


---
<!-- trackio-cell
{"type": "code", "id": "cell_deb141d86ee9", "created_at": "2026-07-17T06:43:08+00:00", "title": "Pytest suite (8 tests)", "command": ["python", "-m", "pytest", "repro/tests/test_ard.py", "-q"], "exit_code": 0, "duration_s": 0.655}
-->
````bash
$ python -m pytest repro/tests/test_ard.py -q
````

exit 0 · 0.7s


````python title=test_ard.py
"""Formal pytest suite: Joint ARD sparsification (arXiv 2605.29908)."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
import pytest
from ard import marginal_loglik, posterior, em_update, fit_joint, predict
from run_claims import make_problem


# -- C2: closed-form marginal likelihood is the Gaussian N(0, ΦΓ⁻¹Φᵀ+Λ⁻¹) -------
def test_c2_marginal_loglik_is_gaussian():
    rng = np.random.default_rng(0)
    Phi = rng.normal(0, 1, (12, 4)); y = rng.normal(0, 1, 12)
    g = np.array([1.0, 2.0, 0.5, 3.0]); l = np.ones(12) * 1.5
    C = Phi @ np.diag(1 / g) @ Phi.T + np.diag(1 / l)
    ll = -0.5 * (12 * np.log(2 * np.pi) + np.linalg.slogdet(C)[1] + y @ np.linalg.solve(C, y))
    assert abs(marginal_loglik(Phi, y, g, l) - ll) < 1e-9


# -- C2: posterior μ,Σ match the precision-form definition -------------------
def test_c2_posterior_definition():
    rng = np.random.default_rng(1)
    Phi = rng.normal(0, 1, (15, 5)); y = rng.normal(0, 1, 15)
    g = rng.uniform(0.5, 2, 5); l = rng.uniform(0.5, 2, 15)
    mu, Sig = posterior(Phi, y, g, l)
    Prec = np.diag(g) + Phi.T @ (l[:, None] * Phi)
    assert np.max(np.abs(mu - np.linalg.solve(Prec, Phi.T @ (l * y)))) < 1e-8
    assert np.max(np.abs(Sig - np.linalg.inv(Prec))) < 1e-8


# -- C2: EM updates are monotone ascent on the marginal likelihood -----------
@pytest.mark.parametrize("feature_only", [False, True])
def test_c2_em_ascent(feature_only):
    X, y, oi, _, _, _ = make_problem(2)
    gamma = np.ones(8) * 1.0; lam = np.ones(len(y)) * 1.0
    ll_prev = marginal_loglik(X, y, gamma, lam)
    for _ in range(20):
        g_new, l_new = em_update(X, y, gamma, lam)
        gamma = np.clip(g_new, 1e-8, 1e8)
        if not feature_only:
            lam = np.clip(l_new, 1e-8, 1e8)
        ll_new = marginal_loglik(X, y, gamma, lam)
        assert ll_new >= ll_prev - 1e-6
        ll_prev = ll_new


# -- C1: joint ARD recovers feature sparsity (irrelevant 1/γ -> 0) ------------
def test_c1_sparsity():
    X, y, oi, _, _, tw = make_problem(3)
    g, l, _ = fit_joint(X, y, feature_only=False, seed=3)
    irr = 1.0 / g[np.where(np.abs(tw) < 1e-9)[0]]
    assert np.max(irr) < 0.1


# -- C1: outliers get smaller λ than clean samples ---------------------------
def test_c1_outlier_downweighting():
    X, y, oi, _, _, _ = make_problem(4)
    g, l, _ = fit_joint(X, y, feature_only=False, seed=4)
    assert np.mean(l[oi]) < np.mean(np.delete(l, oi))


# -- C1: joint is more robust than feature-only (test RMSE) ------------------
def test_c1_robust_vs_feature_only():
    X, y, oi, Xt, yt, _ = make_problem(5)
    gj, lj, _ = fit_joint(X, y, feature_only=False, seed=5)
    gf, lf, _ = fit_joint(X, y, feature_only=True, seed=5)
    rj = np.sqrt(np.mean((predict(X, y, Xt, gj, lj) - yt) ** 2))
    rf = np.sqrt(np.mean((predict(X, y, Xt, gf, lf) - yt) ** 2))
    assert rj < rf


# -- Negative control: feature-only (homoscedastic) CANNOT down-weight outliers
def test_negative_feature_only_no_per_sample():
    X, y, oi, _, _, _ = make_problem(6)
    g, l, _ = fit_joint(X, y, feature_only=True, seed=6)
    # all λ equal (homoscedastic) -> no per-sample down-weighting
    assert np.max(np.abs(l - l[0])) < 1e-9

````


````output
........                                                                 [100%]
8 passed in 0.40s

````


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_06257dda6d51", "created_at": "2026-07-17T06:44:47+00:00", "title": "Method & environment"}
-->
**Paper:** "Joint Model and Data Sparsification via the Marginal Likelihood" (arXiv 2605.29908, ICML 2026, vMcu1h3fOV). Clean-room from PDF (standard Sparse Bayesian Learning / ARD framework, Tipping 2001, extended with per-sample noise precision).

**Environment:** Python 3.12, numpy; CPU, <1 s.

**Implementation (repro/src):** `ard.py` (closed-form marginal likelihood, Gaussian posterior, EM/MacKay γ & λ updates, joint vs feature-only fit, predict); `run_claims.py` (orchestrator). 8/8 pytest tests: marginal-likelihood-is-Gaussian, posterior-definition, EM-ascent (joint & feature-only), sparsity recovery, outlier down-weighting, robustness vs feature-only, negative control (feature-only = homoscedastic, no per-sample λ).


---
<!-- trackio-cell
{"type": "code", "id": "cell_59d23776eab8", "created_at": "2026-07-19T00:22:15+00:00", "title": "Expanded repair test suite (22 tests)", "command": [".venv/bin/python", "-m", "pytest", "repro/tests", "-q"], "exit_code": 0, "duration_s": 2.875}
-->
````bash
$ .venv/bin/python -m pytest repro/tests -q
````

exit 0 · 2.9s


````output
......................                                                   [100%]
22 passed in 2.44s

````
