# Joint Model & Data Sparsification via the Marginal Likelihood (ARD) — ICML 2026 Reproduction

Reproduction of **"Joint Model and Data Sparsification via the Marginal
Likelihood"** (arXiv [2605.29908](https://arxiv.org/abs/2605.29908), ICML 2026,
OpenReview [`vMcu1h3fOV`](https://openreview.net/forum?id=vMcu1h3fOV)).

Extends Sparse Bayesian Learning (ARD) to **jointly** learn feature relevancies
(γ) and **per-sample** noise precisions (λ) under one marginal likelihood —
symmetric model+data pruning that preserves Gaussian conjugacy.

## Claims reproduced

| # | Claim | Status |
|---|---|---|
| **C1** | Joint (per-sample λ) ARD yields **sparse + robust** models vs feature-only. | ✅ Verified (10/10) |
| **C2** | Single marginal-likelihood objective; **closed-form** posterior + EM/MacKay updates (ascent). | ✅ Verified |

## Method

Model `y = Φθ`, `θ ~ N(0, Γ⁻¹)`, `y|θ ~ N(Φθ, Λ⁻¹)` (heteroscedastic per-sample
noise). Closed forms:
- marginal likelihood `y ~ N(0, ΦΓ⁻¹Φᵀ + Λ⁻¹)`
- posterior `θ|y ~ N(μ, Σ)`, `Σ⁻¹ = Γ + ΦᵀΛΦ`, `μ = ΣΦᵀΛy`
- EM updates: `γ_j ← 1/(Σ_jj+μ_j²)`, `λ_i ← 1/((y_i−Φ_iμ)²+Φ_iΣΦ_iᵀ)`

* `repro/src/ard.py`, `repro/src/run_claims.py`, `repro/tests/test_ard.py` (8 tests).

## How to run
```bash
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install numpy pytest
python -m pytest repro/tests/test_ard.py -q
python repro/src/run_claims.py
```

## Headline results (CPU)
**C2:** EM is monotone ascent on the marginal likelihood (8/8); closed-form posterior matches the definition (<1e-8).
**C1:** feature sparsity recovered (10/10); outliers down-weighted via small λ (10/10, e.g. 0.02 vs 5×10⁶); joint test RMSE < feature-only (10/10, e.g. 0.12 vs 0.98).

## Scope & cost
| | This reproduction | Full replication |
|---|---|---|
| Scope | Both claims; 8–10 random sparse-signal+outlier problems | + official-code real benchmarks |
| Hardware | 4 vCPU (CPU only) | any CPU |
| Time | <1 s | — |
| Cost | $0 | — |
| Outcome | Both claims verified | — |
