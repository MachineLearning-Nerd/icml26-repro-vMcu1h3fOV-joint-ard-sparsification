# Claim 1 — Sparse + robust


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_aa22131504ac", "created_at": "2026-07-17T06:44:46+00:00", "title": "C1: joint ARD yields sparsity AND robustness"}
-->
**Claim 1:** jointly learning feature (γ) AND sample (λ) relevancies yields **sparse** models that are **robust** to data contamination, outperforming feature-only (homoscedastic) ARD.

Across 10 random sparse-signal problems with 8 injected outliers each:
- **Feature sparsity recovered (10/10):** irrelevant features pruned (1/γ → ~0).
- **Outliers down-weighted (10/10):** per-sample λ on outliers ≪ λ on clean samples (e.g. 0.02 vs 5×10⁶), so the contamination is automatically de-emphasized.
- **Robust prediction (10/10):** joint test RMSE < feature-only test RMSE in every trial (e.g. 0.12 vs 0.98) — feature-only ARD, with shared noise precision, is corrupted by the outliers.

The single marginal-likelihood objective thus performs symmetric model+data pruning.
