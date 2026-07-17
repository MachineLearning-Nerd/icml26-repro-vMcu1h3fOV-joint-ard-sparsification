# Conclusion


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_16d1b8888498", "created_at": "2026-07-17T06:44:48+00:00", "title": "Executive summary", "pinned": true, "pinned_at": "2026-07-17T06:44:49+00:00"}
-->
**Both claims reproduced on CPU, verified.**

Joint model+data ARD optimizes a single closed-form marginal likelihood (y ~ N(0, ΦΓ⁻¹Φᵀ+Λ⁻¹)) with a closed-form Gaussian posterior and EM/MacKay updates that are monotone ascent (8/8). It recovers feature sparsity (10/10), automatically down-weights outliers via per-sample λ (10/10), and is robust — lower test RMSE than feature-only ARD in 10/10 trials.

**Verdict:** C1 ✅ · C2 ✅. 8/8 tests pass.

## Scope & cost
| | This reproduction | Full replication |
|---|---|---|
| Scope | Both claims; 8–10 random sparse-signal problems with outliers; EM ascent; closed-form posterior | official-code real-data benchmarks |
| Hardware | 4 vCPU (CPU only) | any CPU |
| Time | <1 s | — |
| Cost | \$0 | — |
| Outcome | Both claims verified | — |
