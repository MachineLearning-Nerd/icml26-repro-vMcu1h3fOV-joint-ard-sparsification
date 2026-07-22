# Closed-form joint-ARD coordinate-ascent updates

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_ard_i", "created_at": "2026-07-21T22:50:00+00:00", "title": "Closed-form EM updates for the heteroscedastic objective"}
-->
### Claim — VERIFIED

The heteroscedastic sparse-Bayesian objective admits **closed-form** coordinate-ascent (EM) updates: `γ_j ← 1/(μ_j²+Σ_jj)` (ARD weight precision) and `λ_i ← r_i²+x_iᵀΣx_i` (per-sample noise variance). Implemented on synthetic data with 3 relevant features and 3 injected outliers.

---
<!-- trackio-cell
{"type": "code", "id": "cell_ard_r", "created_at": "2026-07-21T22:50:00+00:00", "title": "Executed joint-ARD reproduction", "command": ["python", "repro/src/verify_ard.py"], "exit_code": 0, "duration_s": 1.0}
-->
````bash
$ python repro/src/verify_ard.py
````

````output
claim: JointARD_closed_form_EM_updates
Joint ARD closed-form EM: gamma_j <- 1/(mu_j^2+Sigma_jj), lambda_i <- r_i^2 + x_i^T Sigma x_i.

(i) marginal likelihood: -240.237 -> 55.098, monotone increasing: True
(ii) feature pruning: relevant gamma mean=0.377 vs irrelevant gamma min=337.9 -> irrelevant pruned (gamma->inf): True
(iii) robustness: outlier lambda mean=66.42 vs inlier lambda mean=0.065 -> outliers down-weighted: True
     recovered relevant weights [2.542, -1.82, 1.217] (truth [2.5,-1.8,1.2]) close: True
verdict: supports
````

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_ard_c", "created_at": "2026-07-21T22:50:00+00:00", "title": "Interpretation"}
-->
**VERIFIED.** The closed-form EM updates increase the marginal likelihood **monotonically** (`−240 → 55`); irrelevant features' precision `γ→∞` (min `337.9` vs relevant `0.38`) — exact feature pruning; and outlier samples acquire large noise variance (`λ 66` vs inlier `0.065`) — the robustness mechanism. Recovered weights `[2.54,−1.82,1.22]` match the sparse truth `[2.5,−1.8,1.2]`.