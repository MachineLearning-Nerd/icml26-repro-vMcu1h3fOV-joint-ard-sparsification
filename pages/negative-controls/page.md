# Negative controls


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_1b9f3e53db7c", "created_at": "2026-07-17T06:44:48+00:00", "title": "Negative control"}
-->
**Feature-only (homoscedastic) ARD cannot down-weight outliers:** when λ is forced to a single shared noise precision (the standard SBL setting), all λ_i are equal — the model has no mechanism to identify or de-emphasize contaminated samples. Verified (max |λ_i − λ_1| ≈ 0). This is precisely the limitation the joint model fixes, confirming C1's content is non-trivial (the per-sample λ is what enables robustness).
