# Claim 4 source audit

Primary source: ar5iv rendering of arXiv:2605.29908, retrieved 2026-07-23,
SHA-256 `0a3ed827e899a92f2ed3b1e877ea2f6456ec3d827af184045acac3b8ebd4cb49`.

Figure 3 (`S5.F3`) states `n=500`, `d=50`, ten trials, and says the largest
heteroscedastic-versus-homoscedastic recovery gains occur at high weight
sparsity and low contamination. Appendix D.2 fixes the weight grid at
`rho=0.2`, multiplier 10, and varies support and sigma; it fixes the data grid
at support 0.2 and sigma 0.2 and varies rho and multiplier.

Figure 4 (`S5.F4`) uses real Boston data (`n=506`), a 20% test split, twenty
trials, an RBF RVM with a fixed scalar lengthscale, and contamination from 0%
through 30%. Section 5.3 explicitly states that the RVM basis has
`d=n=506`. Table 6 (`A5.T6`) gives the seven l2-IRLS RMSE means. Section 5.3
claims consistently low RMSE and improved robustness relative to non-robust
alternatives; the paper describes the comparison with Student-t and Huber as
competitive, not a universal win. The Figure 4 PNG was retrieved from the
ar5iv asset URL on 2026-07-23 with an explicit browser User-Agent and has
SHA-256 `cb5857bfaaabc3bff2b8964b1e70b8070ddd3c914aa0ccc3a7daa73629be71a8`.

The source does not publish split seeds, the scalar lengthscale value, or the
precise kernel-construction order. To implement `d=n=506` literally, this
reproduction uses all standardized Boston covariates as kernel centers and
computes their median-distance lengthscale. This is transductive use of test
covariates, not test targets; feature standardization is fit on training rows.
The paper reports scikit-learn for Huber but no solver settings. Direct
`HuberRegressor` fits failed to terminate after 5,000 iterations on the
rank-deficient full-center design, so the final candidate uses a deterministic
fixed-threshold convex Huber-loss IRLS baseline with explicit stationarity
checks. These declared choices can verify alignment but a divergent result
would not falsify a universal statement.
