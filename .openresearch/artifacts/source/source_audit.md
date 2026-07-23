# Paper source audit

- Paper: *Joint Model and Data Sparsification via the Marginal Likelihood*
- arXiv: `2605.29908v1`
- URL: `https://ar5iv.labs.arxiv.org/html/2605.29908`
- Retrieved: `2026-07-23` (Asia/Kolkata)
- Retrieval User-Agent: `OpenResearch-Reproduction/1.0 paper-provenance-audit`
- Bytes: `1,217,517`
- SHA-256: `0a3ed827e899a92f2ed3b1e877ea2f6456ec3d827af184045acac3b8ebd4cb49`
- Preserved source: `2605.29908.html`
- Author repository: `alextimans/robust-sbl@58908b880b1367dd417b34160b8e324c54b44e42`
- Author repository state: README, license, and gitignore only; latest message
  says “code to come”.

## Audited anchors and quantifiers

- Formulation and updates: `S3`, `S3.Ex5.m1`, `A2.Ex35.m1`,
  `A2.Ex37.m1`, `A2.Ex41.m1`, `A2.Ex42.m1`, `A2.Ex48.m1`,
  `A2.Ex50.m1`, `A2.Ex55.m1`, `A2.Ex60.m1`, `A2.Ex63.m1`,
  `A2.Ex66.m1`, `A2.Ex68.m1`.
- Table 3: `S5.T3`. Energy/Carbon/Protein, 10% response contamination,
  10 trials, mean ± one standard deviation. The paper reports EM reductions
  of 32.07% and 69.91% for Energy and Carbon, and a MacKay reduction of
  45.12% for Protein. Reported ESS(y) is near 90%; ESS(theta) spans roughly
  5–40% across the SBL rows.
- Figure 3: `S5.F3`, `S5.SS1`, and Appendix-D synthetic protocol. Exactly
  `n=500`, `d=50`, `n_test=1000`, averaged over 10 trials. Weight grid fixes
  contamination `rho=0.2` and multiplier `m=10`; data grid fixes support
  fraction `0.2` and inlier noise `sigma=0.2`. The paper states that the
  largest heteroscedastic-over-homoscedastic gains occur with high weight
  sparsity and low contamination.
- Figure 4: `S5.F4`, `S5.SS3`, and `A5.T6`/`A5.T7`. Real Boston housing,
  `n=506`, 20% uncontaminated test split, 20 trials, contamination levels
  0–30% in 5-point increments, RBF RVM with fixed scalar lengthscale and
  l2-IRLS. The prose claims consistently low RMSE and improved NLL relative
  to non-robust alternatives; it does not claim universal superiority to
  every robust comparator.

## Assumptions and boundaries

The tabular and RVM claims are empirical, not universally quantified theorems.
Their contract is therefore the paper’s named datasets, preprocessing,
contamination process, optimizer family, split/trial counts, and reported
uncertainty. A nearby synthetic function or another optimizer does not test
the same claim. The author code is unavailable, so any remaining unspecified
initialization, stopping, or bandwidth detail must be declared as a deviation.
