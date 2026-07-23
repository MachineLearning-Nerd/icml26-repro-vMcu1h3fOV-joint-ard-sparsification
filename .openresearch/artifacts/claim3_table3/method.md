# Claim 3 method — fast authorized endpoint

This child tests the fastest endpoint explicitly allowed by Appendix D.1:
arithmetic damping 0.02, a 50-step weight-only warm-up, and a heteroscedastic
variance update every two steps. Hyperparameters initialize at 0.1 and clip to
`[1e-6, 1e6]`. Convergence uses the paper's relative log-parameter criterion
at 1e-6 for five consecutive steps, with a declared 1,500-iteration safety
limit.

Energy and Carbon use Algorithm 1 (EM); Protein uses Algorithm 2 (MacKay).
Each joint fit is compared with a homoscedastic fit using the same optimizer,
preprocessing, RFF draw, and seed. Ten deterministic seeds are used. The
contamination protocol and data loader are inherited unchanged from the frozen
baseline. The experiment prints every raw row plus a final JSON payload so the
OpenResearch run log is a complete evidence channel.

This is a protocol endpoint, not a tuned answer. A second endpoint is warranted
if the result is numerically stable but incompatible with Table 3.
