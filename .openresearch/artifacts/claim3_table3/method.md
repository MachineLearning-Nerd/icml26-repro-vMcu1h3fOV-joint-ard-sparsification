# Claim 3 method — conservative authorized endpoint

This sibling tests a conservative point explicitly allowed by Appendix D.1:
arithmetic damping 0.005, a 300-step weight-only warm-up, and a
heteroscedastic variance update every five steps. Hyperparameters initialize
at 0.1 and clip to `[1e-3, 1e2]`. Convergence uses the paper's relative
log-parameter criterion at 1e-6 for five consecutive steps, with a declared
5,000-iteration safety limit.

Energy and Carbon use Algorithm 1 (EM); Protein uses Algorithm 2 (MacKay).
Each joint fit is compared with a homoscedastic fit using the same optimizer,
preprocessing, RFF draw, and seed. Ten deterministic seeds are used. The
contamination protocol and data loader are inherited unchanged from the frozen
baseline. The experiment prints every raw row plus a final JSON payload so the
OpenResearch run log is a complete evidence channel.

This is a protocol endpoint, not a tuned answer. It was selected before seeing
this sibling's outputs because the fast endpoint collapsed sample ESS far below
Table 3 and had not converged after 1,500 steps.

This descendant also corrects the preprocessing order to the explicit Section
5.2 wording: raw input features are standardized and then mapped through 256
RFFs. The inherited reproduction had applied an additional, undocumented
column-wise standardization after the RFF mapping. No other setting changes.
