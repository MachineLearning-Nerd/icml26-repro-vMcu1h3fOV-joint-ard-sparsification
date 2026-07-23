# Claim 3 source audit

The primary source is the retrieved ar5iv rendering of arXiv:2605.29908,
SHA-256 `0a3ed827e899a92f2ed3b1e877ea2f6456ec3d827af184045acac3b8ebd4cb49`.
The result is Table 3 (`S5.T3`); the protocol is Appendix D.2 and the
implementation envelope is Appendix D.1.

Table 3 reports ten trials at 10% response contamination. The exact reductions
tested here are 32.07% (Energy, EM), 69.91% (Carbon, EM), and 45.12% (Protein,
MacKay). Appendix D.2 standardizes training targets, uses signed perturbations
`3 s_i δ_i` with `δ_i ~ N(1, 0.25²)`, an uncontaminated 20% test split, a
2,000-row training cap, and 256 random Fourier features.

Appendix D.1 authorizes arithmetic EMA damping from 0.0005 through 0.02,
clipping from lower bounds 1e-6--1e-3 to upper bounds 1e2--1e6, 50--300
weight-only warm-up steps, and heteroscedastic updates every 2--5 iterations.
It gives an example initialization of 0.1 and requires a relative log-parameter
tolerance of 1e-6 for five patience steps.

Section 5.2 states that features are standardized and mapped via random Fourier
features. It does not state that the resulting RFF columns are standardized a
second time. This descendant follows the stated order literally.

The paper does not state the ten split/RFF seeds, RFF bandwidth rule, exact
point in each implementation range, or a maximum iteration count. The linked
author repository contains no experiment code at commit
`58908b880b1367dd417b34160b8e324c54b44e42`. Those omissions prevent a
divergent run from falsifying this non-universal empirical table entry.
