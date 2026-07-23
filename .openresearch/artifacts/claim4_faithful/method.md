# Claim 4 method

The Figure 3 stage reuses the fixed campaign's immediately preceding exact
synthetic-grid regeneration and adds the missing homoscedastic weight-recovery
measurement. It evaluates 18 cells with ten seeds each and computes joint-minus-
homoscedastic weight recovery by support plus joint-minus-chance outlier
recovery by contamination.

The Figure 4 stage loads pinned OpenML Boston dataset 531 and validates its
canonical matrix hash. For each of 20 deterministic splits and seven
contamination levels, it standardizes training features and targets, builds an
RBF basis from training centers only, and fixes a scalar median-distance
lengthscale before contamination. It applies Appendix D.2's signed
`3*N(1,0.25²)` target corruption. Joint ARD, its homoscedastic negative control,
and Student-t use the same 60-step evidence solver; Huber uses scikit-learn as
specified by Appendix D.2. Evaluation is on the untouched real Boston test
targets in original units.

The predeclared noninferiority margin is 0.5 Boston RMSE, approximately the
paper's per-cell trial standard deviation in Table 6. A deterministic
independent checker recomputes summaries, checks the exact 20x7 design and
dataset hash, verifies equal homoscedastic noise, and rejects three mutations.
