# Claim 4 method

The Figure 3 stage reuses the fixed campaign's immediately preceding exact
synthetic-grid regeneration and adds the missing homoscedastic weight-recovery
measurement. It evaluates 18 cells with ten seeds each and computes joint-minus-
homoscedastic weight recovery by support plus joint-minus-chance outlier
recovery by contamination.

The Figure 4 stage loads pinned OpenML Boston dataset 531 and validates its
canonical matrix hash. For each of 20 deterministic splits and seven
contamination levels, it standardizes training features and targets, builds an
RBF basis using all 506 covariates as centers, and fixes a scalar median-distance
lengthscale before contamination. It applies Appendix D.2's signed
`3*N(1,0.25²)` target corruption. Following Section 5.3's explicit `d=n=506`,
all 506 standardized covariates are used as RBF centers, with no embedded bias.
This is transductive use of test covariates but never test targets; the feature
scaler is fit on training rows. Student-t uses the retained 60-step evidence
solver, and Huber uses scikit-learn as specified by Appendix D.2. Evaluation is
on the untouched real Boston test targets in original units.

Joint and homoscedastic RVMs use the Appendix-D.1 fast authorized endpoint:
arithmetic damping 0.02, 50 weight-only warm-up steps, variance updates every
two steps, initialization 0.1, clipping `[1e-6,1e6]`, and at most 1,500 steps.
Huber uses its own intercept because the shared 506-column kernel basis has no
bias column. It retains scikit-learn's default epsilon, ridge penalty, and
tolerance, with the maximum iteration count raised to 5,000; all fits must
terminate before that cap.

The predeclared noninferiority margin remains 0.5 Boston RMSE, approximately the
paper's per-cell trial standard deviation in Table 6. A deterministic
independent checker recomputes summaries, checks the exact 20x7 design and
dataset hash, verifies equal homoscedastic noise, and rejects three mutations.

Figure 3's source simultaneously says gains are largest under “high sparsity”
and “low contamination” and warns that recovery degrades for increasingly
sparse signals and scarce contamination. The categorical audit therefore
compares support 0.1/0.2 against 0.4 and contamination 0.05/0.1 against 0.2;
it does not demand that the single most extreme point be the maximum.

All matrix-heavy entrypoints cap their BLAS thread pool at eight through
`threadpoolctl`. This committed performance setting prevents oversubscription
on the 64-core `cpu-upgrade` host; it does not alter seeds, floating-point
precision, update equations, stopping rules, or the fixed campaign command.
