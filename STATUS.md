# STATUS — Joint ARD Sparsification (`vMcu1h3fOV`)

**Session:** perfect-score campaign. **Last updated:** 2026-07-19.
**State:** official 3/4; exactly-10 Claim-1 paper-scale repair scoped.

## Source
- arXiv 2605.29908. Clean-room from PDF (standard SBL/ARD + per-sample noise).

## Evidence (locally complete)
- **C1 verified:** joint ARD sparse (10/10) + robust (10/10 lower RMSE than
  feature-only) + down-weights outliers (10/10).
- **C2 verified:** closed-form marginal likelihood + Gaussian posterior (matches
  def <1e-8); EM monotone ascent (8/8).
- **8/8 pytest tests pass** (<1 s).
- Trackio complete/tagged/pinned/command-captured.

## Next
- Freeze and execute exactly 10 Claim-1 routes: the paper's n=500,d=50
  synthetic experiment plus its nine real tabular datasets. Do not add route 11.
- Preserve the author repository audit: pinned HEAD `58908b8` currently has no
  released experiment code despite the paper's availability statement.
- Require paper-scale row counts, clean/10%-contaminated conditions, realistic
  baselines, uncertainty, raw results, fail-closed verification, public
  readback, and an exact-SHA official 4/4 verdict.
