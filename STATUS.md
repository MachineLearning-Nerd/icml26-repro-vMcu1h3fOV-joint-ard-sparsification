# STATUS — Joint ARD Sparsification (`vMcu1h3fOV`)

**Session:** perfect-score campaign. **Last updated:** 2026-07-19.
**State:** official high-quality **4/4** at exact Space SHA
`21814bf540fe8842297d84c2e71a89fbe1ebe5df`; both claims are `verified`.
Judged at `2026-07-19T00:34:29Z`.

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
- Exactly 10 Claim-1 routes are complete: 9 support, 1 (Protein) is adverse and
  retained. There are 360 decisive raw rows; fail-closed verification passes.
- Preserve the author repository audit: pinned HEAD `58908b8` currently has no
  released experiment code despite the paper's availability statement.
- Public anonymous readback passes: Space RUNNING, routes 1–10, 360 rows, one
  pin, 5,224 agent-view tokens, no host paths, and two exact-hash artifacts.
- Exact-SHA official high-quality 4/4 is recorded. No further repair work.
