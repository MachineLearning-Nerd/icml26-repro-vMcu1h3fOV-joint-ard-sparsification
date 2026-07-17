# STATUS — Joint ARD Sparsification (`vMcu1h3fOV`)

**Session:** NewPaper. **Last updated:** 2026-07-17. **State:** locally complete; GitHub push pending; HF queued.

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
- Push GitHub `MachineLearning-Nerd/icml26-repro-vMcu1h3fOV-joint-ard-sparsification`.
- Publish `DineshAI/vMcu1h3fOV` after HF quota reset; verify tags/bucket; `under_verdict`.
