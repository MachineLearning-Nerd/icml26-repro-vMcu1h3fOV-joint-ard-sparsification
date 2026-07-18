# Claim 1 repair ledger — exactly 10 approaches

Target claim: jointly learning feature and per-sample relevancies through the
marginal likelihood yields prediction models that are both sparse and robust.
The official verdict is `toy` because the first reproduction used only ten
synthetic n=80, d=8 trials. This repair follows the paper's Appendix-D protocol
at its stated scale and freezes exactly 10 routes—no more and no fewer.

Method/baseline comparisons, clean/contaminated conditions, seeds, uncertainty,
source hashes, and negative controls are subchecks within their route. They are
not additional approaches.

Pinned primary sources:

- paper: arXiv `2605.29908v1`, Appendix D.2 and Tables 3, 8–12;
- author repository: `alextimans/robust-sbl@58908b880b1367dd417b34160b8e324c54b44e42`
  (the public repository currently contains the paper README but no experiment
  implementation, so the executable is an equation-faithful clean-room repair);
- tabular protocol: 20% uncontaminated test split, standardized targets,
  256 random Fourier features, clean and 10% target-contamination conditions,
  ten trials, and a 2,000-row training cap for large datasets.

| # | Approach | Paper-scale decisive output | State |
|---:|---|---|---|
| 1 | Synthetic sparse-signal and outlier recovery | n=500, d=50, ntest=1,000; paper sparsity/noise/contamination grid; ten trials | planned |
| 2 | Boston tabular regression | n=506, d=13; clean + 10% contamination; joint vs homoscedastic/sparse/robust baselines | planned |
| 3 | Yacht tabular regression | n=308, d=6; same frozen tabular protocol | planned |
| 4 | Concrete tabular regression | n=1,030, d=8; same frozen tabular protocol | planned |
| 5 | Energy tabular regression | n=768, d=8; first target; same frozen tabular protocol | planned |
| 6 | Carbon tabular regression | n=10,721, d=5; first target; capped paper training split | planned |
| 7 | Protein tabular regression | n=45,730, d=9; capped paper training split | planned |
| 8 | Power tabular regression | n=9,568, d=4; capped paper training split | planned |
| 9 | Kin8nm tabular regression | n=8,192, d=8; capped paper training split | planned |
| 10 | Elevators tabular regression | n=16,599, d=18; capped paper training split | planned |

Fail-closed invariant: the final machine-readable result must report
`approaches_executed == 10`, contain route numbers 1 through 10 exactly once,
and reject any route numbered 11 or higher. All failed, mixed, or adverse
results remain attached to their original route rather than being replaced.
