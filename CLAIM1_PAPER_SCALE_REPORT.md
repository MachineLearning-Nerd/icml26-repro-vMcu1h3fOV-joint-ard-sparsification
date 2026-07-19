# Claim 1 paper-scale repair — exactly 10 approaches

## Outcome

This repair materially supports Claim 1 at paper scale, with an important
boundary case. Exactly ten pre-registered approaches were executed: the
paper's synthetic experiment and all nine tabular benchmarks. Nine approaches
support joint model-and-data ARD; Protein is adverse and is retained. No
eleventh approach was created.

Across the nine real benchmarks, joint ARD beats its homoscedastic model-only
ablation in 80/90 independently seeded 10%-contaminated trials (one-sided sign
test p=5.26e-15). Mean top-10%-noise outlier recall is 95.38% versus 10% chance,
and mean effective weight support is 16.88%, demonstrating simultaneous data
and model sparsification. The mean relative RMSE reduction is 31.94%, although
this aggregate must be read alongside the adverse Protein result.

The synthetic route uses n=500, d=50, ntest=1,000, 18 frozen grid cells, and ten
trials per cell. It obtains 96.67% mean weight-support recall, 61.50% mean
outlier-support recall, and a positive mean joint-vs-homoscedastic RMSE gain of
0.0593 (paired Wilcoxon p=2.41e-29).

## Exactly-ten results

| Route | Approach | Joint RMSE | Model-only RMSE | Wins | Outlier recall | Weight ESS | Result |
|---:|---|---:|---:|---:|---:|---:|---|
| 1 | Synthetic grids | — | — | paired gain 0.0593 | 61.50% | — | supports |
| 2 | Boston | 3.7975 | 5.1315 | 10/10 | 95.00% | 26.78% | supports |
| 3 | Yacht | 1.7612 | 6.7154 | 10/10 | 100.00% | 19.57% | supports |
| 4 | Concrete | 7.1162 | 8.4918 | 10/10 | 97.44% | 16.84% | supports |
| 5 | Energy | 0.8347 | 3.3536 | 10/10 | 100.00% | 7.50% | supports |
| 6 | Carbon | 0.00326 | 0.04312 | 10/10 | 100.00% | 6.50% | supports |
| 7 | Protein | 6.1554 | 5.1090 | 0/10 | 81.05% | 18.63% | **adverse** |
| 8 | Power | 4.2575 | 4.6553 | 10/10 | 98.60% | 23.81% | supports |
| 9 | Kin8nm | 0.14368 | 0.15535 | 10/10 | 91.90% | 18.94% | supports |
| 10 | Elevators | 0.003084 | 0.003400 | 10/10 | 94.45% | 13.32% | supports |

RMSE values are means over the ten contaminated trials in original target
units. The synthetic design has multiple scale/noise cells, so a single RMSE
mean would be misleading; its paired gain and recovery metrics are reported.

Machine invariant: `approaches_executed=10`, route numbers 1–10 exactly once,
`approaches_supported=9`, `approaches_adverse_or_mixed=1`, and
`route_11_executed=false`. Baselines, conditions, seeds, grids, and sensitivity
checks are subchecks within those routes.

## Protocol and primary sources

- Paper: arXiv 2605.29908v1, Sections 5.1–5.2 and Appendix D.2.
- Cited author repository: `alextimans/robust-sbl` pinned at
  `58908b880b1367dd417b34160b8e324c54b44e42`. At audit time it contains a
  README and license but no executable experiments, despite the paper's code
  availability statement. The repair is therefore clean-room and equation
  faithful.
- Synthetic: Gaussian X; sparse Gaussian weights; n=500, d=50, ntest=1,000;
  the paper's fixed heatmap conditions; explicit three-by-three weight and data
  grids; ten seeds.
- Tabular: 20% uncontaminated test split, standardized targets, 256 random
  Fourier features, clean and 10% response-contamination conditions, amplitude
  a=3, ten seeds, and random 2,000-row training cap for larger data.
- Each contaminated target receives the paper's signed Rademacher perturbation
  with magnitude distributed N(1,0.25²). Test targets remain untouched.
- Comparators are the joint EM estimator, its homoscedastic model-ARD ablation,
  RidgeCV, and Huber regression. All share the same split and RFF design.

The nine source schemas match Appendix D.2 exactly: Boston 506×13, Yacht
308×6, Concrete 1,030×8, Energy 768×8, Carbon 10,721×5, Protein 45,730×9,
Power 9,568×4, Kin8nm 8,192×8, and Elevators 16,599×18. UCI archives are
SHA-256 pinned; OpenML datasets are ID/version/MD5 pinned. Canonical numeric
matrix hashes appear in `outputs/claim1_paper_scale/validation.json`.

## Adversarial and negative checks

- Route 11 is rejected by code and unit test.
- The verifier independently reads every CSV, checks route/seed/condition
  coverage, recomputes mean RMSE, wins and recall, verifies every CSV hash, and
  checks the Markdown ledger independently of `validation.json`.
- Protein is not dropped or relabeled. It identifies 81.05% of injected
  outliers and stays sparse, but joint prediction is worse in all ten trials.
  Three Appendix-D.1 warm-up/update schedules at seed 0 also remain adverse
  (joint RMSE 5.58–5.89 vs model-only 5.22), so the limitation is robust to
  that local schedule sensitivity.
- Dataset source digests, raw dimensions, clean conditions, contaminated
  conditions, and all 360 decisive rows are retained.

## Limitations

The public author repository did not provide the claimed experiment code, so
exact optimizer hyperparameters and RFF bandwidth choices could not be copied.
This repair uses a median-distance RFF bandwidth and the paper's EM equations
with its recommended warm-up, damping, clipping, update cadence, and final
posterior recomputation. The fixed 80-step budget did not meet the stringent
log-parameter convergence flag; raw rows disclose this, and the route-7
120-step sensitivity remained adverse. The results therefore establish strong
finite-budget evidence, not exact parity with the unreleased implementation.

The Huber baseline emitted iteration-cap warnings on part of the Carbon run;
it is ancillary, and none of the claim-support decisions use Huber. The
decisive comparator is the same EM implementation with per-sample variance
enabled versus disabled.

## Reproduction and artifacts

```bash
uv venv --python 3.12 .venv
uv pip install -r requirements.txt
.venv/bin/python -m pytest repro/tests -q
.venv/bin/python repro/src/claim1_paper_scale.py
.venv/bin/python repro/src/route07_sensitivity.py
.venv/bin/python repro/src/verify_claim1_paper_scale.py
```

The decisive run took 1,445.58 CPU-wall seconds and produced 360 raw rows.
`validation.json` SHA-256 is
`26e587d679b888dcbc3b856c25606a2eabee476252a5f373ac500f04c4d2c6d9`;
the retained Protein sensitivity JSON SHA-256 is
`b800289ffdf93a0d93037075347cf544faf7711d27002d1d6e7e340b2e3fff88`.
The verifier terminates with:

```text
CLAIM1_VERIFY_PASS approaches=10 routes=1..10 rows=360 supported=9 adverse=1 route11=false sensitivity_route=7
```
