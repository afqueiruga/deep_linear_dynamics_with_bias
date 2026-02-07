# Research Synthesis Log

Date: 2026-02-07

## Sources Synthesized
- `/Users/afq/Documents/Research/deep_linear/AGENT_MEMORY.md`
- `/Users/afq/Documents/Research/deep_linear/Background.md`
- `/Users/afq/Documents/Research/deep_linear/LOG.md`
- `/Users/afq/Documents/Research/deep_linear/THEORY_LOG.md`

## Consolidated Hypotheses
1. Factorized deep linear models (`A = WU`, and deeper variants) induce an implicit low-complexity bias, often seen as low effective rank / low nuclear norm solutions.
2. In partially identifiable settings (full-rank `A*`, low-rank `X`), this bias should appear as lower learned mass in unidentifiable directions:
   - `model_nullX_norm = ||A_hat Q_x||_F`.
3. Mode-wise dynamics in deep linear training are selective: stronger modes activate earlier/faster; weak modes remain suppressed longer.
4. Depth may strengthen or destabilize bias depending on optimization regime (width, LR schedule, balancing, and scale dynamics).

## Theory Results So Far
- Symbolic derivations (2026-02-06) for 2-layer deep linear models confirm:
  - Gradient flow for factorized parameters under whitened squared loss.
  - Induced end-to-end dynamics with Gram-matrix preconditioning.
  - Balanced/aligned single-mode logistic law:
    - `ds/dt = 2 s (sigma - s)`.
- Interpretation:
  - Small modes grow slowly; active modes accelerate then saturate.
  - This provides a mechanism for effective low-rank behavior.
- Scope note:
  - Exact logistic form is for balanced/aligned reduction; general multi-mode settings include coupling.

## Empirical Results So Far
1. 2-layer deep vs shallow (multiple runs):
   - Deep often keeps lower null/off-support components while fitting support well.
   - In low-rank-`X` setups, deep achieves lower `model_nullX_norm` than shallow while matching learnable support.
2. Low-rank-`X` identifiability experiments:
   - Very low train loss can coexist with high full-matrix relative error because `A*Q_x` is unidentifiable.
   - Therefore, support-fit and `model_nullX_norm` are the key metrics, not loss alone.
3. Noise robustness:
   - With label noise, support-fit errors rise as expected, but deep-vs-shallow nullspace ordering persists.
4. Depth sensitivity:
   - 3-layer models are less uniform than 2-layer:
     - Some widths perform well.
     - `Deep3(r=500)` can fit support while carrying large nullspace mass and inflated spectrum, exceeding shallow in nullspace magnitude.

## Main Conclusions
1. Implicit bias is present but regime-dependent.
2. 2-layer deep linear models provide the clearest and most consistent nullspace-suppression advantage over shallow baselines in this project’s settings.
3. Increasing depth does not monotonically improve implicit regularization:
   - 3-layer behavior can be fragile and sensitive to optimization/scale effects.
4. Training loss is insufficient in partially identifiable problems; decomposition metrics are essential.

## Recommended Next Course of Action
1. Add balancing/scale diagnostics for deep-3 runs:
   - Per-layer Fro norms, top singular values, and Gram mismatch measures across adjacent layers.
2. Run controlled interventions for deep-3 (`r=500` and one medium width):
   - Baseline vs weight decay vs explicit balancing penalty vs both.
   - Evaluate whether nullspace mass decreases without harming support fit.
3. Quantify robustness with multi-seed statistics (>=10 seeds):
   - Report mean/std (and confidence intervals) for `support_fit_err` and `model_nullX_norm` across deep-2, deep-3, shallow.
4. Extend theory minimally to 3-layer mode dynamics in balanced reductions:
   - Use it to predict/interpret when scale growth causes high-nullspace interpolants.

## Current Working Claim
In partially identifiable linear regression, factorized deep models can select lower-nullspace interpolants than shallow models, but this advantage depends strongly on depth-width-optimization regime; 2-layer is consistently beneficial here, while 3-layer can either help or fail depending on stability and balancing.
