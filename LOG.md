# Experiment Log

## 2026-02-06

### Context
- Goal: understand training dynamics of deep linear models, especially effective low-rank structure.
- Comparison: deep linear factorized model (`A = WU`) vs shallow linear model (`A = B`).
- Environment: `python3.11`.

### Code changes made
- Refactored training flow into reusable helper `train_model(...)` in `/Users/afq/Documents/Research/deep_linear/script.py`.
- Added projector-based error decomposition from target `A*` SVD:
  - `P_left, P_right` project onto rank-`k` support of `A*`.
  - `P_left_perp, P_right_perp` project onto orthogonal complements.
- Extended `summarize_A(...)` to compute:
  - `err_support = ||P_left (A - A*) P_right||_F`
  - `err_null = ||P_left_perp (A - A*) P_right_perp||_F`
  - `err_mixed` (cross-term residual)
  - `support_frac = err_support / ||A - A*||_F`
  - `null_frac = err_null / ||A - A*||_F`
- Updated per-epoch logs to include these decomposition metrics.
- Changed shallow model init from zeros to small random initialization (`0.01 * randn`) for symmetry with deep init.
- Added explanatory comments in `script.py` for decomposition terms and fractions.

### Experiment run
- Command: `python3.11 script.py`
- Core setup from script:
  - `n=20000`, `d=50`, `m=50`, target rank `k=5`, deep hidden width `r=30`, `lr=0.5`, `epochs=50`.

### Key observed outputs (latest run)
- Deep model final:
  - `rel_err=7.563e-01`
  - `support_err=5.607e+00`
  - `null_err=2.437e-02`
  - `mixed_err=1.557e-01`
- Shallow model final:
  - `rel_err=4.493e-01`
  - `support_err=3.322e+00`
  - `null_err=2.058e-01`
  - `mixed_err=1.579e-01`
- Qualitative takeaway from this run:
  - Deep keeps nullspace error very small throughout training.
  - Shallow reduces training error faster and lower overall in 50 epochs, but retains a noticeably larger nullspace component than deep.

### Notes / interpretation
- `support_frac` answers: "What fraction of current model error lies inside `A*`'s rank-`k` signal subspace?"
- `null_frac` answers: "What fraction of current model error lies in the orthogonal nullspace of `A*`?"
- In this run, `null_frac` stayed close to `0.00` for deep and around `0.06` for shallow in printed logs.

### Next candidate experiments
- Sweep learning rate and epochs to check if shallow nullspace error persists asymptotically.
- Repeat over multiple seeds and report mean/std of final `null_err` and `null_frac`.
- Compare against explicit regularization baselines (e.g., weight decay on `B`, nuclear-norm proxy diagnostics).

### Update: Module refactor (DeepLinear + nn.Linear)
- Refactored model definitions to `nn.Module` style:
  - Added `DeepLinear(nn.Module)` with parameters `W`, `U`, `forward`, and `end_to_end()`.
  - Replaced shallow raw parameter `B` with `nn.Linear(d, m, bias=False)`.
- Updated `train_model(...)` to consume a `model` directly and optimize `model.parameters()`.
- Matrix extraction for diagnostics now uses:
  - Deep: `model.end_to_end()`
  - Shallow: `model.weight`
- Retained prior behavior and logging metrics, including support/null decomposition.

### Validation run after refactor
- Command: `python3.11 script.py`
- Status: success.
- Final decomposition (post-refactor run):
  - Deep: `support=5.607e+00`, `null=2.437e-02`, `mixed=1.557e-01`
  - Shallow: `support=3.326e+00`, `null=2.055e-01`, `mixed=1.639e-01`

### Hyperparameter tuning to reduce loss

#### Trial H0 (previous baseline, from earlier run)
- Deep: SGD, `lr=0.5`, `epochs=50`, `batch_size=512`.
- Shallow: SGD, `lr=0.5`, `epochs=50`, `batch_size=512`.
- Outcome:
  - Deep final loss (epoch 50): about `1.293e-02`.
  - Shallow final loss (epoch 50): about `4.533e-03`.
- Assessment: losses not low enough.

#### Trial H1 (new settings)
- Deep: Adam, `lr=2e-2`, `epochs=400`, `batch_size=n` (full batch).
- Shallow: Adam, `lr=1e-2`, `epochs=400`, `batch_size=n` (full batch).
- Script changes:
  - Added `optimizer_name` support in `train_model(...)` with `sgd` and `adam`.
  - Added separate optimizer/lr knobs for deep and shallow.
  - Increased epochs and switched to full-batch updates.
- Command: `python3.11 script.py`
- Outcome:
  - Deep final loss (epoch 400): `1.820e-09`.
  - Shallow final loss (epoch 400): `7.792e-16`.
- Assessment: very low losses achieved for both models.

#### Selected configuration
- Keep Trial H1 as default in `script.py`.

### Update: Deep hidden-width sweep
- Added deep model width sweep in `script.py` with widths:
  - `r = k`
  - `r = d`
  - `r = 2d`
  - `r = 10d`
- Concretely with current dims (`k=5`, `d=50`): `deep_widths=[5, 50, 100, 500]`.
- Implementation notes:
  - Each width instantiates its own `DeepLinear` model.
  - All deep models are trained sequentially with the same optimizer settings.
  - Shallow `nn.Linear` baseline is still trained afterward for comparison.

#### Run
- Command: `python3.11 script.py`
- Config: Adam/full-batch, `epochs=400`, deep lr `2e-2`, shallow lr `1e-2`.

#### Final error decomposition by model
- `Deep(r=5)`: `support=8.158e-08`, `null=4.467e-07`, `mixed=3.367e-07`
- `Deep(r=50)`: `support=1.788e-06`, `null=2.145e-03`, `mixed=4.001e-05`
- `Deep(r=100)`: `support=4.586e-07`, `null=6.682e-04`, `mixed=1.240e-05`
- `Deep(r=500)`: `support=3.622e-02`, `null=1.527e-03`, `mixed=2.229e-02`
- `Shallow`: `support=1.575e-07`, `null=1.029e-06`, `mixed=6.697e-07`

#### Final spectra (top signal summary)
- `Deep(r=5)` matches target singular spectrum almost exactly.
- `Deep(r=50/100/500)` fit top-5 singular values well but retain larger residual tail than `Deep(r=5)` in this run.
- `Shallow` also matches top-5 with near-zero tail.

### Hyperparameter tuning: LR decay trial
- Request: stabilize deep large-width behavior (especially `Deep(r=500)`) after observing late-epoch degradation.

#### Trial H2 (added LR decay)
- Code change:
  - Added `ExponentialLR` scheduler inside `train_model(...)`.
  - New per-model knobs:
    - `deep_lr_decay_gamma`
    - `shallow_lr_decay_gamma`
  - Logged current LR in epoch printout.
- Settings:
  - Deep: Adam, `lr=2e-2`, `deep_lr_decay_gamma=0.99`
  - Shallow: Adam, `lr=1e-2`, `shallow_lr_decay_gamma=1.0` (no decay)
  - `epochs=400`, `batch_size=n`
- Command:
  - `python3.11 script.py` (filtered for key checkpoints)

#### Key result for `Deep(r=500)`
- With no decay (previous trial H1):
  - epoch 200 loss: `5.060e-12`
  - epoch 400 loss: `8.775e-07` (degraded)
- With decay (H2):
  - epoch 200 loss: `1.916e-12`
  - epoch 400 loss: `4.328e-16` (no late degradation)
- Final decomposition with decay:
  - `support=3.943e-08`, `null=6.084e-07`, `mixed=1.877e-07`

#### Decision
- Keep LR decay enabled for deep models (`deep_lr_decay_gamma=0.99`).

### Metric clarification: `null_err` vs `outside_err`
- Clarification from discussion:
  - Current `null_err` is the two-sided orthogonal block
    - `null_err = ||Q_left E Q_right||_F`, where `E = A_hat - A*`, `Q_left = I - P_left`, `Q_right = I - P_right`.
  - This is **not** all error outside support.

- 4-block decomposition view of `E` using row/column support projectors:
  - inside support: `P_left E P_right`
  - cross block 1: `P_left E Q_right`
  - cross block 2: `Q_left E P_right`
  - two-sided null block: `Q_left E Q_right`

- New metric added to match the intended question "all error outside support":
  - `outside_err = ||E - P_left E P_right||_F`
  - This includes all three off-support blocks (`P_left E Q_right`, `Q_left E P_right`, and `Q_left E Q_right`).

- Code updates in `script.py`:
  - Added `err_outside` and `err_outside_frac` to `summarize_A(...)`.
  - Updated init/epoch/final logging to include `outside_err` and `outside_frac`.

### Update: Experiment 2 (full-rank `A*`, low-rank `X`)
- Added a second setting to test identifiability from low-rank inputs.
- Construction:
  - `A*` is full rank (`k=min(m,d)`).
  - `X` is rank-`kx` via `X = Z Vx^T` with `kx=5`.
  - `Y = X A*^T`.
- Rationale:
  - Only `A*` restricted to `span(X)` is identifiable from data.
  - Components along `null(X)` cannot be inferred from training loss.

#### Metrics used in Experiment 2
- Let `P_x` project onto `span(X)` and `Q_x = I - P_x`.
- `support_fit_err = ||(A_hat - A*) P_x||_F` (learnable part fit error).
- `model_nullX_norm = ||A_hat Q_x||_F` (what model places in unidentifiable nullspace).
- `target_nullX_norm = ||A* Q_x||_F` (target mass in that nullspace; not learnable from data).

#### Run (same optimizer schedule)
- Deep widths: `[5, 50, 100, 500]`.
- Deep: Adam `2e-2`, decay gamma `0.99`.
- Shallow: Adam `1e-2`, no decay.
- `epochs=400`, full batch.

#### Key outputs (epoch 400)
- Training loss is very low for all models (`~1e-14`), but `rel_err` remains high (`~0.95`) because unidentifiable nullspace mismatch dominates matrix error.
- Identifiable/nullspace summary:
  - `LowX-Deep(r=5)`: `support_fit_err=3.913e-06`, `model_nullX_norm=3.681e+00`, `target_nullX_norm=2.156e+01`
  - `LowX-Deep(r=50)`: `support_fit_err=7.472e-07`, `model_nullX_norm=2.267e+00`, `target_nullX_norm=2.156e+01`
  - `LowX-Deep(r=100)`: `support_fit_err=6.758e-07`, `model_nullX_norm=2.313e+00`, `target_nullX_norm=2.156e+01`
  - `LowX-Deep(r=500)`: `support_fit_err=6.308e-07`, `model_nullX_norm=2.204e+00`, `target_nullX_norm=2.156e+01`
  - `LowX-Shallow`: `support_fit_err=7.620e-07`, `model_nullX_norm=4.884e+00`, `target_nullX_norm=2.156e+01`

#### Observation from this run
- All models recover the identifiable support similarly well (`support_fit_err` near zero).
- Deep models produce a smaller learned nullspace component than shallow in this setup (`~2.2-3.7` vs `~4.9`), but not exactly zero.

### Update: support projector source in Experiment 2
- Question addressed: what defines "support" when `A*` is full-rank but `X` is low-rank?
- Change made:
  - Support projector is now built from SVD of the observed training data `X_low`.
  - Using `X_low = U_x S_x V_x^T`, we define identifiable domain support as `span(V_x)`.
  - Projector used for decomposition: `P_x = V_x V_x^T` (feature/input space).
- Clarification:
  - `U_x` lives in sample/index space (`n`-dimensional), so it is not the right projector for decomposing parameter matrix error in `d`-dimensional input space.
  - For error on `A` (shape `m x d`), the relevant subspace is on the right (input-feature) side.
- Numerical rank handling:
  - Added relative singular-value threshold `sv_tol = 1e-6 * max(s)` to estimate rank robustly.
  - Verified output now reports: `X rank proxy kx=5, empirical rank from SVD=5`.

### Update: explicit learnable target matrix in Experiment 2
- Added explicit decomposition of full-rank target:
  - `A*_learnable = A* P_x` (your `A U U^T` form)
  - `A*_unlearnable = A* Q_x`
- Here `P_x` is built from SVD of training data `X_low` in feature space (`V_x V_x^T`).
- Added printed norms:
  - `||A*_learnable||_F`
  - `||A*_unlearnable||_F`
- Added direct learnable-fit metric for each model:
  - `learnable_target_err = ||A_hat P_x - A*_learnable||_F`
- This complements `support_fit_err = ||(A_hat - A*) P_x||_F` (numerically equivalent up to floating-point error).

### Hypothesis (Experiment 2)
- When `A*` is full-rank and `X` is low-rank, only `A*_learnable = A* U U^T` (support of training data) is identifiable.
- Hypothesis: deep linear models will recover the learnable support component and drive their own nullspace component (`A_hat (I-UU^T)`) toward zero, while shallow linear models will retain a larger nullspace component.

### Update: added per-epoch `model_nullX_norm` logging
- Added optional model-space null component metric to `train_model(...)`:
  - `model_nullX_norm = ||A_hat Q_x||_F`
- Enabled this for Experiment 2 (both deep widths and shallow baseline).
- This metric tracks exactly the quantity tied to the nullspace-bias hypothesis and is now logged every reporting epoch, alongside loss and error decomposition terms.

### Probe: longer run for slow implicit-regularization hypothesis
- Request: test whether deep nullspace mass decays with more epochs and less aggressive decay.

#### Trial H3 (Experiment 2 only)
- Changed low-`X` schedule:
  - `epochs_lowX=2000`
  - `deep_lr_lowX=1e-2`
  - `deep_lr_decay_gamma_lowX=0.999` (slower decay)
  - `shallow_lr_lowX=1e-2`, `shallow_lr_decay_gamma_lowX=1.0`
  - `svd_every_epochs_lowX=100`
- Kept Experiment 1 schedule unchanged.

#### Key checkpoints: `LowX-Deep(r=500)`
- epoch 400: `model_nullX_norm=2.019e+00`, `loss=6.411e-15`, `lr=6.702e-03`
- epoch 1000: `model_nullX_norm=2.019e+00`, `loss=6.355e-15`, `lr=3.677e-03`
- epoch 1500: `model_nullX_norm=2.019e+00`, `loss=6.329e-15`, `lr=2.230e-03`
- epoch 2000: `model_nullX_norm=2.019e+00`, `loss=6.315e-15`, `lr=1.352e-03`

#### Comparison at epoch 2000
- `LowX-Deep(r=500)`: `model_nullX_norm=2.019e+00`, `support_fit_err=1.752e-06`
- `LowX-Shallow`: `model_nullX_norm=4.884e+00`, `support_fit_err=5.267e-05`

#### Observation
- Under this longer/slower schedule, deep still has smaller nullspace mass than shallow, but `model_nullX_norm` for `Deep(r=500)` appears plateaued rather than continuing to decay toward zero.

### Update: efficiency mode + spectral tracking
- Experiment 1 is now disabled by default for efficiency:
  - `run_experiment_1 = False`
  - Script prints a message when skipped.
- Added optional per-epoch spectral logging in `train_model(...)`:
  - `top_sv_every_epochs` (cadence)
  - `top_sv_k` (number of singular values)
  - `top_sv_method` in `{exact, lowrank}`
- Cheap option implemented:
  - `top_sv_method="lowrank"` uses `torch.svd_lowrank(..., niter=2)` as an approximate top-spectrum tracker.
- Experiment 2 currently uses:
  - `top_sv_every_epochs_lowX = 200`
  - `top_sv_k = 10`
  - `top_sv_method_lowX = "lowrank"`
- Validation:
  - Confirmed Experiment 1 is skipped.
  - Confirmed top-10 singular values are printed at configured cadence for deep and shallow models in Experiment 2.

### Update: synchronized spectral logging cadence
- Changed `top_sv_every_epochs_lowX` to match the main metric cadence exactly:
  - `top_sv_every_epochs_lowX = svd_every_epochs_lowX`
- Result: top singular-value snapshots are now printed whenever the standard metric line is printed.

### Update: label-noise option + noisy run observations
- Added label-noise support in script:
  - Helper: `add_label_noise(Y, noise_std, generator=None)`
  - Config knobs:
    - `label_noise_std_exp1` (currently `0.0`)
    - `label_noise_std_lowX` (currently `1e-2`)
- Noise is additive Gaussian on labels: `Y_noisy = Y + eps`, `eps ~ N(0, noise_std^2)`.

#### Noisy run executed (Experiment 2)
- Command: `python3.11 script.py`
- Setting: `label_noise_std_lowX = 0.01`
- Key checkpoints for `LowX-Deep(r=500)`:
  - epoch 400: loss `1.003e-04`, `model_nullX_norm=2.171e+00`
  - epoch 1000: loss `1.003e-04`, `model_nullX_norm=2.171e+00`
  - epoch 1500: loss `1.003e-04`, `model_nullX_norm=2.171e+00`
  - epoch 2000: loss `1.003e-04`, `model_nullX_norm=2.171e+00`
- Shallow at epoch 2000:
  - loss `1.003e-04`, `model_nullX_norm=5.074e+00`

#### Observations
- Loss plateaus around `~1e-4`, consistent with a noise floor for `noise_std=1e-2`.
- Learnable support fit error for both deep and shallow is near `2.47e-3`.
- Deep still keeps a smaller learned nullspace component than shallow (`2.17` vs `5.07`).
- For `Deep(r=500)`, `model_nullX_norm` remains flat over long training under this schedule (no visible further decay).

### Update: print all `summarize_A` stats + compact formatting
- Training logs now print all scalar stats returned by `summarize_A` at each reporting epoch:
  - `rel_err`, `fro`, `nuc`, `spec`, `eff_rank`, `num_rank`
  - `err_support`, `err_outside`, `err_null`, `err_mixed`
  - `err_support_frac`, `err_outside_frac`, `err_null_frac`, `err_mixed_frac`
  - plus `model_nullX_norm` when enabled.
- Added compact singular-value formatter to avoid wrapped NumPy array output:
  - singular values now print as one-line comma-separated scientific notation.

### Update: table-style metrics rows with singular values inline
- Replaced key=value epoch logging with table-style rows.
- Each reporting epoch now emits exactly one row containing:
  - all scalar `summarize_A` metrics,
  - optional `model_nullX_norm`,
  - and `top_sv` as the final column.
- Removed separate extra singular-value lines per epoch; singular values are now appended in-row.
- Added an epoch-0 initialization row for each model in the same table format.

### Update: multi-noise Experiment 2 sweep + shorter runs
- Refactor:
  - Experiment 2 now runs over `label_noise_values_lowX = [0.0, 1e-2]` in one script execution.
  - For each noise value, the script trains all deep widths and shallow baseline, then prints final decomposition summaries.
- Runtime/cadence changes:
  - `epochs_lowX = 200`
  - `svd_every_epochs_lowX = 5`
  - full metric table rows are printed every 5 epochs.
  - singular values are appended in the same row (`top_sv`) at that same cadence.

#### Run executed
- Command: `python3.11 script.py`
- Experiment 1 remained disabled for efficiency.

#### Key observations from final decomposition (r=500 focus)
- Noise `0.0`:
  - `LowX-Deep(r=500)`: `support_fit_err=1.001e-04`, `model_nullX_norm=2.186e+00`
  - `LowX-Shallow`: `support_fit_err=1.560e-04`, `model_nullX_norm=5.076e+00`
- Noise `0.01`:
  - `LowX-Deep(r=500)`: `support_fit_err=2.408e-03`, `model_nullX_norm=2.183e+00`
  - `LowX-Shallow`: `support_fit_err=2.419e-03`, `model_nullX_norm=5.089e+00`

#### Interpretation
- Adding noise increases support-fit error (learnable-part error) as expected.
- Deep vs shallow nullspace pattern persists under both noise levels: deep keeps significantly smaller `model_nullX_norm` than shallow.

### Retune: why loss regressed and fix to recover ~1e-16+
- Question: why did no-noise loss move from ~`1e-16`/lower to around `1e-4`?
- Main causes:
  - We shortened Experiment 2 runs to `200` epochs for the multi-noise sweep.
  - That was not enough optimization steps for the no-noise case to converge to numerical precision.
  - Additionally, float32 precision can impose a practical floor for extremely small losses.

#### Retuning applied
- Added per-noise schedule:
  - For `noise=0.0`: longer run and stronger convergence settings
    - `epochs=2000`, `deep_lr=2e-2`, `deep_gamma=0.99`, `shallow_lr=1e-2`.
  - For noisy runs (`noise=0.01`): keep shorter sweep schedule (`epochs=200`).
- Increased numerical precision for the script:
  - `torch.set_default_dtype(torch.float64)`

#### Verification (no-noise run)
- `LowX-Deep(r=500)` at epoch 2000: loss `7.155e-31`
- `LowX-Shallow` at epoch 2000: loss `2.372e-20`

#### Conclusion
- The regression to `~1e-4` came from under-training for the no-noise case (and precision limits).
- With retuned schedule + float64, no-noise training loss is now far below `1e-16`.

### Update: add 3-layer deep linear sweep, no-noise-only run
- Added new model class:
  - `DeepLinear3` with factors `[m, r] @ [r, r] @ [r, d]`.
- Experiment 2 now sweeps both deep model families over same widths `r in [k, d, 2d, 10d]`:
  - `LowX-Deep2(r=...)` (2-layer)
  - `LowX-Deep3(r=...)` (3-layer)
- Disabled noisy run for this pass:
  - `label_noise_values_lowX = [0.0]`
- Final summary now also prints spectrum of effective target:
  - `A* (effective) = A* P_x`

#### Validation run (noise=0.0)
- Confirmed script prints both model families and final `A* (effective)` spectrum.
- Example final metrics for `r=500`:
  - `LowX-Deep2(r=500)`: `support_fit_err=1.949e-14`, `model_nullX_norm=2.557e+00`
  - `LowX-Deep3(r=500)`: `support_fit_err=1.026e-11`, `model_nullX_norm=1.008e+01`
  - `LowX-Shallow`: `support_fit_err=2.972e-05`, `model_nullX_norm=5.057e+00`

#### Immediate observation
- In this run, 3-layer deep model (at `r=500`) fits the learnable part very well but carries substantially larger nullspace mass than 2-layer and shallow.

### Column glossary + interpretation (from `outputs/script_20260206_135825.log`)

#### Output file used
- `outputs/script_20260206_135825.log`

#### Table columns (definitions)
- `model`: model/run identifier (e.g., `LowX-Deep2(r=500)`, `LowX-Deep3(r=100)`, `LowX-Shallow`).
- `epoch`: training epoch number. `0` is initialization snapshot before updates.
- `loss`: training MSE on current labels `Y` (for this run, no label noise).
- `lr`: optimizer learning rate at that epoch (after scheduler stepping policy).
- `rel_err`: relative matrix error `||A_hat - A*||_F / ||A*||_F`.
- `fro`: Frobenius norm `||A_hat||_F`.
- `nuc`: nuclear norm `||A_hat||_*` (sum of singular values).
- `spec`: spectral norm `||A_hat||_2` (largest singular value).
- `eff_rank`: entropy-based effective rank of `A_hat` singular values.
- `num_rank`: numeric rank proxy = number of singular values above threshold (`rank_thresh`).
- `err_support`: `||P_left E P_right||_F`, with `E=A_hat-A*`. In Experiment 2 this is the identifiable part `||(A_hat-A*)P_x||_F`.
- `err_outside`: `||E - P_left E P_right||_F`, all error outside that support block.
- `err_null`: two-sided orthogonal block `||P_left_perp E P_right_perp||_F`. In Experiment 2 (`P_left=I`, `P_right=P_x`) this equals `||(A_hat-A*)Q_x||_F`.
- `err_mixed`: residual cross-term block `||E - err_support_block - err_null_block||_F`.
- `supp_frac`: `err_support / ||E||_F`.
- `out_frac`: `err_outside / ||E||_F`.
- `null_frac`: `err_null / ||E||_F`.
- `mixed_frac`: `err_mixed / ||E||_F`.
- `model_nullX_norm`: `||A_hat Q_x||_F`, learned model mass in unidentifiable input-nullspace directions.
- `top_sv`: approximate top singular values of `A_hat` (here via `torch.svd_lowrank`, per config).

#### Important interpretation notes
- In Experiment 2, `A*` is full rank but `X` is low rank, so only `A*P_x` is learnable.
- Because `A*Q_x` is large and unidentifiable, `rel_err`, `err_outside`, and `err_null` can stay large even when train `loss` is tiny.
- Therefore, key metrics for learning behavior are:
  - learnable-fit terms: `err_support` / `support_fit_err`
  - implicit-bias terms: `model_nullX_norm`

#### Trends observed in this run
- Most models (except `Deep3(r=5)`) drive training `loss` extremely low by epoch 2000.
- `Deep2` models:
  - very small `support_fit_err` (down to `1.949e-14` at `r=500`).
  - `model_nullX_norm` around `2.56` to `3.75`, better than shallow (`5.057`).
- `Deep3` models:
  - `r=50` and `r=100` fit support extremely well (`~1e-11` to `1e-12`) with `model_nullX_norm ~2.76-2.89`.
  - `r=5` fails to fit support well (`support_fit_err ~1.167`, `loss ~5.47e-3`), indicating optimization/capacity limitations in this depth-width regime.
  - `r=500` fits support (`~1e-11`) but has very large nullspace mass (`model_nullX_norm ~10.08`) and inflated spectrum (`spec ~9.01`, large `nuc`), suggesting severe parameter growth in unidentifiable directions.
- Shallow model:
  - support fit is good but not as tight as best deep runs (`~2.97e-05`).
  - retains larger nullspace mass than well-behaved deep runs (`~5.06` vs `~2.6-2.9`).

#### Conclusions implied by observations
- In this low-rank-`X` setting, train loss alone is insufficient; matrix-space decomposition is necessary.
- 2-layer deep linear models show a clearer beneficial implicit bias on nullspace suppression vs shallow (lower `model_nullX_norm`) while fitting support very accurately.
- 3-layer behavior is more sensitive to width/optimization:
  - can match support recovery,
  - but may exhibit unstable or amplified nullspace components (notably `r=500`), weakening the desired implicit regularization effect.
- Practical takeaway: deeper factorization does not uniformly improve low-rank implicit bias; monitor `model_nullX_norm`, spectral growth (`spec`, `nuc`, `top_sv`), and support-fit metrics together.

### Deeper interpretation: literature hypotheses vs current observations

#### Hypotheses going in (from deep linear dynamics literature)
- **H1: Implicit low-complexity bias under factorization.**
  - For linear regression solved via factorized parameters (e.g., `W U`), gradient descent from small init tends to favor low-complexity end-to-end maps (often discussed via nuclear-norm / effective-rank bias in the end-to-end matrix).
- **H2: Mode-wise staged learning (“spectral filtering”).**
  - Singular directions with stronger signal tend to be learned earlier/faster; weak directions are delayed.
- **H3: Depth changes optimization timescales.**
  - Greater depth can sharpen implicit bias in some regimes but also slows/complicates optimization and can make dynamics more sensitive to width/learning-rate/symmetry.
- **H4: In underdetermined/partially identifiable settings, implicit bias should select among infinitely many interpolants.**
  - Here, with low-rank `X`, only `A*P_x` is identifiable from data. A bias toward low-complexity solutions is expected to reduce learned mass in `Q_x` (tracked by `model_nullX_norm = ||A_hat Q_x||_F`).

#### How this run compares to those hypotheses
- **H4 supported for 2-layer deep vs shallow:**
  - `Deep2` consistently achieved very small support-fit error and lower `model_nullX_norm` than shallow (`~2.56-3.75` vs `~5.06`).
  - This matches the expected beneficial selection among interpolating solutions.
- **H3 strongly visible for 3-layer:**
  - `Deep3(r=50,100)` behaved well (small support-fit error and low-ish nullspace mass).
  - `Deep3(r=5)` underfit (large support error), indicating optimization/capacity limits at this depth-width.
  - `Deep3(r=500)` fit support but produced large nullspace mass (`~10.08`) and inflated top singular values (`spec ~9.01`), indicating depth-related optimization/symmetry issues can dominate and break the hoped-for nullspace suppression.
- **H1/H2 partially supported, but conditional on stable optimization regime:**
  - We do see low-rank/structured learning in many deep runs, but not uniformly across depth+width.
  - In particular, depth-3 at very large width appears to exploit scale freedoms and land on a high-nullspace-norm interpolant despite tiny training loss.

#### Revised conclusions (stronger version)
- The experiments support a nuanced claim, not a blanket one:
  - **2-layer deep linear** in this setting shows the expected implicit bias (better support recovery + smaller nullspace component than shallow).
  - **3-layer deep linear** is more fragile: depending on width and optimization schedule, it can either match the bias or drift to high-norm nullspace solutions.
- Therefore, the primary scientific takeaway is:
  - **Implicit bias is architecture- and optimization-regime-dependent.**
  - Increasing depth does not monotonically improve nullspace suppression in partially identifiable problems.
- Practical implication for future experiments:
  - When comparing implicit bias across depths, control for optimization artifacts (step size, decay, possible regularization, and per-layer balancing diagnostics), otherwise “depth effect” may be confounded by unstable factor scaling.

### Update: initialization-magnitude sweep case
- Added a new Experiment 2 case to study how initialization scale affects nullspace/spectrum dynamics.
- New config:
  - `init_scale_values_lowX = [1e-3, 1e-2, 5e-2]`
- For each noise setting (currently only `0.0`) and each init scale, the script now runs:
  - `LowX-Deep2(r)` sweep over `r in [k, d, 2d, 10d]`
  - `LowX-Deep3(r)` sweep over same widths
  - `LowX-Shallow` with matching init std
- Model constructors now use `init_scale` for all factor matrices (and shallow weight std).
- Output labels include init scale (e.g., `s=1e-03`) so runs are separable in logs.

Purpose:
- Compare regimes where nullspace-associated singular values and `model_nullX_norm` grow/plateau/decay under different initial magnitudes.

### Experiment outcome: initialization-scale sweep (no-noise)
- Run file: `outputs/script_20260207_181928.log`
- Settings:
  - `init_scale_values_lowX = [1e-3, 1e-2, 5e-2]`
  - `label_noise_std_lowX = 0.0`
  - long no-noise schedule (`epochs=2000`).

#### Key summary trends
- **2-layer deep (`Deep2`) remains stable across init scales**:
  - Support fit is excellent for medium/large widths (down to `~1e-14` to `1e-13`).
  - `model_nullX_norm` stays in a relatively narrow band (`~2.48` to `~3.98` depending on width/scale).
- **3-layer deep (`Deep3`) is more sensitive to init scale and width**:
  - `r=5` underfits for small/medium scale (`support_fit_err ~2.6`), improves at `5e-2` (`~4e-5`) but still not as strong as deeper widths.
  - `r=500` consistently develops large nullspace mass:
    - `s=1e-3`: `model_nullX_norm ~1.008e+01`
    - `s=1e-2`: `~8.213e+00`
    - `s=5e-2`: `~1.091e+01`
- **Shallow baseline**:
  - `model_nullX_norm` stays around `~5.04` to `~5.46`.

#### Nullspace singular-value behavior (growth vs decay)
- For `Deep3(r=500)`, top/tail singular values **grow rapidly early** and then **plateau** at elevated levels (no meaningful decay by epoch 2000).
  - Example (`s=1e-3`): `spec` grows from `6.7e-06` (epoch 0) to `~9.49` (epoch 100) and stays near `~9.49` at epoch 2000.
  - Example (`s=1e-2`): `spec` grows from `~7.1e-03` to `~6.52` and plateaus.
  - Example (`s=5e-2`): `spec` starts high (`~0.915`), spikes (`~9.99` at epoch 1), then settles near `~6.78`.
- For `Deep2(r=500)`, singular values also grow early but settle into a lower-magnitude regime (`spec ~3.9–4.1`) with smaller nullspace norm.

#### Interpretation
- Initialization scale changes the regime, but in this setup it does **not** induce nullspace decay for the problematic `Deep3(r=500)` case; it mainly changes the plateau level after rapid growth.
- The “low-rank implicit bias” appears robust for `Deep2`, while `Deep3` at large width is prone to high-norm nullspace solutions across scales.
- The most promising region for controlled nullspace behavior here is moderate width/depth combinations (e.g., `Deep2` and `Deep3` with `r=50/100`) rather than very wide `Deep3`.

### Update: added biased deep-linear variants (Deep2OB, Deep2IOB)

Implemented and ran two additional model families in Experiment 2:
- `LowX-Deep2OB`: outer bias only, `y = W(Ux) + w`
- `LowX-Deep2IOB`: inner+outer bias, `y = W(Ux + u) + w`

Run artifact:
- `outputs/script_20260207_191422.log`

Status:
- Completed summary/decomposition blocks for `init_scale=1e-3` and `init_scale=1e-2`.
- `init_scale=5e-2` run was partially recorded before interruption (log ends mid-block), so conclusions below are based on the two completed scales plus consistent partial trends at `5e-2`.

#### Main quantitative comparison (completed scales)
`init_scale=1e-3` (final decomposition):
- `Deep2(r=500)`: `support_fit_err=3.756e-14`, `model_nullX_norm=2.706`
- `Deep2OB(r=500)`: `support_fit_err=5.252e-14`, `model_nullX_norm=2.748`
- `Deep2IOB(r=500)`: `support_fit_err=2.897e-14`, `model_nullX_norm=2.815`
- `Shallow`: `support_fit_err=4.158e-04`, `model_nullX_norm=5.036`

`init_scale=1e-2` (final decomposition):
- `Deep2(r=500)`: `support_fit_err=2.020e-14`, `model_nullX_norm=2.613`
- `Deep2OB(r=500)`: `support_fit_err=1.916e-14`, `model_nullX_norm=2.541`
- `Deep2IOB(r=500)`: `support_fit_err=1.899e-14`, `model_nullX_norm=2.487`
- `Shallow`: `support_fit_err=5.927e-08`, `model_nullX_norm=5.056`

At moderate widths (`r=50,100`) the same pattern holds: all three deep-2 families fit support essentially perfectly (`~1e-12` to `1e-13`) while keeping `model_nullX_norm` around `~2.7-2.9`, much lower than shallow (`~5.0`).

#### Interpretation
- Adding output bias and input+output bias **does not destroy** the favorable implicit bias seen in 2-layer deep linear models in this setup.
- Across completed runs, `Deep2OB` and `Deep2IOB` are at least competitive with unbiased `Deep2` on learnable support fit, and often slightly better on `model_nullX_norm` at larger width.
- Relative to shallow linear, all deep-2 variants still produce much smaller learned nullspace mass (`||A_hat Q_x||_F`), consistent with the original low-complexity/implicit-regularization hypothesis in this partially identifiable regime.
- Bias terms therefore look like a quantitative perturbation of the same qualitative behavior, not a qualitative regime change.

#### Caveat
- The `init_scale=5e-2` block in this run is incomplete. Partial rows are directionally consistent with the above (deep-2 variants remain close to each other and below shallow on nullspace norm), but this should be rerun to obtain full final summary lines for that scale.

### Update: singular-value evolution plotting added
- Added plotting support to `script.py` to visualize singular-value trajectories over training.
- `train_model(...)` now stores singular-value history at each metric checkpoint (`sv_history_epochs`, `sv_history`).
- Added a multi-subplot plotting function that places one model per subplot and draws the top singular values over epoch on a log-y scale.
- Added timestamped artifact organization per run:
  - `outputs/run_<timestamp>/plots/*.png`
  - `outputs/run_<timestamp>/data/*_singular_value_history.pt`
- For Experiment 2, one figure is saved per `(noise, init_scale)` block; each figure contains all model variants in that block (`Deep2`, `Deep3`, `Deep2OB`, `Deep2IOB`, `Shallow`).
- The script prints `artifact_dir` at startup and prints each saved plot/data path after writing.

### Minor output-layout update
- Flattened run artifacts to a single directory per run (no `data/` and `plots/` subfolders).
- New layout:
  - `outputs/run_<timestamp>/*.png`
  - `outputs/run_<timestamp>/*_singular_value_history.pt`
  - `outputs/run_<timestamp>/script_<timestamp>.log`
- Added internal tee logging in `script.py` so each run automatically writes its full console output to the run directory log file while still printing to terminal.

## Consolidated Findings Across Outputs (Deeper Synthesis)

This section consolidates evidence from:
- `outputs/script_20260207_181928.log` (complete init-scale sweep for Experiment 2 without bias variants)
- `outputs/script_20260207_191422.log` (adds `Deep2OB` / `Deep2IOB`, complete for `init_scale={1e-3,1e-2}`)
- `outputs/script_20260206_135825.log` (older baseline run; same qualitative pattern for Experiment 2)
- `outputs/run_20260208_012845/*` (smoke verification of new plotting artifacts)

### Literature-Grounded Hypotheses (pre-registered style)

H1. **Implicit low-complexity bias from factorization**  
Deep linear parameterization (product of matrices) with small init should prefer lower-complexity interpolants than shallow linear, especially in underdetermined settings.

H2. **Support-first learning in identifiable subspaces**  
With low-rank `X`, learnable component is `A*P_x`; good training dynamics should drive `support_fit_err = ||(A_hat - A*)P_x||_F` near zero.

H3. **Nullspace suppression under implicit bias**  
Among exact (or near-exact) fits on data, deep models should place less energy in unidentifiable directions: smaller `model_nullX_norm = ||A_hat Q_x||_F`.

H4. **Depth-sensitive optimization and scaling pathologies**  
Deeper factorizations can be more sensitive to width/schedule and may amplify scale in unidentifiable directions despite tiny train loss.

H5. **Bias terms may perturb but not necessarily destroy implicit bias**  
Adding affine terms (`w`, `u`) could weaken or preserve the same qualitative preference.

### What the data shows

#### 1) Learnable support is fit extremely well in most regimes
- In complete runs, many deep models reach `support_fit_err` around `1e-12` to `1e-14` (e.g., `Deep2(r=500)` often `~2e-14` to `8e-14`).
- Shallow also fits support very well, but generally not as tightly (`~1e-6` to `1e-4` depending on init; sometimes better in specific runs with aggressive optimization).
- Exception: narrow `Deep3(r=5)` frequently underfits or converges very slowly (e.g., `support_fit_err ~2.6` for `init_scale=1e-3,1e-2`; improves to `~4e-5` at `5e-2` in the complete sweep).

Interpretation: H2 is strongly supported in broad regimes, but narrow 3-layer models can be optimization-limited.

#### 2) Two-layer deep models usually suppress nullspace better than shallow
- Representative values from complete sweep (`script_20260207_181928.log`):
  - `init=1e-3`: `Deep2(r=500) model_nullX_norm=2.826` vs `Shallow=5.040`
  - `init=1e-2`: `Deep2(r=500)=2.477` vs `Shallow=5.060`
  - `init=5e-2`: `Deep2(r=500)=3.720` vs `Shallow=5.458`
- Similar trend holds at `r=50,100` with `Deep2` usually around `~2.8-3.0`, below shallow `~5+`.

Interpretation: H1/H3 supported for 2-layer factorization in this setup.

#### 3) Three-layer very-wide models show strong nullspace inflation
- `Deep3(r=500)` repeatedly yields large `model_nullX_norm`:
  - `init=1e-3`: `~1.008e+01`
  - `init=1e-2`: `~8.213e+00`
  - `init=5e-2`: `~1.091e+01`
- Yet `support_fit_err` is still tiny (`~1e-10` to `1e-11`), meaning these are near-interpolating but high-nullspace-mass solutions.
- Spectra corroborate this: much larger tail/overall scale for `Deep3(r=500)` than `Deep2`.

Interpretation: H4 strongly supported; depth can hurt implicit selection under current optimizer/schedule.

#### 4) Bias variants (`Deep2OB`, `Deep2IOB`) preserve the good 2-layer behavior
- From `script_20260207_191422.log` (`init=1e-3,1e-2` complete):
  - `Deep2OB/Deep2IOB` achieve support errors comparable to unbiased `Deep2` (`~1e-14` to `1e-13` at larger widths).
  - `model_nullX_norm` remains around the same favorable band (`~2.5-2.9` at larger widths), still much lower than shallow (`~5.0`).

Interpretation: H5 supported. In this setting, adding inner/outer bias changes quantitative details but not the main qualitative implicit-bias outcome.

#### 5) Why `rel_err`, `err_outside`, `err_null` stay large even with tiny train loss
- In Experiment 2, `A*` has a large unlearnable component `A*Q_x` (`||A*Q_x||_F ~ 2.169e+01`).
- So matrix-space errors that include the unidentifiable block cannot vanish from data alone.
- This matches persistent `rel_err ~ 0.95-1.05` despite near-zero training loss and near-zero support-fit error.

Interpretation: observed metrics are internally consistent; no implementation bug is indicated by this pattern.

### Revised global conclusions

1. **Core claim supported (with caveats):**  
   Deep linear factorization (especially 2-layer) exhibits a favorable implicit bias in partially identifiable linear regression: it tends to fit the learnable component while using less nullspace mass than shallow linear.

2. **Depth is not monotonic improvement:**  
   3-layer models can be good at moderate widths but can become pathological at very large width (high nullspace norm, inflated spectra) under current optimization.

3. **Bias terms are not the main failure mode:**  
   Outer/inner bias additions to 2-layer models did not remove the low-nullspace behavior.

4. **Optimization regime matters as much as architecture:**  
   LR schedule, initialization scale, depth, and width jointly determine whether implicit regularization manifests cleanly.

### Falsifiable next-step hypotheses (for follow-up runs)

- NH1: Per-layer balancing or explicit norm control (e.g., mild weight decay or balanced initialization constraints) will reduce `Deep3(r=500)` nullspace inflation while preserving support fit.
- NH2: Extending training time alone (without additional regularization) will **not** materially reduce `model_nullX_norm` for pathological `Deep3(r=500)` runs; it plateaus.
- NH3: In noisy-label settings, 2-layer deep models should retain a stronger nullspace suppression gap versus shallow than 3-layer very-wide models.
- NH4: Tracking per-factor Frobenius norms and their ratios will predict when `Deep3` enters high-nullspace interpolants before final convergence.

### Plotting/artifact integrity

- Smoke run (`outputs/run_20260208_012845/`) confirms singular-value history and multipanel spectrum plots are being written correctly.
- This enables direct visual validation of mode-wise growth/plateau behavior in future full runs.

## New Experiment Design + Executed Ablation (2026-02-08)

### Proposed next experiments (designed)

1. **Balancedness ablation (executed below)**  
Keep data/task fixed (Experiment 2 setting), vary only per-factor initialization scales while keeping approximate product scale fixed.  
Goal: test whether factor imbalance alone changes nullspace selection.

2. **Optimizer geometry ablation**  
Repeat the same balanced/unbalanced setup with SGD vs Adam (same LR schedule normalized for effective step size).  
Goal: test whether nullspace selection is optimizer-dependent.

3. **Deep3 stabilization ablation**  
For 3-layer models, add mild per-layer weight decay and compare against no decay.  
Goal: test if Deep3 nullspace inflation can be reduced without hurting support fit.

### Executed: Experiment 3 (balanced vs unbalanced init)

- Run ID: `outputs/run_20260208_115050/`
- Script invocation:
  - `RUN_EXPERIMENT_2=0 RUN_EXPERIMENT_3=1 python3.11 script.py`
- Setting:
  - Same low-rank-input identifiability setup as Experiment 2 (`A*` full-rank, `X` low-rank).
  - Deep model: 2-layer, `r=100`, `epochs=600`, Adam, `lr=1e-2`, decay `0.995`.
  - Conditions:
    - `Deep2-Balanced`: `(std_W, std_U) = (1e-2, 1e-2)`
    - `Deep2-WideOuter`: `(1e-1, 1e-3)`
    - `Deep2-WideInner`: `(1e-3, 1e-1)`
  - Reference baseline: `Shallow-Ref`.

### Final quantitative outcomes

- `Deep2-Balanced`
  - `support_fit_err=4.761e-14`
  - `learnable_target_err=4.766e-14`
  - `model_nullX_norm=2.683e+00`
- `Deep2-WideOuter`
  - `support_fit_err=4.938e-14`
  - `learnable_target_err=4.949e-14`
  - `model_nullX_norm=3.012e+00`
- `Deep2-WideInner`
  - `support_fit_err=5.085e-14`
  - `learnable_target_err=5.094e-14`
  - `model_nullX_norm=3.305e+00`
- `Shallow-Ref`
  - `support_fit_err=2.258e-12`
  - `learnable_target_err=2.258e-12`
  - `model_nullX_norm=5.048e+00`

Artifacts:
- Log: `outputs/run_20260208_115050/script_20260208_115050.log`
- Singular-value history: `outputs/run_20260208_115050/exp3_balanced_vs_unbalanced_singular_value_history.pt`
- Plot: `outputs/run_20260208_115050/exp3_balanced_vs_unbalanced_singular_value_evolution.png`

### Observations and interpretation

1. All deep conditions solved the learnable component to numerical precision.  
Support-fit errors are all `~5e-14`, so optimization success on the identifiable subspace is not the discriminant.

2. Initialization balancedness changes nullspace bias even when training loss is effectively zero.  
`model_nullX_norm` is strictly ordered:
`Balanced (2.683) < WideOuter (3.012) < WideInner (3.305) << Shallow (5.048)`.

3. This supports a stronger claim: implicit bias in deep linear models is not only architecture-level, but parameterization-trajectory-level.  
Different factorizations that can all interpolate the data pick measurably different nullspace solutions.

4. The result is consistent with deep linear dynamics literature emphasizing path geometry and scale coupling, not just representational capacity.  
The low-rank/low-complexity tendency appears strongest with more balanced factors.

5. Caution: this is one width (`r=100`) and one optimizer (Adam).  
The direction is clear, but generalization requires the designed follow-ups above.

### Repeat run (reproducibility check)

- Repeat run ID: `outputs/run_20260208_125833/`
- Command repeated exactly:
  - `RUN_EXPERIMENT_2=0 RUN_EXPERIMENT_3=1 python3.11 script.py`
- Final decomposition matched the prior run exactly (to printed precision):
  - `Balanced: model_nullX_norm=2.683e+00`
  - `WideOuter: model_nullX_norm=3.012e+00`
  - `WideInner: model_nullX_norm=3.305e+00`
  - `Shallow-Ref: model_nullX_norm=5.048e+00`

Interpretation: with fixed random seeds and deterministic setup, this ablation is reproducible; the balancedness ordering appears robust in this controlled setting.

## Experiment Design Cycle 2 (2026-02-08): Deep3 regularization frontier

### Why this cycle

A key unresolved question was whether the previously observed deep3 nullspace inflation is:
1. an unavoidable property of deep3 factorization, or
2. a controllable optimization/regularization artifact.

### New hypotheses (cycle 2)

- CH1: Mild explicit regularization (weight decay) can reduce `Deep3` nullspace mass `||A_hat Q_x||_F`.
- CH2: There is a Pareto frontier: stronger null suppression may come at cost of support fit.
- CH3: Optimizer choice can dominate this tradeoff (Adam vs SGD can land in very different regions).

### Executed experiment

- Run ID: `outputs/run_20260208_153813/`
- Command:
  - `RUN_EXPERIMENT_2=0 RUN_EXPERIMENT_3=0 RUN_EXPERIMENT_4=1 python3.11 script.py`
- Setting:
  - low-rank-input identifiability setup (`A*` full-rank, `X` low-rank).
  - model width fixed at `Deep3(r=500)` (problematic regime from earlier logs).
  - compared:
    - `Deep3-Adam-noWD`
    - `Deep3-Adam-WD1e-4`
    - `Deep3-SGD-WD1e-4`
  - plus `Shallow-Ref-Exp4`.

### Final outcomes

- `Deep3-Adam-noWD`
  - `support_fit_err=1.582e-14`
  - `model_nullX_norm=2.859e+00`
- `Deep3-Adam-WD1e-4`
  - `support_fit_err=1.992e-02`
  - `model_nullX_norm=9.933e-01`
- `Deep3-SGD-WD1e-4`
  - `support_fit_err=6.982e+00`
  - `model_nullX_norm=2.451e-02`
- `Shallow-Ref-Exp4`
  - `support_fit_err=2.272e-13`
  - `model_nullX_norm=5.067e+00`

Artifacts:
- log: `outputs/run_20260208_153813/script_20260208_153813.log`
- history: `outputs/run_20260208_153813/exp4_deep3_optimizer_weight_decay_singular_value_history.pt`
- plot: `outputs/run_20260208_153813/exp4_deep3_optimizer_weight_decay_singular_value_evolution.png`

### Interpretation

1. **CH1 supported**: adding weight decay to deep3 can substantially reduce nullspace energy (`2.859 -> 0.993` under Adam).
2. **CH2 strongly supported**: this reduction came with support underfitting (`~2e-2` instead of `~1e-14`).
3. **CH3 supported**: optimizer/regularizer interaction is dominant.
   - SGD + WD pushes nullspace near zero but catastrophically underfits support.
   - Adam + no WD fits support perfectly but keeps moderate nullspace mass.
4. The results suggest a **regularization frontier**, not a single winner:
   - low support error / moderate nullspace (`Adam-noWD`)
   - moderate support error / low nullspace (`Adam-WD1e-4`)
   - poor support error / extremely low nullspace (`SGD-WD1e-4`)

### Design implications for next cycle

To locate a practically useful point on the frontier, sweep weaker regularization and late-phase scheduling instead of fixed WD:

1. Adam + WD grid: `wd in {1e-6, 3e-6, 1e-5, 3e-5, 1e-4}`.
2. Two-stage schedule: train no-WD to near-zero support error, then enable tiny WD for a short fine-tuning phase.
3. Optional projector penalty test: add `lambda * ||A Q_x||_F^2` directly to objective for controlled nullspace suppression while tracking support-fit degradation.
