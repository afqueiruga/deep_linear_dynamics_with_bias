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
