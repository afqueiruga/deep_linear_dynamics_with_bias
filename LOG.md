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
