# THEORY LOG

Running notebook for symbolic/theoretical results (separate from numerical experiment logs in `LOG.md`).

## 2026-02-06

- Started a theoretical track for 2-layer deep linear models.
- Added SymPy proof script: `theory/sympy_low_rank_proof.py`.
- Symbolically verified:
  - Gradient flow for factors `W, U` under whitened squared loss.
  - Induced end-to-end ODE for `A = WU`.
  - Single-mode balanced logistic law `ds/dt = 2 s (sigma - s)` explaining low-rank mode selection.
- Saved execution log: `outputs/sympy_low_rank_proof_20260206_141438.log`.
