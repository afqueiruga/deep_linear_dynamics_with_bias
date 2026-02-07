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

## 2026-02-06 (interpretation note)

- Interpreted result from `outputs/theory_sympy_low_rank_proof_20260206_142353.md`.
- End-to-end dynamics:
  - `dA/dt = -(A-A_*)(U^T U) - (W W^T)(A-A_*)`.
  - This shows error is preconditioned by factor Gram matrices, not by identity as in shallow linear regression.
  - Consequence: optimization speed is anisotropic and tied to the active subspaces of `W` and `U`.
- Single-mode dynamics:
  - In aligned coordinates, `ds/dt = -(w^2+u^2)(s-\sigma)`.
  - In the balanced regime (`w=u`, `s=w^2`), this becomes `ds/dt = 2 s (\sigma - s)`.
  - Interpretation: this is logistic growth for each singular mode.
- Low-rank implication:
  - If `s` is near zero, the multiplicative `s` factor makes growth slow (small modes stay small).
  - Modes that become nontrivial accelerate, while large modes saturate at `sigma`.
  - This creates selective mode activation, i.e. effective low-rank behavior during training.
- Scope/assumptions:
  - The logistic statement is exact in the aligned, balanced reduction; outside it, mode coupling can appear.
  - The symbolic derivation still establishes the core mechanism used in low-rank implicit-bias arguments.

## 2026-02-06 (target-vs-off-target clarification)

- Clarification requested: does the derived law only imply slow growth, or also decay of non-target modes?
- In the balanced aligned reduction, the mode ODE is `ds/dt = 2 s (sigma - s)`.
- For target-aligned modes (`sigma > 0`): small `s` grows slowly at first, then accelerates, then saturates.
- For off-target modes (`sigma = 0`): `ds/dt = -2 s^2 <= 0`, so these modes decay toward zero.
- Interpretation: the mechanism is dual-purpose in this regime: it promotes target modes and suppresses non-target directions.
