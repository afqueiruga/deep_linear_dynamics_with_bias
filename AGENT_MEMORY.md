# Agent Memory

- Execution environment: `python3.11`
- `Background.md` contains background notes about training dynamics from ChatGPT.
- Research goal: understand training dynamics of deep linear models, with focus on effective low-rank structure, and compare against models that do not exhibit that behavior.
- Error decomposition idea: for `E = A_hat - A*`, project onto `A*` support and nullspace via `A*` SVD projectors. Hypothesis is low-rank inductive bias drives both components down, while models without that bias retain larger nullspace error.
- `LOG.md` is the running notebook for detailed code changes and experiment results.
- `THEORY_LOG.md` is the running notebook for theoretical/symbolic derivations and proof progress.
- A new theory line started on 2026-02-06 with SymPy (`theory/sympy_low_rank_proof.py`) for 2-layer deep linear low-rank dynamics.
- Whenever running `script.py`, save full stdout to timestamped files under `outputs/`, e.g. `outputs/script_YYYYMMDD_HHMMSS.log`, so runs can be revisited.
- Whenever running theory scripts (e.g. files under `theory/`), save timestamped Markdown outputs under `outputs/`, e.g. `outputs/theory_<script_name>_YYYYMMDD_HHMMSS.md`.
