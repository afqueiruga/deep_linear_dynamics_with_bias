# Agent Memory

- Execution environment: `python3.11`
- `Background.md` contains background notes about training dynamics from ChatGPT.
- Research goal: understand training dynamics of deep linear models, with focus on effective low-rank structure, and compare against models that do not exhibit that behavior.
- Error decomposition idea: for `E = A_hat - A*`, project onto `A*` support and nullspace via `A*` SVD projectors. Hypothesis is low-rank inductive bias drives both components down, while models without that bias retain larger nullspace error.
- `LOG.md` is the running notebook for detailed code changes and experiment results.
