# QA evidence (GitHub public-release refactor)

Executed locally with the current environment; **not** real-data trading verification:

- `python -m pytest -q --disable-warnings` — **10 passed** (unit checks and one end-to-end synthetic MLP walk-forward test).
- `run_stage_a.py --demo --demo-bars 220 --mode full --model lstm --epochs 1 --patience 1 --optuna-trials 1 --explain` — completed; development-only CV and feature-explanation outputs created.
- `run_stage_a.py --demo --demo-bars 220 --mode ensemble --epochs 1 --patience 1 --test-bars 80 --window 20` — completed; four distinct models trained for a one-epoch smoke test and ensemble output generated.
- `scripts/plot_results.py` — generated a local comparison plot from the generated synthetic experiment outputs.

**Not tested:** authenticated Tushare API calls, market-data revision behavior, Windows-specific platform compatibility, actual VeighNa runtime, authorized real-data reruns, broker interactions, market execution or strategy profitability. The optional SHAP branch was not part of the automated smoke verification. The workflow file independently reruns the ten tests after each GitHub push.
