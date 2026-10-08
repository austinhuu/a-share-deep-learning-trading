# Public Release Notes

## Refactoring from the supplied internal Stage A archive

**Preserved:** technical indicator features; PyTorch LSTM / Transformer-LSTM / CNN1D / MLP model classes; training loop with early stopping; Optuna search implementation; gradient, permutation, SHAP and feature-stability research helpers; rolling forecast and single-asset research risk backtest; optional VeighNa CSV signal bridge.

**Changed for public presentation and safer evaluation:**

- Deleted bundled historical market price datasets, duplicate `artifacts.zip`, training checkpoints, binary scalers, old research result tables, runtime caches and compiled Python files.
- Removed the hard-coded secret from `data/loader.py`; Tushare is imported only when the provider is requested, and the token is read from the environment.
- Fixed the incorrect treatment of unknown terminal labels (`NaN` rather than class 0).
- Fixed sample date alignment: 30-bar sequence now ends at the prediction timestamp rather than the previous bar.
- Added a one-bar label purge at chronological validation and test boundaries.
- Refitted standardization exclusively using past inner-training data per walk-forward fold and per development CV fold.
- Reworked the top-level CLI: `--demo`, `--data-csv`, real Tushare daily ingestion, `--mode walk_forward|full|ensemble`, optional Optuna/explainability, output path and repeatable seed.
- `full` mode is clearly **development-only CV** and no longer generates a misleading full-sample performance figure.
- The public `ensemble` mode uses **equal weights** on each model's independent walk-forward predictions. The older AUC-weighted research helper is retained but not used by the current runner.
- Replaced the VeighNa adapter's local Windows path and potentially active stop-order branch with a safe historical CSV monitor with **no order submission**.
- Rebuilt financial metric helpers and accounted for the starting equity of 1.0 in maximum drawdown calculations.
- Added offline synthetic data generator, plot script, tests, bilingual docs, `.gitignore` and `.env.example`.

**Important compatibility note:** models, scalers and archived backtest metrics created by the old project are **not** interchangeable with this version. Rerun experiments on legitimately accessible data before comparing or quoting results. This repo is a refactor of a research prototype, not a certified or audited production trading system.
