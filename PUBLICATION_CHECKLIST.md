# Before Making This Repository Public

- [ ] Obtain any required authorization for internship-created code, model designs, report material, and employer-associated IP. A public repository is not proof of ownership.
- [ ] Rotate/revoke the **real** Tushare token that existed in the submitted legacy project and its `__pycache__` files. The public ZIP deliberately omits it; it must not remain active.
- [ ] Inspect **every** `git add` result before the first commit. Never commit data, `.env`, model weights, logs, or private reports. Do not commit an older ZIP even as a convenience.
- [ ] Confirm CSV dataset reuse and sharing rights; no historical market-data CSV is included in this release.
- [ ] Run `python -m pytest -q` on the actual deployment platform. Verify `--demo` works before running licensed/real data.
- [ ] If advertising any Sharpe or returns externally, **re-run** with the repaired code and a clearly defined real-data out-of-sample period, execution assumptions and transaction costs.
- [ ] Review README and repository description for truthful positioning: **daily-bar research prototype**, VeighNa listener only, no verified live fills.
- [ ] Decide whether a software reuse license can legally be granted before adding MIT/Apache/etc. No license is included pending that decision.

## Suggested GitHub setup

- Name: `a-share-deep-learning-trading`
- Description: `Deep learning and walk-forward evaluation for A-share quantitative research (PyTorch).`
- Repository visibility: **Private** until all authorizations and secret rotation are confirmed; then optionally Public.
- GitHub Topics: `quantitative-finance`, `pytorch`, `lstm`, `transformer`, `time-series`, `backtesting`, `a-share`.

## First push on Windows PowerShell

```powershell
cd path\to\a-share-deep-learning-trading
git init
git branch -M main
git add .
git status --short     # inspect BEFORE first commit
git commit -m "Public research release: secure config and purged walk-forward pipeline"
git remote add origin https://github.com/YOUR_USERNAME/a-share-deep-learning-trading.git
git push -u origin main
```

Create the **empty** GitHub repository first. Replace `YOUR_USERNAME` with your own username. `artifacts/` is ignored by Git, so it will remain on your machine without accidental upload.
