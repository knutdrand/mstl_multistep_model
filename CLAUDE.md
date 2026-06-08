# mstl_multistep_model

CHAP-compatible disease-case forecaster: MSTL decomposition → a shared fixed-order
ARIMA on the deseasonalised series (mean + spread) → a pooled RandomForest that
corrects the ARIMA residual using lagged climate anomalies, engineered IRS features
and lagged self-history → a location-scale variance head. See `docs/champion_model.pdf`.

The published default is `config.yaml`; a config usually only needs to name its
covariate / IRS columns and (optionally) tune lags. The model lives in
`mstl_multistep/`; the calibration utility is `mstl_multistep/calibration.py`
(+ `scripts/calibrate_forecast.py`). Past experiments are archived under
`experiments/` (configs/scripts under `experiments/archive/`).

## Data

Chap-compatible CSVs live at `/Users/knutdr/Data/CH/`:

- `chap_data_level5_irs_allocated_monthly.csv` (the main 406-sector dataset)
- `chap_LAO_admin1_monthly-3.csv`, `chap_VNM_admin1_monthly.csv`

## Constraints / guidance

- Data is small, noisy, highly seasonal.
- Use **log-CRPS** as the main eval metric (secondary: CRPS).
- Keep the champion bit-identical: `tests/test_golden.py` guards it — run it after any change.
- Always pass `--run-config.track` when running `chap eval` (MLflow setup is in the global
  `~/.claude/CLAUDE.md`). For monthly data use `--backtest-params.n-splits 12`,
  `--n-periods 3`, `--stride 1`.
