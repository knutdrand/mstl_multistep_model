# Experiment log — irs_allocated feature extraction

**Frozen harness** (identical for every row; see `experiments/run_eval.sh`):
- Model: `/Users/knutdr/Sources/mstl_multistep_model`
- Dataset: `/Users/knutdr/Data/CH/chap_data_level5_irs_allocated_monthly.csv` (406 locations, monthly 2013-01..2026-01)
- Backtest: `--n-periods 3 --n-splits 12 --stride 1`, `--run-config.track`
- chap-core: 2.0.0.dev1
- Primary metric: **log_crps** (lower is better). Secondaries: crps, mae, coverage.

Goal: extract features from the raw `irs_allocated` allocation column (0–1 coverage
fraction, sparse ~2.5% of rows, ~8-month allocation runs in 188/406 locations) in the
**model code**, and beat the climate-only baseline.

| # | branch / commit | lever | hypothesis / change | log_crps | crps | mae | coverage | verdict | mlflow | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | improve/spray-setup @2c6eebe | — | baseline: rf_residual + deseasonalized climate, **no IRS** | **0.3253** | 83.6 | 113.3 | 0.757 / 0.477 | baseline | — | config_irs_baseline.yaml; ~14 min/eval. Matches old leaderboard 0.3258 that had binary sprayed_* flags → flags added ~nothing |
