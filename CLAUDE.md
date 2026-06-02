# mstl_multistep_model

CHAP-compatible model: MSTL decomposition + the recursive multistep model
from `../simple_multistep_model` as the trend/remainder forecaster.

## Data

Chap-compatible CSVs live at `/Users/knutdr/Data/CH/`:

- `chap_LAO_admin1_monthly-3.csv`
- `chap_VNM_admin1_monthly.csv`
- `chap_data_level5_with_spray.csv`

## Constraints / guidance

- Prefer self-forecasting (no future covariates) — the default config uses
  no covariates and lets MSTL estimate seasonality.
- Data is small, noisy, highly seasonal.
- Use **log-CRPS** as the main eval metric.
- Always pass `--run-config.track` when running `chap eval` (MLflow setup is
  in the global `~/.claude/CLAUDE.md`). For monthly data use
  `--backtest-params.n-splits 12`, `--n-periods 3`, `--stride 1`.

## Dependency note

`simple_multistep_model` is a uv path dependency (`../simple_multistep_model`).
Keep the two repos as siblings.
