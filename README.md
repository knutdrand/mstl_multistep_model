# mstl_multistep_model

A CHAP-compatible external model that **MSTL-decomposes** each location's
series and fits the recursive multistep model from
[`simple_multistep_model`](../simple_multistep_model) as the **trend (+
remainder) forecaster**. Seasonality is held fixed and extrapolated
seasonal-naive, then added back before the forecast is returned.

```
log1p(y) ──MSTL──▶ trend + remainder ──▶ MultistepModel (recursive RF) ──┐
              │                                                          ├─▶ expm1 ─▶ samples
              └──▶ seasonal ──seasonal-naive tile──────────────────────-─┘
```

The trend model is reused *verbatim* — same RandomForest regressor, same
probabilistic wrapper (`bootstrap` / `cross-conformal` / `bucketedresidual`),
same lag handling — it just forecasts the deseasonalized series instead of
the raw target. Because MSTL self-estimates seasonality, the model needs **no
future covariates** by default.

## Why this shape

The data is small, noisy and highly seasonal (see `CLAUDE.md`). MSTL pins the
strong, stable seasonal cycle so the learned model only has to predict the
comparatively smooth trend + remainder, which a recursive autoregressor with a
few target lags handles well even on short series.

## Installation

This project uses [uv](https://docs.astral.sh/uv/) and depends on its sibling
`simple_multistep_model` via a path dependency. Clone both side by side:

```
Sources/
  simple_multistep_model/
  mstl_multistep_model/      <- this repo
```

```bash
uv sync
```

## Configuration

Single wrapped YAML in the shape CHAP calls a *model configuration*:

```yaml
additional_continuous_covariates: []       # [] = pure self-forecasting trend
user_option_values:
  target_variable: disease_cases
  log_transform: true                       # decompose log1p(y), expm1 forecasts
  season_length_monthly: 12
  season_length_weekly: 52
  n_target_lags: 6                          # deseasonalized-target lags
  n_samples: 100
  feature_min_lag: 1                        # only used if covariates are listed
  feature_max_lag: 3
  prob_wrapper: bootstrap                   # bootstrap | cross-conformal | bucketedresidual
  min_bucket_size: 5                        # only for bucketedresidual
  use_location_dummies: true                # keep per-series identity in the pooled model
  discretize_samples: false
  random_seed: 42
  rf:
    n_estimators: 200
    max_depth: 10
    min_samples_leaf: 5
    max_features: sqrt
    random_state: 42
```

Listing names under `additional_continuous_covariates` feeds those columns
(lagged `feature_min_lag..feature_max_lag`) to the trend model. Calendar
features are intentionally omitted — the target is already deseasonalized.

## Running with chap eval

```bash
chap eval \
  --model-name /Users/knutdr/Sources/mstl_multistep_model \
  --dataset-csv /Users/knutdr/Data/CH/chap_LAO_admin1_monthly-3.csv \
  --output-file output/lao.nc \
  --model-configuration-yaml config.yaml \
  --backtest-params.n-periods 3 \
  --backtest-params.n-splits 12 \
  --backtest-params.stride 1 \
  --run-config.track
```

(`--run-config.track` records the run to MLflow — see the global CLAUDE.md.)

## Two probabilistic modes

`prob_model` selects how the deseasonalized series is forecast:

- `multistep` (default) — the recursive RF from `simple_multistep_model` with a
  `prob_wrapper` (`bootstrap` / `cross-conformal` / `bucketedresidual`).
- `arima_residual` — a **deterministic** RF point forecast plus **AutoARIMA on
  the RF out-of-bag residuals**. ARIMA supplies horizon-growing predictive
  variance, which the recursive bootstrap badly underestimates. This is the
  inverse of chap_nixtla's `mstl_arima_residual` (ARIMA base + neural residual).

## Benchmark (Lao admin1 monthly, 12 splits × h=3, tracked to MLflow)

| Model | log-CRPS | CRPS | MAE | cov 10–90 |
|---|---|---|---|---|
| `multistep` (RF+bootstrap) | 0.924 | 76.5 | 79.2 | 0.066 |
| `arima_residual` (RF→ARIMA) | 0.766 | 60.7 | 76.4 | 0.484 |
| chap_nixtla `mstl_arima` (baseline) | 0.662 | 53.4 | 68.3 | 0.580 |

The `arima_residual` mode fixes the severe under-dispersion of the bootstrap
mode (coverage 0.066 → 0.484) and improves log-CRPS, but does not yet beat the
classical MSTL+ARIMA baseline.

## Running standalone

```bash
python train.py training_data.csv model.pkl --config config.yaml
python predict.py model.pkl historic.csv future.csv predictions.csv --config config.yaml
```
