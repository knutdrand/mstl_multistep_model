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

## Running standalone

```bash
python train.py training_data.csv model.pkl --config config.yaml
python predict.py model.pkl historic.csv future.csv predictions.csv --config config.yaml
```
