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

This project uses [uv](https://docs.astral.sh/uv/). The trend model
`simple_multistep_model` is pulled directly from GitHub (pinned to a commit in
`pyproject.toml`), so no sibling checkout is needed:

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
  deseasonalize_covariates:                 # covariate names to MSTL-deseasonalize
    - rainfall                              # (others kept raw — e.g. list climate but
    - mean_temperature                      #  not seasonal sprayed_* intervention flags)
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
- `rf_residual` — **AutoARIMA is the base** forecaster on the deseasonalized
  series (mean + horizon-growing σ) and a **deterministic RF models the ARIMA
  residual** using climate-anomaly features. ARIMA owns the spread, RF nudges
  the mean. Same shape as chap_nixtla's `mstl_arima_residual` but with RF as the
  residual learner. **Best mode — beats the classical baseline (see below).**
- `arima_residual` — the inverse: a deterministic RF point forecast plus
  **AutoARIMA on the RF out-of-bag residuals**. Fixes the bootstrap mode's
  under-dispersion but stays just behind the baseline.
- `recursive_residual` — deterministic RF point with one-step OOB residuals
  resampled and compounded through the recursion. Fixes dispersion but scores
  *worse* log-CRPS than the other residual modes; kept only for comparison.

The recommended config (`config_rf_residual.yaml`) is `rf_residual` +
deseasonalized climate covariates, RF `max_depth=null`.

## Benchmark — Lao admin1 monthly, 12 splits × h=3, tracked to MLflow

| Model | log-CRPS | CRPS | MAE | cov 10–90 |
|---|---|---|---|---|
| `multistep` (RF+bootstrap) | 0.924 | 76.5 | 79.2 | 0.066 |
| `arima_residual`, no covariates | 0.766 | 60.7 | 76.4 | 0.484 |
| `arima_residual` + raw climate | 0.727 | 56.0 | 71.1 | 0.493 |
| `arima_residual` + deseasonalized climate (tuned) | 0.712 | 54.1 | 69.9 | 0.510 |
| chap_nixtla `mstl_arima` (baseline) | 0.662 | 53.4 | 68.3 | 0.580 |
| **`rf_residual` (ARIMA base + RF correction)** | **0.633** | **50.8** | **66.0** | 0.587 |

Story: `arima_residual` fixed the bootstrap mode's severe under-dispersion
(coverage 0.066 → 0.484) and adding deseasonalized climate brought it level with
the baseline (ablation: most of the climate gain is from adding it at all,
0.766 → 0.727, deseasonalizing a smaller extra → 0.716; HPO ~0.004 more). But
the decisive change was **swapping the roles** — making ARIMA the base and RF
the residual corrector (`rf_residual`) — which beats the classical baseline on
every metric.

## Benchmark — VNM admin1 monthly (generalization check)

| Model | log-CRPS | CRPS | MAE | cov 10–90 |
|---|---|---|---|---|
| `arima_residual` | 0.549 | 75.4 | 97.5 | 0.624 |
| chap_nixtla `mstl_arima` (baseline) | 0.508 | 71.0 | 94.1 | 0.729 |
| **`rf_residual`** | **0.498** | **70.0** | **93.5** | 0.725 |

`rf_residual` beats the baseline on **both** datasets (Lao 0.633 vs 0.662, VNM
0.498 vs 0.508), on log-CRPS, CRPS and MAE, with coverage matching. **ARIMA is
the better base for these AR/seasonal series; RF adds the nonlinear
climate-anomaly correction a univariate ARIMA can't — together they edge out
classical MSTL+ARIMA.**

## Benchmark — level-5 spray dataset (406 locations, with intervention covariates)

`rf_residual` with climate (deseasonalized) + raw `sprayed_this_season` /
`sprayed_last_season` (`config_spray.yaml`):

| metric | value |
|---|---|
| log-CRPS | 0.326 |
| CRPS / MAE | 83.3 / 113.1 |
| coverage 10–90 / 25–75 | 0.757 / 0.477 (nominal 0.80 / 0.50) |

Best-calibrated of the three datasets — the larger panel gives the ARIMA base
more signal. The list-form `deseasonalize_covariates` keeps the seasonal
`sprayed_*` intervention flags raw while deseasonalizing climate.

Leaderboard vs every other model previously run on this dataset (MLflow,
identical 12×3×1 backtest and dataset hash `2aa64188`):

| Model | log-CRPS |
|---|---|
| **`rf_residual` (this model)** | **0.3258** |
| `mstl_arima` | 0.3304 |
| `mstl_arimax` | 0.3331 |
| `mstl_nhits` | 0.3430 |
| `mstl_arima_residual` | 0.3443 |
| `joint_structural` (best) | 0.3649 |
| `chtorch` (best) | 0.3918 |

`rf_residual` has the best log-CRPS of all models tried here. (`chap_pymc` 0.378
and `simple_multistep` 0.377 ran on a different dataset version, hash
`ffe9ffaa`, so are excluded as not directly comparable.)

## Running standalone

```bash
python train.py training_data.csv model.pkl --config config.yaml
python predict.py model.pkl historic.csv future.csv predictions.csv --config config.yaml
```
