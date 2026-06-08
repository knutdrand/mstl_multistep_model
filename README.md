# mstl_multistep_model

A CHAP-compatible probabilistic forecaster for monthly disease cases (e.g. malaria) at many
locations. It combines a classical time-series base model with a machine-learning correction:

> **MSTL** seasonal decomposition → a shared fixed-order **ARIMA** on the deseasonalised series →
> a pooled **RandomForest** that corrects the ARIMA residual using lagged climate anomalies,
> engineered indoor-residual-spraying (IRS) features and self-history → a **location-scale
> variance head** that widens the predictive interval by the forest's own uncertainty.

On the reference dataset (406 Rwanda sectors, monthly, 3-month horizon, 12 backtest splits) it
reaches **log-CRPS 0.3123 / CRPS 81.88** — a ~4% log-CRPS improvement over a climate-only baseline.

---

## How it works

The model works entirely in `log1p` space, `L = log(1 + cases)`, and writes that signal per
location `ℓ` and month `t` as a sum of three estimated components plus noise:

```
L(ℓ,t) = S(ℓ,t)          seasonal              (MSTL)
       + A(ℓ,t)          AR / trend            (ARIMA)
       + g(x(ℓ,t))       covariate correction  (RandomForest)
       + ε
```

Each component is fit by the estimator best suited to it. The forecast is Gaussian in `log1p`
space and is mapped back to counts with `expm1`.

### Step 1 — MSTL seasonal decomposition
Each location's `log1p` series is decomposed by MSTL (annual period 12 for monthly data) into a
**seasonal** component `S` and a **deseasonalised** remainder `D = L − S` (trend + remainder).
`S` is extrapolated *seasonal-naive* into the forecast window; `D` is what the rest of the
pipeline forecasts.

### Step 2 — Shared fixed-order ARIMA base
A separate ARIMA is fit per location to `D`, but **all locations share the same fixed order**
(`arima_order`, default `[0, 1, 2]`) — only the coefficients are estimated per series. This
replaced per-location order *selection*, which on short, noisy series is itself a source of
noise; one robust order generalises better and was the single biggest accuracy improvement found.
The data's modal auto-selected order is `[0, 1, 1]` — i.e. simple exponential smoothing.

ARIMA plays two roles:
- **Mean** — the h-step forecast `μ_h` carries persistence/trend; its in-sample one-step fit `Â`
  defines the residual `R = D − Â` that the RandomForest learns.
- **Spread** — the predictive interval (`arima_level`, default 68%) is converted to a Gaussian
  standard deviation `σ_h` that grows with horizon. This is the model's main source of uncertainty.

### Step 3 — RandomForest residual correction
A single **pooled** RandomForest `g` is trained across all locations to predict the ARIMA
residual `R` from a feature vector — a deterministic *point* correction that nudges the mean using
structure the univariate ARIMA cannot see. The feature vector has four blocks:

1. **Lagged climate anomalies** — each climate covariate is MSTL-deseasonalised first (so the
   model sees anomalies), then lagged. Per-covariate lag windows reflect the physical delay from
   weather to transmission (e.g. rainfall 1–6, humidity 1–4, temperature 1–2 months).
2. **Engineered IRS features** — the sparse `irs_allocated` campaign column is turned into a dense
   *feature bank* (see below).
3. **Self-history** — the deseasonalised target `D` lagged `rf_target_lags` (default 3) months;
   future lags are bridged with the ARIMA mean.
4. **Location identity** — one-hot location dummies, so the pooled forest retains per-series level.

### Step 4 — Location-scale variance head
The RF correction is a point, so on its own it injects no uncertainty. The variance head restores
that: the forest's **inter-tree variance** `v(x)` (how much its trees disagree) is folded into the
spread,

```
σ_eff² = σ_ARIMA(h)² + residual_variance_scale · v(x)
```

(`residual_variance: tree`, `residual_variance_scale: 0.5`). This is the honest "total spread =
forecast variance + variance of the correction", and it improves both log-CRPS and CRPS.

### Step 5 — Reconstruction and sampling
For each location and horizon step, `n_samples` (default 100) draws are taken from
`N(S + μ_h + g(x), σ_eff²)`, mapped back to counts with `expm1`, and clipped at zero. Those
samples are the probabilistic forecast, scored by log-CRPS (primary) and CRPS.

A fuller write-up with figures and the SES / ARIMA(0,1,1) math is in `docs/champion_model.pdf`,
`docs/arima_in_model.pdf` and `docs/arima011_math.pdf`.

---

## Installation

Uses [uv](https://docs.astral.sh/uv/). No git dependencies:

```bash
uv sync
```

---

## Configuration

CHAP hands the model a YAML in the *model configuration* shape: the covariate columns under
`additional_continuous_covariates` and model knobs under `user_option_values`. **All defaults are
the published champion**, so a config typically only needs to name its covariate / IRS columns and
(optionally) tune lags. `config.yaml` is the reference config:

```yaml
additional_continuous_covariates: [rainfall_era5, mean_temperature, relative_humidity]
user_option_values:
  covariate_lags: {rainfall_era5: [1, 6], relative_humidity: [1, 4], mean_temperature: [1, 2]}
  deseasonalize_covariates: [rainfall_era5, mean_temperature, relative_humidity]
  irs_features: [level, since, cumulative, chem_channels, decay2, decay8, recent3, recent6, recent12, rounds12]
```

The IRS inputs `irs_allocated` and `irs_insecticide_used` are declared as **`required_covariates`**
in the `MLproject` (read from those fixed column names); climate covariates are supplied freely
under `additional_continuous_covariates`.

### Option reference

| group | option | default | meaning |
|---|---|---|---|
| data | `target_variable` | `disease_cases` | column to forecast |
| | `log_transform` | `true` | model in `log1p` space |
| | `season_length_monthly` / `_weekly` | 12 / 52 | MSTL seasonal period |
| features | `feature_min_lag`, `feature_max_lag` | 1, 3 | global covariate lag window |
| | `covariate_lags` | `{}` | per-covariate `{name: [min, max]}` override |
| | `deseasonalize_covariates` | `[]` | covariates to MSTL-deseasonalise before lagging |
| | `use_location_dummies` | `true` | one-hot location identity |
| | `rf_target_lags` | 3 | lagged deseasonalised-target features |
| ARIMA | `arima_order` | `[0, 1, 2]` | shared `(p,d,q)` order for all locations |
| | `arima_level` | 68 | interval level used to back out σ |
| IRS | `irs_features` | `[]` | which IRS features to engineer from the required IRS columns (see below) |
| | `irs_halflife` | 4.0 | decay half-life (months) for the plain `decay` |
| variance head | `residual_variance` | `tree` | `tree` widens spread by forest variance; `none` off |
| | `residual_variance_scale` | 0.5 | weight of `v(x)` in `σ_eff²` |
| output | `n_samples` | 100 | number of posterior samples |
| | `random_seed` | 42 | RNG seed |
| | `rf` | n_est 200, leaf 3, sqrt | RandomForest hyperparameters |

### IRS feature bank
`irs_features` selects which dense features to engineer from the required IRS columns
(`irs_allocated`, the sparse campaign column, and `irs_insecticide_used`); all contemporaneous,
since spraying is known in advance:

- `level` — this month's allocation coverage; `since` — months since last campaign; `cumulative`
  — campaign-count stock.
- `decay` — geometric-decay protection (`irs_halflife`); `decay2`, `decay8` — a decay basis at
  half-lives 2 and 8 months.
- `chem_channels` — **per-chemical decay channels** `decay_{carbamate, pyrethroid,
  organophosphate, clothianidin}`, each active only while that insecticide class is current
  (from `irs_insecticide_used`), so the forest learns each chemical's effect magnitude. Literature
  half-lives: carbamate 2, pyrethroid 3, organophosphate 5, clothianidin 8 months.
- `recent3/6/12` — sprayed within the last k months; `rounds12` — campaigns in the trailing year.

---

## Usage

The model is a CHAP external model. Evaluate it with `chap eval`:

```bash
chap eval \
  --model-name . \
  --dataset-csv /path/to/data.csv \
  --model-configuration-yaml config.yaml \
  --backtest-params.n-periods 3 \
  --backtest-params.n-splits 12 \
  --backtest-params.stride 1 \
  --run-config.track          # records to MLflow

# headline metrics (log-CRPS is primary)
chap export-metrics --input-files output/eval.nc \
  --metric-ids crps_log1p --metric-ids crps --output-file metrics.csv
```

Programmatic use:

```python
from mstl_multistep import build_chap_model, load_model_configuration

mc = load_model_configuration("config.yaml")
model = build_chap_model(mc.user_option_values, mc.additional_continuous_covariates)
model.fit(historic_df)                            # time_period, location, target, covariates
samples = model.predict(historic_df, future_df)   # wide frame of sample_0..sample_{n-1}
```

The input is a long panel with `time_period` (`YYYY-MM`), `location`, the target column, and any
covariate / IRS columns named in the config.

---

## Interval calibration (optional)

Sector forecasts are well-calibrated on average, but two interval adjustments — learned from the
backtest residuals (conformal: fit-on-backtest → apply) — are available in
`mstl_multistep/calibration.py` and via `scripts/calibrate_forecast.py`:

- **per-sector** (`--level sector`) — fixes coverage heterogeneity across locations (log-space,
  shrunk; per-location coverage std 0.12 → 0.05).
- **per-district** (`--level district`) — undoes the under-dispersion of bottom-up aggregation
  (district coverage 0.68 → 0.80, and also improves district log-CRPS).

```bash
chap aggregate-eval output/eval.nc areas.geojson output/eval_district.nc
uv run python scripts/calibrate_forecast.py output/eval_district.nc --level district \
    --out output/eval_district_cal.nc
```

---

## Repository layout

```
config.yaml                   the published default config (= champion)
mstl_multistep/
  rf_residual.py              the model (ArimaBaseRFResidualModel: fit / predict)
  decomposition.py            MSTL helpers
  features.py                 lagged-covariate + location-dummy features
  irs_features.py             the IRS feature bank
  calibration.py              interval calibration (fit on backtest, apply to forecasts)
  run_config.py               config schema (RunConfig)
  pipeline.py                 build_chap_model entry point
scripts/calibrate_forecast.py calibration CLI
tests/                        test_pipeline.py (smoke) + test_golden.py (regression)
docs/                         PDF write-ups of the model + the ARIMA math
experiments/                  methodology notes + archived experiment configs/scripts
```

## Development

```bash
uv run pytest tests/          # smoke + golden regression
```

`tests/test_golden.py` pins the champion's forecast bit-for-bit — run it after any change to the
model to confirm behaviour is preserved.
