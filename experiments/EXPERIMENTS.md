# Experiment log — irs_allocated feature extraction

**Frozen harness** (identical for every row; see `experiments/run_eval.sh`):
- Model: `/Users/knutdr/Sources/mstl_multistep_model`
- Dataset: `/Users/knutdr/Data/CH/chap_data_level5_irs_allocated_monthly.csv` (406 locations, monthly 2013-01..2026-01)
- Backtest: `--n-periods 3 --n-splits 12 --stride 1`, `--run-config.track`
- chap-core: 2.0.0.dev1
- Primary metric: **log_crps** (= `crps_log1p`, lower is better). Secondaries: crps, mae, coverage_10_90 / coverage_25_75.

Goal: extract features from the raw `irs_allocated` allocation column (0–1 coverage
fraction, sparse ~2.5% of rows, ~8-month allocation runs in 188/406 locations) in the
**model code**, and beat the climate-only baseline. ~14 min/eval.

| # | branch / commit | lever | hypothesis / change | log_crps | crps | mae | cov 10-90 / 25-75 | verdict |
|---|---|---|---|---|---|---|---|---|
| 0 | improve/spray-setup @2c6eebe | — | baseline: rf_residual + deseasonalized climate, **no IRS** (config_irs_baseline.yaml) | 0.3253 | 83.6 | 113.3 | 0.757 / 0.477 | baseline |
| 1 | exp/irs-raw-covariate @1217766 | config | + raw irs_allocated as lagged covariate (1–3) | 0.3247 | 83.6 | 113.4 | 0.758 / 0.478 | flat (noise) — sparse col ~all zeros in lag window |
| 2 | exp/irs-code-features @9fbcc4f | **code** | engineer dense IRS features `level+decay+since+cumulative` (hl 4) at lag 0 | **0.3222** | 83.4 | 113.3 | 0.762 / 0.480 | **improved** ✅ −0.0031 vs baseline; tag `promising/irs-code-features` |
| 3 | exp/irs-halflife-8 @4da1cd4 | config | irs_halflife 4 → 8 | 0.3223 | 83.4 | 113.2 | 0.762 / 0.481 | flat — decay half-life insensitive 4↔8 |
| 4 | exp/irs-feat-subset @ce16893 | config | IRS subset `[level, decay]` only | 0.3239 | 83.5 | 113.4 | 0.759 / 0.480 | regressed — since+cumulative carry ~half the gain |
| 5 | exp/arima-sigma-scale @57148f4 | code | arima_sigma_scale = 1.10 (widen intervals) | 0.3227 | 83.8 | 113.5 | **0.800** / 0.519 | regressed on log-CRPS though coverage hit nominal — model already near CRPS-optimal dispersion |
| 6 | exp/add-vegetation @afb9040 | config | + ndvi, evi (deseasonalized) covariates | 0.3221 | 83.9 | 113.9 | 0.761 / 0.481 | flat on log-CRPS, **worse** crps/mae — veg adds point noise |
| 7 | exp/rf-leaf3 @f29393c | config | RF min_samples_leaf 5 → 3 (on exp2) | **0.3221** | **83.2** | **113.0** | 0.762 / 0.482 | **best** — log-CRPS tied w/ exp2 but crps/mae better, no regression |
| 8 | exp/rf-leaf2 @77f1f41 | config | RF min_samples_leaf 3 → 2 (on exp7) | 0.3225 | 82.9 | 112.5 | 0.761 / 0.481 | regressed on log-CRPS — leaf<3 overfits log-scale tails; crps/mae keep dropping |
| 9 | exp/per-covariate-lags @9d49cd6 | **code** | per-covariate lags: rainfall [1,6], humidity [1,4], temp [1,2] (on exp7) | **0.3217** | 83.8 | 113.7 | 0.764 / 0.482 | **CHAMPION** ✅ best log-CRPS (−1.1% vs baseline); crps/mae regress ~0.6% (log-scale vs raw-scale trade-off, accepted) |

## Outcome

- **Robust win = IRS code feature extraction (exp 2).** Turning the sparse raw
  `irs_allocated` column into dense contemporaneous features (level + geometric-decay
  persistence + months-since + cumulative stock, all at lag 0 since allocation is a
  *known* future covariate) lowered log-CRPS from **0.3253 → 0.3222** (−0.0031, −0.95%).
  The naive config raw-lag path (exp 1) did **not** — the sparse column is ~all zeros
  inside a 1–3 month lag window.
- **Champion = exp 9** (`exp/per-covariate-lags`, tag `best/per-covariate-lags`): IRS
  features + `min_samples_leaf=3` + per-covariate lags (rainfall [1,6], humidity [1,4],
  temp [1,2]). **log-CRPS 0.3217 — best of the session, −1.1% vs baseline.** Trades ~0.6%
  on crps/mae (longer moisture lags help log-scale small-count calibration at a small cost
  to raw-count accuracy); accepted since log-CRPS is the primary metric. The intermediate
  exp 7 (`exp/rf-leaf3`) remains the best *balanced* config (0.3221, best crps/mae).
- **Negative results:** decay half-life is insensitive (3); the full 4-feature IRS set
  beats a subset (4); global sigma inflation fixes coverage but hurts log-CRPS (5);
  vegetation indices add point-forecast noise (6); `min_samples_leaf<3` overfits the
  log-scale tails (8).

## Reproduce the champion

```bash
git checkout exp/per-covariate-lags
experiments/run_eval.sh config_irs_lags.yaml irs_lags        # -> output/irs_lags.nc
chap export-metrics --input-files output/irs_baseline.nc --input-files output/irs_lags.nc \
  --output-file experiments/comparison.csv
```

## Long-horizon benchmark — 12-month-ahead, 14 splits (separate harness)

`--backtest-params.n-periods 12 --n-splits 14 --stride 1` on the same dataset. NOT comparable
to the 12×3×1 numbers above (much harder horizon). Best models only; log-CRPS / CRPS.

| model | log-CRPS | CRPS |
|---|---|---|
| baseline (no IRS) | 0.5137 | 96.70 |
| + IRS features | 0.5132 | 95.76 |
| + per-covariate lags | 0.5120 | 95.78 |
| **+ target lags=3 (champion, config_tgtlag3)** | **0.5033** | **95.14** |

**Findings:** the full progression stays monotonic at long range — champion best on both metrics
(vs baseline: log-CRPS −2.0%, CRPS −1.6%). **Target lags are far more valuable here** (−0.0087
log-CRPS, vs −0.0014 at h=3): over a 12-month horizon the ARIMA-mean-bridged AR signal the RF
captures matters much more. IRS features help CRPS (96.70→95.76) with a small log-CRPS gain.
The improvements generalize to and amplify at long-range forecasting.

## Seasonal features on the residual RF (champion + Fourier)

Tested adding Fourier seasonal features (sin/cos annual cycle, lag 0) to the champion's
residual RF. Full frozen harness, log-CRPS / CRPS.

| model | log-CRPS | CRPS |
|---|---|---|
| **champion (no seasonal, config_tgtlag3)** | **0.3203** | **82.97** |
| + Fourier order 2 | 0.3210 | 83.24 |
| + Fourier order 3 | 0.3212 | 83.46 |

**Negative result.** Seasonal features hurt both metrics, monotonically with harmonics —
confirming the model's design choice to omit them: MSTL removes seasonality cleanly enough that
the ARIMA residual has no exploitable seasonal structure, so the sin/cos terms only add overfit
noise. `seasonal_fourier_order` left inert by default (0).

## Per-horizon residual prototype (FUTURE #1) — subset screen

Multi-horizon training residuals + horizon feature (rf_horizon_feature), vs baseline (one RF on
1-step residuals), on the per-covariate-lags base (rf_target_lags=0). 68-location subset × 6 splits.

| forecast horizon | baseline log-CRPS / cov | per-horizon log-CRPS / cov |
|---|---|---|
| h=3 | **0.3038 / 0.792** | 0.3126 / 0.772 |
| h=12 | 0.4348 / 0.755 | **0.4268 / 0.768** |

**Horizon-dependent: hurts at h=3, helps at h=12** (−0.008 log-CRPS + better coverage). The
single-RF baseline trains on ~1-step residuals; applied to long-horizon forecasts (ARIMA reverted
to mean) it under-corrects, and the horizon-aware RF fixes it. At h=3 there is no mismatch and the
noisier multi-origin training residuals just add noise. Confirming on the full h=12 harness next.

### Per-horizon — full h=12 confirmation, then SHELVED

Full h=12/14-split harness (per-horizon on the per-cov-lags base, rf_target_lags=0):

| h=12 model | log-CRPS | CRPS | cov |
|---|---|---|---|
| per-cov lags | 0.5120 | 95.78 | 0.720 |
| champion tgtlag3 | 0.5033 | **95.14** | 0.727 |
| **per-horizon** | **0.4980** | 96.77 | 0.732 |

Per-horizon (no target lags) **beats the champion on log-CRPS at h=12** (0.4980 vs 0.5033, +coverage)
but is worse on CRPS. Combined with the screen (hurts at h=3: 0.3126 vs 0.3038), per-horizon is a
**long-horizon-only** technique.

**Decision: SHELVED.** h=3 (12×3×1) is the main harness, where per-horizon hurts. Code left inert
by default (`rf_horizon_feature=False`). Revisit only for long-range (h≫3) forecasting; the
per-horizon + target-lags combination remains an untested long-horizon idea.

## Main-harness champion (h=3, reaffirmed)

**`config_tgtlag3` (IRS features + per-covariate lags + min_samples_leaf=3 + rf_target_lags=3):
log-CRPS 0.3203, CRPS 82.97.** All further h=3 improvements measured against this.

## Target-lag source: ARIMA residual R vs deseasonalized target D (h=3)

| model | log-CRPS | CRPS |
|---|---|---|
| **champion (lag D, deseason)** | **0.3203** | **82.97** |
| lag R = D−A (rf_target_lag_source=residual) | 0.3215 | 83.58 |

**Negative result.** Lagging the ARIMA residual R is worse on both metrics. ARIMA whitens R well
enough that its lags carry little signal, whereas lagged D gives the RF the recent level/trend
(which correlates with the residual). The `deseason` source is validated; `rf_target_lag_source`
left inert at default `deseason`. Champion unchanged.
