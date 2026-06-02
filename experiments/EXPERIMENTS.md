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

## Outcome

- **Robust win = IRS code feature extraction (exp 2).** Turning the sparse raw
  `irs_allocated` column into dense contemporaneous features (level + geometric-decay
  persistence + months-since + cumulative stock, all at lag 0 since allocation is a
  *known* future covariate) lowered log-CRPS from **0.3253 → 0.3222** (−0.0031, −0.95%).
  The naive config raw-lag path (exp 1) did **not** — the sparse column is ~all zeros
  inside a 1–3 month lag window.
- **Champion = exp 7** (`exp/rf-leaf3`): IRS features + `min_samples_leaf=3`. log-CRPS
  0.3221 (tied with exp 2 within noise) with the best crps (83.2) and mae (113.0). The
  leaf=3 gain over exp 2 on log-CRPS is within noise; the durable improvement is the IRS
  features.
- **Negative results:** decay half-life is insensitive (3); the full 4-feature IRS set
  beats a subset (4); global sigma inflation fixes coverage but hurts log-CRPS (5);
  vegetation indices add point-forecast noise (6); `min_samples_leaf<3` overfits the
  log-scale tails (8).

## Reproduce the champion

```bash
git checkout exp/rf-leaf3
experiments/run_eval.sh config_irs_leaf3.yaml irs_leaf3      # -> output/irs_leaf3.nc
chap export-metrics --input-files output/irs_baseline.nc --input-files output/irs_leaf3.nc \
  --output-file experiments/comparison.csv
```
