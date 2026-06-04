# Future improvements — evidence-based roadmap

Derived from a backtest-error analysis of the champion (`exp/per-covariate-lags`,
`output/irs_lags.nc`: 406 locations × 14 target months × h=1–3 × 100 samples) on the
frozen harness (`chap_data_level5_irs_allocated_monthly.csv`, 12×3×1). Diagnostics use a
sample-based log-CRPS estimate (relative breakdowns are valid; absolute level differs from
chap's `crps_log1p` aggregation).

## What the errors look like

| symptom | evidence | reading |
|---|---|---|
| over-forecast growing with horizon | signed mean err +8.5 / +18.2 / +29.9 at h=1/2/3; mean > median at every h | right-skew **retransformation (Jensen) bias** from `expm1(Normal)`: sample mean ≈ `expm1(μ+σ²/2)`, σ grows with horizon |
| small counts massively over-predicted | 1–9 bucket: signed err **+102%**, coverage 0.688, highest per-obs log-CRPS (0.535) | same Jensen effect (worst where log-space σ is large vs mean) + no count discretization |
| horizon degradation | log-CRPS 0.39 → 0.47 → 0.54 (h=1→3) | partly the bias above, partly genuine multi-step error |
| outbreaks overflow the upper tail | 12.8% of obs > pred q90 (nom 10%), 3.6% > q99 (nom 1%); 200+ bucket coverage 0.74 | symmetric Gaussian-on-log spread underestimates the right tail; counts are overdispersed |
| lower tail OK | 10.8% < pred q10 (nom 10%) | under-dispersion is one-sided (upper only) |
| error is diffuse | worst 10/406 locations = 4.8% of total log-CRPS | no per-location pathology — fixes must be structural, not location-specific |

## Prioritised hypotheses

### 1. Correct the `expm1` retransformation bias  ⭐ highest EV, low effort (code)
The single largest *structured* error. When we draw `Normal(μ, σ)` in log1p space and `expm1`,
the back-transformed mean is inflated by `exp(σ²/2)`; σ grows with horizon, which exactly
reproduces the +8→+30 horizon-growing over-forecast and the +102% small-count over-forecast.
- **Change** (`rf_residual.py`, in log space before `expm1`): subtract the bias, e.g. shift the
  log-space draws by `-σ_h²/2`, or center on the intended conditional mean. Try as a switch
  (`lognormal_debias: true`) so it is one comparable change.
- **Expected:** lower mean/mae and log-CRPS, especially at h=2/3 and on small-count locations.
- **Risk:** low; purely a mean shift. Verify it doesn't over-correct the (already OK) median.

### 2. Heavier / right-skewed predictive tail  ⭐ high EV, medium effort (code)
Outbreaks exceed pred q90/q99 and high-count locations are under-covered, while the lower tail
is fine — the symmetric Gaussian is the wrong shape.
- **Options:** (a) draw ARIMA residuals **empirically** (bootstrap the in-sample ARIMA residuals)
  instead of Gaussian; (b) Student-t innovations; (c) a Negative-Binomial / count layer keyed to
  the predicted mean (models overdispersion directly). (a) is the smallest change and directly
  targets the asymmetry.
- **Expected:** fixes upper-tail coverage (q90/q99) and the 200+ bucket without over-widening the
  body (the failed global `arima_sigma_scale` widened symmetrically — this widens only the right
  tail, which is the actual deficit).

### 3. Count discretization for small counts  — cheap to test (config, already exists)
`discretize_samples: true` Poisson-resamples the expected counts; never evaluated this session.
The 1–9 bucket is the worst-calibrated (cov 0.688). Combine with #1 (debias first, then
discretize). **One config flag — run it next.**

### 4. Per-horizon residual modelling  — medium effort (code)
Error grows with horizon partly beyond the bias. The RF residual corrector is shared across
horizons and the recursion is in ARIMA only. Consider **direct** (per-horizon) RF residual models
(`h`-specific feature rows) or adding horizon as a feature, so the correction can differ by lead time.

### 5. Model rates, not raw counts  — medium effort (code)
`population` is available and unused. Forecasting incidence per capita (offset) then multiplying
back normalises the huge dynamic range (counts 0–569k structures, cases up to 1000s) and may
stabilise both the RF residual and the variance.

### 6. Unexploited covariates / sources  — cheap (config), lower EV
- Alternative rainfall products `rainfall_chirps`, `rainfall_iri` (ensemble or best-of vs `era5`).
- `dewpoint_temperature`, `min_temperature`, `max_temperature` (humidity/temperature extremes).
- `ndvi`/`evi` helped neither as deseasonalised covariates (exp 6) — only revisit as interactions
  or with their own (longer) per-covariate lags now that `covariate_lags` exists.
- District-level hierarchy / partial pooling instead of one-hot location dummies (455 dummies on a
  pooled RF is crude).

## Suggested order
1 (debias) → 3 (discretize) → 2 (skewed tail) → then 4/5 if the tail/horizon gaps persist.
Each is one comparable change on the frozen harness; expect #1–#3 to move log-CRPS most because
they attack the measured biases directly, whereas covariate tweaks (exp 5/6) and HPO have already
shown flat/diminishing returns.

---

## Next directions (post-session synthesis)

The model is at the **log-CRPS optimum for its architecture** — every tuning/recalibration lever
is now flat or negative (debias, pseudocount, sigma, discretize, empirical residuals, EM, seasonal
features, vegetation, lag-window HPO). Only levers that gave the RF new *information* moved the
metric (IRS features, target lags). Further gains need **architectural** change, not knobs.

**Tier 1 (most promising, untested):**
- **Per-horizon / direct multi-step residual modeling.** Horizon degradation is real (log-CRPS
  0.39→0.54, h=1→3) and target lags helped 6× more at h=12 — the residual structure differs by
  lead time, but one RF (trained on ~1-step in-sample residuals) corrects all horizons. Train
  horizon-aware residuals (multi-origin rolling residuals + horizon feature, or H separate models).
  **← prototyping now.**
- **Target-lag variants (cheap):** recursive target lags; lag the *ARIMA residual* itself.

**Tier 2 (the real remaining structural error):**
- **Upper tail / outbreaks.** Right tail uncaptured (12.8%>q90, 3.6%>q99); symmetric sigma failed.
  Swap the point-RF residual for a **quantile gradient-boosting** model → asymmetric residual
  distribution, attacks point + tail together.
- **Metric fork / ensemble.** per-capita rate wins CRPS+coverage decisively; blend it with the
  log-CRPS champion (different errors) for a possible Pareto gain.

**Tier 3:** district-level partial pooling instead of 406 one-hot dummies; alt rainfall sources.
