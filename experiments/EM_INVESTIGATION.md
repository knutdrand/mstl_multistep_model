# Investigation: an EM / backfitting variant of the rf_residual pipeline

## The idea
Current `rf_residual` is an additive model fit in **one forward pass**:
`log1p(y) = Seasonal(MSTL) + AR/trend(ARIMA) + CovariateEffect(RF) + noise`, where MSTL and
ARIMA are estimated on the raw signal and the RF mops up the leftover ARIMA residual.

EM/backfitting variant: iterate — subtract the RF's covariate-explained pattern from the input,
re-run MSTL+ARIMA on the cleaned signal, re-fit the RF, repeat to a joint fit. The current model
is just iteration 0. This is standard coordinate descent for an additive model (GAM backfitting,
STL-with-xreg).

## Does it make sense here? — empirical headroom probe

Headroom exists only if MSTL/ARIMA currently absorb covariate-driven signal the RF should own.
Measured on an 80-location subset (in-sample, historic through 2025-01):

| quantity | value |
|---|---|
| var(ARIMA fit, within-location, level removed) `A_dm` | 1.534 |
| var(ARIMA residual) `R` | 0.188 |
| **R² of `A_dm` from climate+IRS covariates** | **0.80** (corr 0.91) |
| R² of `R` (what RF currently claims) | 0.67 |
| covariate-explainable variance inside `A_dm` | ≈ 1.23 |

**Conclusion: strong competition.** Climate+IRS covariates can explain ~80% of ARIMA's
within-location dynamics — variance ~1.23 currently attributed to ARIMA's AR/trend, vs the ~0.02
the RF claims from the residual. ARIMA and the covariates are fighting over the same autocorrelated,
climate-driven variation; the single pass gives ARIMA first claim.

(First probe wrongly used location dummies + per-location ARIMA → both encoded the per-location
*level*, giving a spurious corr 0.89. The number above removes that confound. The 0.80 is in-sample
RF so optimistic, but R on the residual is also in-sample 0.67, so the relative overlap is real.)

## Why it could help — and the catch

- **Mean / long horizon (upside).** Because the eval provides future climate, attributing the
  shared variation to the *covariate* effect (which uses known future climate) instead of ARIMA's
  *AR* (which just extrapolates decaying persistence) should forecast better — especially at long
  horizons. This rhymes with the h=12 benchmark, where the AR-flavoured target lags helped 6× more
  than at h=3.
- **Uncertainty (the catch).** ARIMA deliberately owns the predictive spread (σ); the RF is a
  deterministic point. Moving variation from ARIMA → RF would **narrow the intervals** and likely
  hurt log-CRPS/coverage — and this whole session showed the metric is sensitive and the model is
  near its log-CRPS optimum, with recalibrations (sigma, rate, debias) all backfiring. So an EM that
  improves the mean could still lose on log-CRPS unless uncertainty is handled.

## Proposed prototype (with safeguards)
1. Add `em_iterations: int = 1` (1 = current behaviour, exactly reproducible).
2. Per iteration: `Lclean = L − Ĉ`; MSTL(Lclean) → S; ARIMA(deseasonalized) → A; `Ĉ = RF(cov → L−S−A)`.
3. **Cross-fit the RF (OOB / k-fold)** when forming Ĉ, so the RF doesn't fit noise that then
   corrupts the re-decomposition.
4. **Re-estimate ARIMA σ on the cleaned residual** (and/or add damping `Ĉ ← α·Ĉ_new+(1−α)·Ĉ_old`)
   so the spread isn't silently lost — this is the make-or-break detail for log-CRPS.
5. Screen `em_iterations ∈ {1,2,3}` in-process on a location subset; confirm on the full frozen
   harness (log-CRPS + CRPS). Watch coverage closely.

## Verdict (pre-prototype)
Worth prototyping. The headroom is real (strong ARIMA↔covariate competition), and the mechanism
plausibly helps the mean and long-horizon forecasts. The decisive risk is **uncertainty handling**.

---

## RESULT — prototype built and tested: EM does NOT help (negative result)

Prototype: `em_iterations`, `em_damping`, `em_sigma_scale` on the covariate+IRS RF (no target
lags, so predict stays non-circular); each iteration removes the OOB RF covariate effect, re-runs
MSTL+ARIMA on the cleaned signal, re-fits the RF. `em_iterations=1` = current path (untouched).

Subset screen (102 locations × 8 splits), log-CRPS + coverage:

| config | log-CRPS | cov 10–90 |
|---|---|---|
| **em=1 (current)** | **0.3005** | 0.798 |
| em=2 damp=0.5 σ=1.2 (best EM) | 0.3042 | 0.790 |
| em=2 damp=1.0 σ=1.0 | 0.3087 | 0.712 |
| em=3 damp=1.0 σ=1.0 | 0.3103 | 0.700 |

Two findings:
1. **Coverage collapses** 0.80→~0.71 at em>1 — removing the covariate effect shrinks ARIMA's σ, but
   the RF effect is added back as a deterministic point with no uncertainty → under-dispersion.
2. **Re-inflating σ recovers coverage but not log-CRPS.** The best-calibrated EM (damp=0.5, σ=1.2)
   reaches coverage 0.790 ≈ em=1's 0.798, yet log-CRPS is 0.3042 vs **0.3005** — still worse. So once
   calibration is fixed, **the EM mean is simply no better than the single pass.** em=1 won all 7
   comparisons; fewer iterations and more damping (both → em=1) always helped.

**Conclusion.** The 80% ARIMA↔covariate overlap is *benign*: letting ARIMA take first claim of the
autocorrelated covariate signal forecasts just as well as backfitting, and iterating only reshuffles
variance between components (hurting calibration) without improving the mean. The single forward pass
is already optimal for this model/data/metric. EM abandoned. The `em_*` code remains in the model,
inert by default (`em_iterations=1`), in case a future dataset with stronger, less-autocorrelated
covariate effects revisits it.
