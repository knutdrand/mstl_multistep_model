# Investigation: hierarchical reconciliation (sector -> district -> national)

## Idea
The model forecasts 406 sectors, which nest under 46 districts and 1 nation. Hierarchical
reconciliation (bottom-up / MinT) makes forecasts coherent across levels and can improve the
noisy bottom level by borrowing strength from smoother aggregates. Worth trying because
sector-level forecasting here is noise-dominated.

## Phase 0 -- diagnostic (looked promising)
From the champion sector backtest (`scripts/recon_phase0.py`):
- **Aggregation denoises:** normalised RMSE 0.58 (sector) -> 0.39 (district) -> 0.28 (national).
- **Error structure for MinT:** within-district base-error corr 0.27 vs cross-district 0.08
  (overall 0.08; common-shock share 0.02). Roughly block-diagonal W.
Both preconditions for reconciliation appeared met.

## Phase 1 -- independent base forecasts at every level
Built population-weighted district (46) and national (1) datasets
(`scripts/build_hierarchy_datasets.py`, cases-coherent) and ran the champion
(`config_arima012_bank`) on each -> `output/{district,national}_base.nc`.

## Phase 2 -- MinT reconciliation (NEGATIVE)
`scripts/reconcile_mint.py`: build summing matrix S, estimate W (OLS / WLS / MinT-shrink),
reconcile means in count space, mean-shift the sector samples; score log-CRPS/CRPS at sector
and district. 36 interior backtest cells.

| level | base / bottom-up | MinT-shrink | independent district |
|---|---|---|---|
| sector log-CRPS | **0.3123** | 0.3777 | -- |
| sector CRPS | **81.88** | 102.23 | -- |
| district log-CRPS | **0.2070** | 0.2400 | 0.2111 |

(OLS/WLS are numerically unstable in count space given the national/sector scale disparity;
MinT-shrink is the fair comparison.)

## Verdict: reconciliation does NOT help here
The decisive number is **independent district 0.2111 vs bottom-up 0.2070**: forecasting district
totals directly is *not* better than simply summing the champion's sector forecasts. So there is
**no more-accurate aggregate to reconcile toward** -- the Phase-0 denoising is already fully
realised by bottom-up aggregation of the granular, pooled model. MinT's projection (in count
space, with extreme cross-level scale disparity) only adds mis-scaled noise: it degrades sector
(+0.065 log-CRPS) and district (0.207 -> 0.240) forecasts.

Why Phase 0 over-promised: "the aggregate is more predictable (lower relative error)" is true, but
that predictability is captured by **summing coherent sector forecasts**, not by an independent
aggregate model -- so the precondition "a better aggregate forecast exists" fails.

## Practical takeaway
For coherent **district-level** forecasts, use **bottom-up** (sum the champion's sector samples,
i.e. `chap aggregate-eval`): it is the best district forecast (log-CRPS 0.207, beating both an
independent district model 0.211 and MinT 0.240) and is automatically coherent. No reconciliation
layer added; champion unchanged (`config_arima012_bank`).

Caveats: the probabilistic reconciliation used here is a first-cut mean-shift of samples, and
count-space MinT is scale-sensitive; a variance-stabilised-space or fully-probabilistic MinT
might recover a little, but the architecture-level signal (independent-aggregate ~= bottom-up)
caps the available upside.

## Follow-up: bottom-up district intervals are under-dispersed -- and a calibration that fixes it (WIN)

`scripts/coverage_check.py`: the **sector** forecasts are well-calibrated (cov@80% = 0.80,
cov@50% = 0.51), but **bottom-up district** intervals are too narrow -- cov@80% = **0.68**,
cov@50% = **0.39**. Cause: bottom-up sums *independently-drawn* sector samples, so the district
spread = sum of sector variances and ignores the positive within-district error correlation
(~0.27); true district variance is larger. (The independent district model, which forecasts the
district series directly, is well-calibrated at 0.85 -- ~1.6x wider SD.)

**Fix (`scripts/calibrate_district_ci.py`):** learn a per-horizon multiplicative spread factor
from the backtest district residuals (conformal-style: lambda_h = 0.80-quantile of the per-cell
edge ratio, so the 80% interval covers 80%), and scale each district's samples around their mean.
Validated leave-one-target-period-out (out-of-sample).

| district forecast | log-CRPS | CRPS | cov@80% | cov@50% |
|---|---|---|---|---|
| bottom-up (uncalibrated) | 0.2070 | 553.3 | 0.68 | 0.39 |
| **bottom-up + calibrated CI** | **0.2039** | **536.7** | **0.80** (CV 0.795) | **0.49** |
| independent district model | 0.2111 | 523.3 | 0.85 | 0.58 |

lambda = [1.25, 1.26, 1.37] (grows with horizon). Calibrating to nominal coverage **also improves
log-CRPS and CRPS** (the under-dispersion was penalised by the proper scores), and the out-of-sample
(LOPO-CV) coverage matches in-sample -> it generalises. The calibrated bottom-up is now the best
district product: best log-CRPS, nominal coverage, coherent. Plot:
`docs/figs/champion_eval_district_calibrated.html`. (Currently a post-processing step on the
aggregated .nc; could be integrated into the aggregation/model pipeline.)
