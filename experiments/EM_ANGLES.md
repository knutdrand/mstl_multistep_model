# Revisiting EM / coordinate descent — second pass

The first EM attempt (`EM_INVESTIGATION.md`) was **mean-backfitting**
(covariate-effect ↔ MSTL+ARIMA) and came back a clean negative. Its decisive
finding: *once calibration is equalized, the EM mean is no better than the
single pass* — the 80% ARIMA↔covariate overlap is benign.

Corollary that shapes this second pass: **any coordinate descent whose goal is a
better mean will re-confirm that negative.** The productive angles must attack a
*different coordinate* than the conditional mean. The unexplored coordinate is
the **predictive spread / distribution shape** — the model draws a symmetric
Gaussian `N(mean_h, σ_ARIMA²)` and the data has a heavy right tail (outbreaks).
log-CRPS rewards distributional fit, so this is where slack remains.

The first EM's own failure names the fix it lacked: the RF mean-correction has
uncertainty that was thrown away ("a deterministic point with no uncertainty").
Honest predictive variance is

    σ²_total(x, h) = σ²_ARIMA(h) + v(x)

where `v(x)` is the variance of the RF correction. Estimating the two heads
(mean, variance) and alternating is **location–scale coordinate descent** (IRLS
on the Gaussian NLL) — EM-adjacent, and it leaves the champion's proven mean
untouched.

## Angles

1. **Location–scale coordinate descent (PRIMARY).** Keep the champion mean RF.
   Add a variance head `v(x)`; predictive spread becomes
   `sqrt(σ²_ARIMA + scale·v(x))`. Two estimators, screened head-to-head:
   - `tree` — RF inter-tree variance (epistemic uncertainty of the correction).
     Cheap; reuses the mean RF.
   - `model` — a GBM regressing the **squared OOB residuals** on covariates
     (aleatoric, heteroscedastic: outbreak-prone context → wider interval).
   IRLS option: refit the mean RF weighted by `1/v` and iterate
   (`residual_variance_iterations`) — the actual coordinate-descent loop.
   Default `residual_variance="none"` reproduces the champion bit-for-bit.

2. **Variance-honest mean-EM (secondary, direct re-test of the doc claim).**
   The old EM scaled σ by a global constant. Replace that with the per-point
   `tree` variance added in quadrature, so the EM mean improvement (if any) is
   no longer paid for in calibration. Tests the doc's "mean is no better" claim
   under honest local variance rather than a global scale.

3. **Regime-mixture EM (held in reserve).** True latent-variable EM over
   {baseline, outbreak} regimes; E-step responsibilities, M-step per-regime
   spread → a mixture predictive targeting the right tail. Most work; only if
   1–2 show signal that the spread axis is live.

## Protocol
Subset screen first (in-process, ~100 locations, like `scripts/hpo_em.py`),
report **log-CRPS + coverage(10–90) + frac>q90/q99**; confirm any subset winner
on the full frozen h=3 harness (406 loc, n-splits 12, n-periods 3, stride 1)
with `--metric-ids crps_log1p --metric-ids crps`. Champion to beat:
`config_tgtlag3` → log-CRPS 0.3203, CRPS 82.97.

## Results

### Angle 1 — location-scale variance head, h=3 subset (82 loc × 6 splits)

| config | log-CRPS | cov 10–90 | f>q90 | f>q99 |
|---|---|---|---|---|
| **champion (none)** | 0.3278 | 0.760 | 0.121 | 0.027 |
| tree scale=0.25 | **0.3277** | 0.780 | 0.111 | 0.020 |
| tree scale=0.5 | 0.3278 | **0.797** | 0.103 | 0.016 |
| tree scale=1.0 | 0.3286 | 0.833 | 0.081 | 0.012 |
| model scale=0.25–1.0 | 0.3279 | 0.768–0.790 | — | 0.016–0.022 |

Unlike every prior spread-axis lever (which *hurt* log-CRPS), the `tree`
variance head is **log-CRPS-neutral and calibration-improving**: at scale≈0.5 it
moves coverage 0.760→0.797 (to nominal) and the right-tail exceedance
f>q99 0.027→0.016, at log-CRPS tied with champion. The earlier thin 13-loc run
showed over-coverage (0.808) — small-sample noise; the truer 82-loc picture is
that the champion is mildly **under-dispersed** at h=3, and the variance head
corrects it essentially for free. The `model` (GBM-on-OOB²) head behaves the
same but slightly weaker than `tree`. log-CRPS deltas are within subset noise;
the calibration direction is consistent.

Read: at h=3 there is little log-CRPS to win (the body is nearly calibrated),
but the variance head is the first lever that improves calibration without a
log-CRPS cost — the honest-uncertainty reframe of EM works as designed.

### Angle 1 — location-scale variance head, h=12 subset (51 loc × 4 splits)

| config | log-CRPS | cov 10–90 | f>q90 | f>q99 |
|---|---|---|---|---|
| champion (none) | 0.4719 | 0.752 | 0.176 | 0.040 |
| **model scale=1.0** | **0.4693** | 0.786 | 0.156 | 0.030 |
| **tree scale=0.5** | **0.4694** | 0.793 | 0.152 | 0.029 |
| model scale=2.0 | 0.4696 | 0.803 | 0.145 | 0.027 |
| tree scale=1.0 | 0.4701 | 0.816 | 0.137 | 0.022 |
| tree scale=2.0 | 0.4731 | 0.849 | 0.113 | 0.017 |

**Decisive.** At long horizon ARIMA reverts to mean and the predictive intervals
are genuinely too narrow (champion cov 0.752, f>q99 0.040 ≈ 4× nominal). Here the
variance head is a real **log-CRPS win**, not just calibration: champion ranks
5th of 6, beaten by every modest-scale variant. Sweet spot scale≈0.5–1.0
(scale=2.0 over-widens → log-CRPS worsens). `model` and `tree` are
near-identical; `tree` is far cheaper (no GBM) so it's the preferred form.

## Verdict — the EM angle that works

The first EM (mean-backfitting) failed because coordinate descent on the
**mean** only redistributes an information set the one-pass already splits
near-optimally. The working angle moves the descent to the **variance**
coordinate: propagate the RF correction's uncertainty into the predictive
spread (`σ²_total = σ²_ARIMA + scale·v(x)`), the honest-uncertainty step the
first EM threw away.

- **h=3** (body already calibrated): log-CRPS-neutral, small calibration gain — free.
- **h=12** (intervals genuinely too narrow): a real log-CRPS win + much better
  calibration and tail.

So the user's intuition was right — EM/coordinate descent *does* help here — but
the leverage is in the spread, not the mean, and it grows with horizon. Recommend
adopting `residual_variance="tree"`, `residual_variance_scale≈0.5` as a calibration
option (default-off preserves the champion bit-for-bit). `model` and IRLS add cost
without beating plain `tree`.

### Full-harness confirmation — h=3 main harness (406 loc, 12×3×1)

`config_tgtlag3_var` = champion + `residual_variance=tree`, `scale=0.5`
(`output/tgtlag3_var.nc` vs champion `output/irs_tgtlag3.nc`):

| config | log-CRPS | CRPS |
|---|---|---|
| champion (config_tgtlag3) | 0.320328 | 82.972 |
| **+ variance head (config_tgtlag3_var)** | **0.320022** | **82.881** |
| Δ | −0.000306 (−0.10%) | −0.091 (−0.11%) |

**Confirmed: a small win on BOTH metrics** on the frozen h=3 harness — notable
because almost every other lever in this project traded log-CRPS against CRPS;
the honest variance head improves them together (it removes a mild
under-dispersion rather than re-allocating mass). The gain is marginal at h=3 by
construction (the body is nearly calibrated); the subset evidence says it is
several times larger at h=12. New champion at h=3: **config_tgtlag3_var
→ log-CRPS 0.320022, CRPS 82.881**.

### Angle 1c — why IRLS iterations regress (`residual_variance_iterations>1`)

The variance head can be iterated (IRLS): refit the mean RF weighted by `1/v`, re-estimate
`v`, repeat. The champion uses one pass. Instrumented investigation (`scripts/investigate_irls.py`,
82 loc × 3 splits, tree mode, scale 0.5):

| iters | log-CRPS | CRPS | cov | OOB-MSE outbreak pts |
|---|---|---|---|---|
| 1 (champion) | 0.3429 | **157.20** | **0.786** | 0.826 |
| 2 | 0.3423 | 157.72 | 0.779 | 0.831 |
| 3 | 0.3427 | 158.43 | 0.775 | 0.837 |

log-CRPS is flat (noise), but **CRPS and coverage degrade monotonically**. Mechanism (direct
from the instrumented loop): the IRLS weight `w=1/v` correlates **−0.53/−0.55** with `|R|` and
gives top-decile-`|R|` (outbreak) points only **~0.5×** the weight of easy points. So each
iteration shrinks the mean correction on outbreaks (`|corr|` on top-decile points 0.329→0.269→0.267,
~19%) and raises OOB error there (0.826→0.837), while easy points barely improve. Count-scale
CRPS and tail coverage are outbreak-dominated → they worsen; log-CRPS (compressed scale,
outbreaks downweighted) → indifferent.

**Root cause:** `1/v` (GLS/IRLS) reweighting is valid only for *aleatoric* noise independent of
signal. Here `v` is the RF's *epistemic* inter-tree variance — largest exactly at the
high-signal outbreak points — so inverse-`v` weighting discards the highest-information
observations. `v`'s value is in the **spread** (iteration 1 uses it correctly, widening the
interval at outbreaks), **not** as a mean-fit weight. One pass is correct by construction;
`residual_variance_iterations` stays inert (default 1).

### Angle 1b — horizon-aware variance scale (NEGATIVE)

Hypothesis: v(x) is roughly flat across horizon while ARIMA's σ_h grows, so the
head's relative contribution shrinks with horizon — and since the 1-step-trained
RF correction should get *less* reliable further out, multiplying the per-step
scale by `step**power` ought to help long horizon without hurting h=3. Added
`residual_variance_horizon_power` (0.0 = flat, exactly reproducible).

Result — **falsified, robust negative in both horizons.** log-CRPS worsens
monotonically with `power`, coverage over-inflates:

| | p=0 (flat) | p=0.5 | p=1.0 | p=2.0 |
|---|---|---|---|---|
| h=3 log-CRPS / cov | 0.3099 / 0.821 | 0.3102 / 0.831 | 0.3110 / 0.843 | 0.3171 / 0.877 |
| h=12 log-CRPS / cov | **0.4694** / 0.793 | 0.4703 / 0.816 | 0.4766 / 0.868 | 0.5891 / 0.951 |

The premise was wrong: the *flat* head already brings coverage to ~nominal at
h=12 (0.793) — the long-horizon under-dispersion is **not** horizon-growing in a
way that needs extra power, so any `power>0` just over-disperses. Lower-base +
power combos (s=0.25, p=1–2) also lose. `residual_variance_horizon_power` stays
in the code, inert (default 0.0); the flat `scale=0.5` `config_tgtlag3_var`
remains champion.

(Subset caveat: the `nunique//n_locations` stride selects different location
subsets at different `--n-locations`, so absolute subset log-CRPS is comparable
only *within* a run, not across. The power monotonicity above is within-run and
robust; the flat-head-vs-champion margin at h=3 is full-harness-authoritative.)
