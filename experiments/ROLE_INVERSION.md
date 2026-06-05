# Investigation: inverting the ARIMA <-> RF roles (dynamic regression)

## Idea
Current champion is **ARIMA-mean + RF-residual** (ARIMA owns the mean dynamics and the
predictive spread; the RF nudges the mean via covariates). The proposed inversion is the
mirror -- **RF-mean + ARIMA-errors** (regression with ARIMA errors / dynamic regression):
`D_t = m_RF(covariates, lags) + eta_t`, `eta_t ~ ARIMA`, with ARIMA still owning the spread
(fit on the RF's out-of-bag residuals). Two flavours: (A) sequential, (B) joint backfitting.

## Pre-test (cheap, decisive) -- does an RF mean even match ARIMA?
Before building anything, compare the multi-step point-forecast skill of the two candidate
*mean* models on the deseasonalised log series `D` (136-location subset, 6 splits,
`scripts/pretest_rfmean.py`): ARIMA(0,1,2) vs a recursive RF (D-lags 1-3 + climate lags 1-3)
vs random walk. MAE in deseasonalised log space:

| horizon | ARIMA(0,1,2) | recursive RF | random walk | RF/ARIMA |
|---|---|---|---|---|
| 1 | **0.3490** | 0.3550 | 0.3574 | 1.017 |
| 2 | **0.4363** | 0.4455 | 0.4497 | 1.021 |
| 3 | **0.5080** | 0.5150 | 0.5347 | 1.014 |
| pooled | **0.4311** | 0.4384 | 0.4472 | 1.017 |

## Verdict: NOT worth building (dead on arrival as a champion bet)
- ARIMA(0,1,2) is the best mean at **every** horizon; the RF mean is ~2% worse.
- The RF beats random walk (0.4384 < 0.4472), so it *is* learning (climate + persistence) --
  but it cannot match ARIMA.
- The gap is present already at **h=1** (no recursion), so it is not a recursion-compounding
  artifact -- it is the fundamental fact that SES/ARIMA is the optimal mean for this
  near-random-walk series (cf. `arima011_math.pdf`), and an RF can't beat it.

Inverting would make the **weaker** model the primary mean and hope the ARIMA-error layer
compensates -- unlikely, since the current architecture already secures the best mean (ARIMA)
*and* the RF covariate correction *and* ARIMA's spread. Option B (joint backfitting) shares the
handicap, and the earlier EM on the sibling decomposition was already negative
(`EM_INVESTIGATION.md`).

Corollary: ARIMA (pure univariate persistence) beating RF-with-climate on the *mean*
re-confirms that climate adds little to the **mean** of `D` -- consistent with the ARIMA
residual `R` being near-white-noise throughout this project. The covariates' value is as a
small correction, not as a primary mean driver. No code added; champion unchanged
(`config_arima012_bank`, log-CRPS 0.31232 / CRPS 81.88).
