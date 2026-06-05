# Spatial neighbor features — NEGATIVE

406 sectors nested under 46 districts (every district >=3 sectors), no coordinates,
so same-district sectors are the proximity proxy. Hypothesis: a spatial signal the
per-location model misses — "is the surrounding district seeing elevated transmission
(or vector control)?" — could add information the way IRS and target lags did.

## Lever (code)
`mstl_multistep/spatial_features.py` + config `neighbor_group_col` / `neighbor_target_lags`
(default off = champion bit-for-bit). Feature: lagged **leave-one-out district mean of the
deseasonalized target D** (`nbr_lag1..k`), a spatial-autoregressive signal — the spatial
analogue of `rf_target_lags`. Forecast-window lags bridged by persistence (last observed
neighbour mean). Wired into the rf_residual fit/predict alongside the variance head + target lags.

## Result — neighbor-cases (neighbor-D)

Subset screens (district-subset, champion = config_tgtlag3_var):

| | h=3 (16 dist/142 loc) | h=12 (12 dist/108 loc) |
|---|---|---|
| champion (no nbr) | 0.3206 | 0.4703 |
| nbr_lags=1 | **0.3195** | **0.4697** |
| nbr_lags=2 | 0.3201 | 0.4698 |
| nbr_lags=3 | 0.3198 | 0.4700 |

Subset suggested a small gain (nbr_lags=1 best, −0.0011 at h=3, −0.0006 at h=12 — note
spatial helps *less* at long horizon, opposite of target lags: the recent-neighbour signal
and the persistence bridge both decay).

**Full h=3 harness (406 loc, decision-maker) — did NOT replicate:**

| config | log-CRPS | CRPS |
|---|---|---|
| champion (config_tgtlag3_var) | **0.320022** | **82.881** |
| + neighbor nbr_lags=1 (nbr1) | 0.320071 | 82.898 |

Marginally **worse** on both. The subset's +0.0011 was sample-specific noise — direct
evidence that at these magnitudes (~1e-3) the location-subset proxy is unreliable and only
the full harness decides.

## Result — neighbor-IRS (not built; redundant by construction)

Spraying is **district-coordinated, not sector-targeted**: in active district-months 91.5%
spray *all* sectors (mean fraction 0.96, median 1.0). corr(own_irs, neighbor_mean_irs) =
**0.988** overall. A neighbor-IRS feature would near-duplicate the own-IRS features the
champion already has, so it cannot add information. Skipped (no eval).

## Conclusion
Spatial neighbor features do not help on this dataset, and it is mechanistically clear *why*:
the spatial signal is already captured — shared climate is removed by deseasonalization,
spray is district-uniform (own = neighbour), and case autocorrelation is modelled per-location.
The neighbor code stays in, inert (default off). Champion unchanged: **config_tgtlag3_var
(log-CRPS 0.320022, CRPS 82.881)**.
