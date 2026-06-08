"""Interval calibration for the model's backtest forecasts (the correct integration point).

Honest interval calibration needs *out-of-sample* forecast residuals, which exist only in the
backtest -- not inside a single ``predict()`` (an attempt to derive a per-sector factor from
in-sample out-of-bag one-step residuals was biased: it understates the true h-step forecast
error and over-narrows, pushing sector coverage 0.83 -> 0.76). So the calibrator is **fit on the
backtest and applied to its forecasts** (standard conformal/split-calibration deployment).

Two operations, both scaling the predictive samples around their per-cell mean (so all interval
levels scale together):

- ``calibrate_sector_intervals`` -- per-location factor lambda[loc] = (global per-horizon factor)
  x (per-location level, shrunk toward 1), in **log1p space** (where log-CRPS lives; coverage is
  transform-invariant) so it fixes per-sector coverage heterogeneity without a log-CRPS regression.
- ``calibrate_district_intervals`` -- per-horizon spread inflation lambda_h on a bottom-up
  (aggregate-eval'd) district .nc, in count space, to undo the under-dispersion from summing
  independently-drawn sector samples.

Both target the 80% central interval (lambda = 0.80-quantile of the per-cell edge ratio).
"""
from __future__ import annotations

import numpy as np
import xarray as xr

Z = 1.2815515594  # norm.ppf(0.90): half-width of the 80% central interval in sd units (unused; kept for ref)


def _edge_ratios(fc: np.ndarray, obs: np.ndarray, lo=0.10, hi=0.90):
    """Per-cell r (covered after scaling-by-lambda iff r<=lambda) for the (lo,hi) central interval."""
    L, T, H, _ = fc.shape
    r = np.full((L, T, H), np.nan)
    for i in range(L):
        for t in range(T):
            y = obs[i, t]
            for h in range(H):
                s = fc[i, t, h]; s = s[np.isfinite(s)]
                if s.size == 0 or not np.isfinite(y):
                    continue
                m = s.mean(); ql, qh = np.quantile(s, [lo, hi])
                r[i, t, h] = (y - m) / (qh - m + 1e-9) if y >= m else (m - y) / (m - ql + 1e-9)
    return r


def calibrate_district_intervals(ds: xr.Dataset) -> tuple[xr.Dataset, np.ndarray]:
    """Per-horizon spread inflation so bottom-up district 80% intervals cover 80% (count space)."""
    fc = ds.forecast.values; obs = ds.observed.values
    r = _edge_ratios(fc, obs)
    H = fc.shape[2]
    lam = np.array([np.nanquantile(r[:, :, h], 0.80) for h in range(H)])
    mu = np.nanmean(fc, axis=3, keepdims=True)
    fc_cal = np.clip(mu + lam[None, None, :, None] * (fc - mu), 0, None)
    out = ds.copy(); out["forecast"] = (ds.forecast.dims, fc_cal)
    out.attrs["ci_calibration"] = f"district per-horizon spread inflation lambda={list(np.round(lam, 3))}"
    return out, lam


def calibrate_sector_intervals(ds: xr.Dataset, shrink_k: float = 10.0) -> tuple[xr.Dataset, dict]:
    """Per-location log-space spread factor so each location's 80% interval covers ~80%.

    lambda[loc, h] = lambda_h^global * c_loc, with c_loc the per-location level (0.80-quantile of
    the edge ratio normalised by the global factor) shrunk toward 1 by ``shrink_k``. Calibrated and
    applied in log1p space.
    """
    fc = ds.forecast.values; obs = ds.observed.values
    L, T, H, _ = fc.shape
    flog = np.log1p(np.clip(fc, 0, None)); ylog = np.log1p(np.clip(obs, 0, None))
    r = _edge_ratios(flog, ylog)
    lamH = np.array([np.nanquantile(r[:, :, h], 0.80) for h in range(H)])
    c = np.ones(L)
    for s in range(L):
        q = (r[s] / lamH[None, :]).ravel(); q = q[np.isfinite(q)]
        if q.size >= 4:
            n = q.size
            c[s] = 1.0 + (n / (n + shrink_k)) * (float(np.quantile(q, 0.80)) - 1.0)
    lam_sh = lamH[None, :] * c[:, None]                 # (L, H)
    mu = np.nanmean(flog, axis=3, keepdims=True)
    fc_cal = np.clip(np.expm1(mu + lam_sh[:, None, :, None] * (flog - mu)), 0, None)
    out = ds.copy(); out["forecast"] = (ds.forecast.dims, fc_cal)
    out.attrs["ci_calibration"] = f"sector log-space per-location spread (lambda_h={list(np.round(lamH,3))}, K={shrink_k})"
    locs = [str(x) for x in ds.location.values]
    return out, {locs[s]: lam_sh[s] for s in range(L)}
