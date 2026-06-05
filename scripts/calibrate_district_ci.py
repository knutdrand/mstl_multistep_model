"""Calibrate the bottom-up district prediction intervals to nominal coverage.

Bottom-up sums independently-drawn sector samples, so district intervals are too narrow
(ignores cross-sector error correlation). We learn a per-horizon multiplicative spread factor
lambda_h from the backtest district residuals and scale each district's samples around their
mean by lambda_h. lambda is chosen so the 80% interval covers 80%:

  for each cell, r = (y - mean)/(q90 - mean) if y>mean else (mean - y)/(mean - q10);
  the cell is covered after scaling by lambda iff r <= lambda, so lambda_h = quantile(r, 0.80).

Honesty: leave-one-target-period-out CV (estimate lambda on the other periods, score the held-out
one) -> reported coverage is out-of-sample. The final applied lambda uses all periods.

Writes output/arima012_bank_district_cal.nc and reports coverage before/after.
"""
from __future__ import annotations
import numpy as np, xarray as xr

IN = "output/arima012_bank_district.nc"
OUT = "output/arima012_bank_district_cal.nc"


def ratios(fc, obs, lo, hi):
    """per-cell r at a given interval (lo,hi quantile pair); covered-after-scale-lambda iff r<=lambda."""
    r = np.full(fc.shape[:3], np.nan)  # (loc, tp, h)
    L, T, H, _ = fc.shape
    for i in range(L):
        for t in range(T):
            for h in range(H):
                s = fc[i, t, h]; s = s[np.isfinite(s)]; y = obs[i, t]
                if s.size == 0 or not np.isfinite(y): continue
                mu = s.mean(); ql, qh = np.quantile(s, [lo, hi])
                if y >= mu:
                    r[i, t, h] = (y - mu) / (qh - mu + 1e-9)
                else:
                    r[i, t, h] = (mu - y) / (mu - ql + 1e-9)
    return r


def coverage(r, lam_h):
    """fraction covered given per-horizon lambda (broadcast over loc,tp)."""
    H = r.shape[2]; cov = []
    for h in range(H):
        rr = r[:, :, h][np.isfinite(r[:, :, h])]
        cov.append(np.mean(rr <= lam_h[h]))
    return np.array(cov)


def main():
    ds = xr.open_dataset(IN)
    fc = ds.forecast.values; obs = ds.observed.values
    L, T, H, N = fc.shape
    r80 = ratios(fc, obs, .10, .90)
    r50 = ratios(fc, obs, .25, .75)

    # ---- current coverage (lambda=1) ----
    base80 = coverage(r80, np.ones(H)); base50 = coverage(r50, np.ones(H))

    # ---- in-sample lambda (all periods), per horizon, calibrated to 80% ----
    lam = np.array([np.nanquantile(r80[:, :, h], 0.80) for h in range(H)])

    # ---- leave-one-period-out CV: honest out-of-sample coverage ----
    cv_cov80 = []; cv_cov50 = []
    for tp_out in range(T):
        keep = [t for t in range(T) if t != tp_out]
        lam_cv = np.array([np.nanquantile(r80[:, keep, h], 0.80) for h in range(H)])
        for h in range(H):
            rr80 = r80[:, tp_out, h][np.isfinite(r80[:, tp_out, h])]
            rr50 = r50[:, tp_out, h][np.isfinite(r50[:, tp_out, h])]
            if rr80.size: cv_cov80.append(np.mean(rr80 <= lam_cv[h]))
            if rr50.size: cv_cov50.append(np.mean(rr50 <= lam_cv[h]))

    print(f"per-horizon spread inflation lambda (calibrated to 80%): {np.round(lam,2)}")
    print(f"\n{'':16} {'cov80':>7} {'cov50':>7}")
    print(f"{'before (lam=1)':16} {base80.mean():>7.3f} {base50.mean():>7.3f}")
    print(f"{'after in-sample':16} {coverage(r80,lam).mean():>7.3f} {coverage(r50,lam).mean():>7.3f}")
    print(f"{'after CV(LOPO)':16} {np.mean(cv_cov80):>7.3f} {np.mean(cv_cov50):>7.3f}")
    print("\nper-horizon (before -> after in-sample), cov80:")
    a80 = coverage(r80, lam)
    for h in range(H):
        print(f"  h={h+1}: {base80[h]:.3f} -> {a80[h]:.3f}   (lambda={lam[h]:.2f})")

    # ---- apply lambda: scale district samples around their per-cell mean ----
    mu = np.nanmean(fc, axis=3, keepdims=True)
    fc_cal = np.clip(mu + lam[None, None, :, None] * (fc - mu), 0, None)
    out = ds.copy(); out["forecast"] = (ds.forecast.dims, fc_cal)
    out.attrs["ci_calibration"] = f"per-horizon district spread inflation lambda={list(np.round(lam,3))}"
    out.to_netcdf(OUT)
    print(f"\nwrote calibrated district forecast -> {OUT}")


if __name__ == "__main__":
    main()
