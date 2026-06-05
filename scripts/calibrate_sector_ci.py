"""Per-sector prediction-interval calibration (each sector's intervals adjusted to nominal coverage).

Sector coverage is ~0.80 on average but heterogeneous across sectors. We calibrate each sector with
a multiplicative spread factor = (global per-horizon factor) x (per-sector level, shrunk toward 1):

  r          = per-cell edge ratio (covered after scaling by lambda iff r <= lambda)
  lambda_h   = 0.80-quantile of r over all sectors at horizon h           (global shape)
  c_s        = 0.80-quantile of (r / lambda_h) over sector s's cells       (per-sector level)
  c_s_shrunk = 1 + n_s/(n_s+K) * (c_s - 1)                                 (shrink toward 1)
  lambda[s,h]= lambda_h * c_s_shrunk

Few cells per sector (~36) -> shrinkage K controls overfitting; chosen by leave-one-period-out CV.
Reports per-sector coverage heterogeneity + pooled log-CRPS for: uncalibrated / global / per-sector.

Writes output/arima012_bank_sector_cal.nc.
"""
from __future__ import annotations
import numpy as np, xarray as xr

IN = "output/arima012_bank.nc"
OUT = "output/arima012_bank_sector_cal.nc"
KS = [10, 25, 60]  # shrinkage constants to screen by CV


def crps(s, y, log=True):
    s = np.asarray(s, float); s = s[np.isfinite(s)]
    if s.size == 0 or not np.isfinite(y): return np.nan
    if log: s = np.log1p(np.clip(s, 0, None)); y = np.log1p(max(y, 0))
    s = np.sort(s); m = len(s); i = np.arange(m)
    return float(np.mean(np.abs(s - y)) - np.sum((2 * i - (m - 1)) * s) / (m * m))


def edge_ratio(fc, obs, lo, hi):
    L, T, H, _ = fc.shape
    r = np.full((L, T, H), np.nan)
    mu = np.full((L, T, H), np.nan)
    for i in range(L):
        for t in range(T):
            y = obs[i, t]
            for h in range(H):
                s = fc[i, t, h]; s = s[np.isfinite(s)]
                if s.size == 0 or not np.isfinite(y): continue
                m = s.mean(); ql, qh = np.quantile(s, [lo, hi]); mu[i, t, h] = m
                r[i, t, h] = (y - m) / (qh - m + 1e-9) if y >= m else (m - y) / (m - ql + 1e-9)
    return r, mu


def global_lambda(r, periods):
    H = r.shape[2]
    return np.array([np.nanquantile(r[:, periods, h], 0.80) for h in range(H)])


def sector_c(r, lam_h, periods, K):
    L, T, H = r.shape
    c = np.ones(L)
    for s in range(L):
        q = (r[s][:, :][periods] / lam_h[None, :]).ravel()  # (periods,H) ratio
        q = q[np.isfinite(q)]
        if q.size >= 4:
            c_raw = np.quantile(q, 0.80); n = q.size
            c[s] = 1 + (n / (n + K)) * (c_raw - 1)
    return c


def sector_cov_spread(r, lam_sh):
    """per-sector cov80 (covered iff r<=lambda[s,h]); return mean, std, mean|cov-0.8|."""
    L, T, H = r.shape
    covs = []
    for s in range(L):
        rr = r[s]; lam = lam_sh[s]
        cov = []
        for t in range(T):
            for h in range(H):
                if np.isfinite(rr[t, h]): cov.append(rr[t, h] <= lam[h])
        if cov: covs.append(np.mean(cov))
    covs = np.array(covs)
    return covs.mean(), covs.std(), np.mean(np.abs(covs - 0.80))


def main():
    ds = xr.open_dataset(IN); fc = ds.forecast.values; obs = ds.observed.values
    L, T, H, N = fc.shape
    # Calibrate in log1p space (where log-CRPS lives). Coverage is transform-invariant, so it
    # stays calibrated, but scaling log-samples around their mean avoids the count-space log-CRPS hit.
    flog = np.log1p(np.clip(fc, 0, None)); ylog = np.log1p(np.clip(obs, 0, None))
    r80, mu = edge_ratio(flog, ylog, .10, .90)   # mu = log-space per-cell mean
    allp = list(range(T))

    # --- per-sector coverage heterogeneity, uncalibrated ---
    lam_one = np.ones((L, H))
    m0, sd0, mad0 = sector_cov_spread(r80, lam_one)
    print(f"per-sector cov80 (uncalibrated): mean={m0:.3f} std={sd0:.3f} mean|cov-0.8|={mad0:.3f}")

    # --- global-only calibration ---
    lamH = global_lambda(r80, allp)
    lam_glob = np.tile(lamH, (L, 1))
    mg, sdg, madg = sector_cov_spread(r80, lam_glob)
    print(f"per-sector cov80 (global lambda {np.round(lamH,2)}): mean={mg:.3f} std={sdg:.3f} mean|cov-0.8|={madg:.3f}")

    # --- per-sector with shrinkage: pick K by leave-one-period-out CV (per-sector coverage MAD) ---
    print("\nCV (leave-one-period-out) per-sector mean|cov-0.8| by shrinkage K:")
    best = None
    for K in [1e9] + KS:   # 1e9 ~ global-only
        covbysec = {s: [] for s in range(L)}
        for tout in range(T):
            keep = [t for t in range(T) if t != tout]
            lh = global_lambda(r80, keep); c = sector_c(r80, lh, keep, K)
            for s in range(L):
                for h in range(H):
                    rv = r80[s, tout, h]
                    if np.isfinite(rv): covbysec[s].append(rv <= lh[h] * c[s])
        per = np.array([np.mean(v) for v in covbysec.values() if v])
        mad = np.mean(np.abs(per - 0.80))
        tag = "global" if K > 1e8 else f"K={K}"
        print(f"  {tag:8s}: cov80={per.mean():.3f}  mean|cov-0.8|={mad:.3f}")
        if best is None or mad < best[1]: best = (K, mad)
    Kbest = best[0]
    print(f"-> best: {'global' if Kbest>1e8 else f'K={Kbest}'}")

    # --- apply best (all periods) ---
    c = sector_c(r80, lamH, allp, Kbest) if Kbest < 1e8 else np.ones(L)
    lam_sh = lamH[None, :] * c[:, None]
    ms, sds, mads = sector_cov_spread(r80, lam_sh)
    print(f"\nper-sector cov80 (per-sector, K={Kbest if Kbest<1e8 else 'inf'}): mean={ms:.3f} std={sds:.3f} mean|cov-0.8|={mads:.3f}")

    # log-CRPS / CRPS before vs after (pooled)
    def score(lam_sh):
        lc, cr = [], []
        for s in range(L):
            for t in range(T):
                y = obs[s, t]; yl = ylog[s, t]
                if not np.isfinite(y): continue
                for h in range(H):
                    sl = flog[s, t, h]; sl = sl[np.isfinite(sl)]
                    if sl.size == 0: continue
                    m = sl.mean(); scl = m + lam_sh[s, h] * (sl - m)   # scale in log space
                    lc.append(crps(scl, yl, log=False))                # already log space
                    cr.append(crps(np.expm1(scl), y, log=False))       # back to counts for CRPS
        return np.nanmean(lc), np.nanmean(cr)
    lc0, cr0 = score(np.ones((L, H))); lcg, crg = score(lam_glob); lcs, crs = score(lam_sh)
    print(f"\npooled scores  uncal: log-CRPS={lc0:.4f} CRPS={cr0:.2f}")
    print(f"               global: log-CRPS={lcg:.4f} CRPS={crg:.2f}")
    print(f"     per-sector (in-sample lambda): log-CRPS={lcs:.4f} CRPS={crs:.2f}")

    # honest out-of-sample log-CRPS: lambda from training periods, scored on held-out (LOPO)
    cv_cal, cv_unc = [], []
    for tout in range(T):
        keep = [t for t in range(T) if t != tout]
        lh = global_lambda(r80, keep); c = sector_c(r80, lh, keep, Kbest)
        for s in range(L):
            y = obs[s, tout]; yl = ylog[s, tout]
            if not np.isfinite(y): continue
            for h in range(H):
                sl = flog[s, tout, h]; sl = sl[np.isfinite(sl)]
                if sl.size == 0: continue
                m = sl.mean()
                cv_cal.append(crps(m + lh[h] * c[s] * (sl - m), yl, log=False))
                cv_unc.append(crps(sl, yl, log=False))
    print(f"     per-sector (CV out-of-sample): log-CRPS={np.nanmean(cv_cal):.4f}  "
          f"(uncal same cells={np.nanmean(cv_unc):.4f})")

    # write calibrated sector nc (scale in log space, expm1 back)
    mu_all = np.nanmean(flog, 3, keepdims=True)
    fc_cal = np.clip(np.expm1(mu_all + lam_sh[:, None, :, None] * (flog - mu_all)), 0, None)
    out = ds.copy(); out["forecast"] = (ds.forecast.dims, fc_cal)
    out.attrs["ci_calibration"] = f"per-sector log-space spread: lambda_h={list(np.round(lamH,3))} x shrunk c_s (K={Kbest})"
    out.to_netcdf(OUT)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
