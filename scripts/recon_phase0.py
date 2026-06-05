"""Phase-0 diagnostic for hierarchical reconciliation.

From the champion sector-level backtest, measure the two things that decide whether optimal
reconciliation (MinT) can beat naive bottom-up:
  (1) relative predictability by level -- is the district/national level more predictable
      (lower normalised RMSE) than the sector level? (the denoising premise)
  (2) sector base-error correlation -- within-district vs cross-district. MinT only helps
      beyond bottom-up when base errors are correlated.

Uses point error e = mean(samples) - observed, per (sector, target, horizon). No model runs.

Usage: uv run python scripts/recon_phase0.py
"""
from __future__ import annotations
import numpy as np, pandas as pd, xarray as xr

NC = "output/arima012_bank.nc"
DATASET = "/Users/knutdr/Data/CH/chap_data_level5_irs_allocated_monthly.csv"


def nrmse(err, obs):
    err = np.asarray(err, float); obs = np.asarray(obs, float)
    m = np.isfinite(err) & np.isfinite(obs)
    if m.sum() == 0: return np.nan
    return float(np.sqrt(np.mean(err[m] ** 2)) / (np.mean(np.abs(obs[m])) + 1e-9))


def main():
    d = xr.open_dataset(NC)
    locs = [str(x) for x in d.location.values]
    fc = d.forecast.mean("sample").values            # (loc, tp, h)  point forecast
    obs = d.observed.values                          # (loc, tp)
    L, T, H = fc.shape
    err = fc - obs[:, :, None]                        # (loc, tp, h)

    # sector -> district map
    df = pd.read_csv(DATASET); df["location"] = df["location"].astype(str)
    l2d = dict(zip(df.location, df.district.astype(str)))
    dist = np.array([l2d.get(l, "NA") for l in locs])

    # ---------- (1) relative predictability by level ----------
    print("=== (1) normalised RMSE by hierarchy level (lower = more predictable) ===")
    # sector: pooled + median per-series
    per_series = [nrmse(err[i], np.repeat(obs[i][:, None], H, 1)) for i in range(L)]
    print(f"  sector   nRMSE: pooled={nrmse(err, np.repeat(obs[:,:,None],H,2)):.3f}  "
          f"median-per-series={np.nanmedian(per_series):.3f}  (n={L})")
    # district: sum sectors within district
    dser = []
    for dd in np.unique(dist):
        idx = np.where(dist == dd)[0]
        fcd = np.nansum(np.where(np.isfinite(fc[idx]), fc[idx], np.nan), 0)   # (tp,h) -- keep NaN where all-NaN
        cnt = np.sum(np.isfinite(fc[idx]), 0)
        fcd = np.where(cnt > 0, fcd, np.nan)
        obsd = np.nansum(obs[idx], 0)
        ed = fcd - obsd[:, None]
        dser.append(nrmse(ed, np.repeat(obsd[:, None], H, 1)))
    print(f"  district nRMSE: median-per-series={np.nanmedian(dser):.3f}  (n={len(dser)})")
    # national
    fcn = np.where(np.sum(np.isfinite(fc), 0) > 0, np.nansum(fc, 0), np.nan)
    obsn = np.nansum(obs, 0)
    print(f"  national nRMSE: {nrmse(fcn - obsn[:, None], np.repeat(obsn[:, None], H, 1)):.3f}")
    # per-horizon sector vs district
    print("  per-horizon (sector pooled / district median):")
    for h in range(H):
        s = nrmse(err[:, :, h], obs)
        dv = []
        for dd in np.unique(dist):
            idx = np.where(dist == dd)[0]
            cnt = np.sum(np.isfinite(fc[idx, :, h]), 0)
            fcd = np.where(cnt > 0, np.nansum(np.where(np.isfinite(fc[idx,:,h]), fc[idx,:,h], np.nan), 0), np.nan)
            obsd = np.nansum(obs[idx], 0)
            dv.append(nrmse(fcd - obsd, obsd))
        print(f"    h={h+1}: sector={s:.3f}  district={np.nanmedian(dv):.3f}")

    # ---------- (2) sector base-error correlation ----------
    print("\n=== (2) sector base-error correlation (MinT headroom) ===")
    E = err.reshape(L, T * H).T                       # (obs=T*H, sectors=L)
    edf = pd.DataFrame(E, columns=locs)
    C = edf.corr(min_periods=10).values               # (L,L) pairwise
    iu = np.triu_indices(L, 1)
    same = dist[iu[0]] == dist[iu[1]]
    cc = C[iu]
    fin = np.isfinite(cc)
    print(f"  mean pairwise corr  overall={np.nanmean(cc):.3f}")
    print(f"                      within-district={np.nanmean(cc[same & fin]):.3f}  "
          f"cross-district={np.nanmean(cc[~same & fin]):.3f}")
    # common-shock check: how much of error variance is a single shared (cross-sector) factor?
    Zc = E - np.nanmean(E, 0, keepdims=True)
    Zc = np.where(np.isfinite(Zc), Zc, 0.0)
    # per-row (per target,horizon) cross-sector mean error -> if large, a common temporal shock
    row_mean = np.nanmean(np.where(np.isfinite(E), E, np.nan), 1)
    tot_var = np.nanvar(E)
    shared_var = np.nanvar(row_mean)
    print(f"  variance share of a common cross-sector shock (per target,horizon): "
          f"{shared_var/ (tot_var+1e-9):.2f}")
    print("\nInterpretation hints:")
    print("  - district nRMSE << sector nRMSE  => aggregation denoises (reconciliation can help sectors)")
    print("  - within>cross corr, both moderate => hierarchical error structure for MinT to exploit")
    print("  - high common-shock share         => strong shared signal; top-down/MinT promising")


if __name__ == "__main__":
    main()
