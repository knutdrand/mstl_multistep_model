"""Phase 2: hierarchical reconciliation (MinT) of the champion forecasts, evaluated at all levels.

Loads independent base forecasts at sector / district / national, builds the summing matrix S,
estimates the base-error covariance W (OLS=I, WLS=diag, or MinT-shrink), and reconciles:
    G = (S' W^-1 S)^-1 S' W^-1 ;  bottom_recon = G y_base ;  all = S bottom_recon
Reconciliation is on the MEAN in COUNT space (where the hierarchy sums); the sector sample spread
is preserved by shifting samples by (reconciled_mean - base_mean) -- a first-cut probabilistic
reconciliation. Scores log-CRPS + CRPS at sector and district levels: base/bottom-up vs MinT.

Usage: uv run python scripts/reconcile_mint.py [--method mint_shrink|wls|ols]
"""
from __future__ import annotations
import argparse
import numpy as np, pandas as pd, xarray as xr

SECTOR = "output/arima012_bank.nc"
DISTRICT = "output/district_base.nc"
NATIONAL = "output/national_base.nc"
DATASET = "/Users/knutdr/Data/CH/chap_data_level5_irs_allocated_monthly.csv"


def crps(s, y, log=False):
    s = np.asarray(s, float); s = s[np.isfinite(s)]
    if s.size == 0 or not np.isfinite(y): return np.nan
    if log: s = np.log1p(np.clip(s, 0, None)); y = np.log1p(max(y, 0))
    s = np.sort(s); m = len(s); i = np.arange(m)
    return float(np.mean(np.abs(s - y)) - np.sum((2 * i - (m - 1)) * s) / (m * m))


def shrink_cov(E):
    """Schafer-Strimmer shrinkage of sample covariance toward its diagonal (MinT-shrink)."""
    E = E - np.nanmean(E, 1, keepdims=True)
    E = np.where(np.isfinite(E), E, 0.0)
    n = E.shape[1]
    S = (E @ E.T) / max(n - 1, 1)
    d = np.diag(S).copy(); d[d <= 0] = np.mean(d[d > 0]) if np.any(d > 0) else 1.0
    R = S / np.sqrt(np.outer(d, d))
    # lambda: shrink correlations to 0
    var_r = (R ** 2).sum() - np.trace(R ** 2)
    lam = 0.5 if var_r == 0 else np.clip((var_r) / ((R ** 2).sum()), 0.01, 0.99)
    Sd = np.diag(d)
    return (1 - lam) * S + lam * Sd


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--method", default="mint_shrink")
    a = ap.parse_args()
    sec = xr.open_dataset(SECTOR); dis = xr.open_dataset(DISTRICT); nat = xr.open_dataset(NATIONAL)
    secs = [str(x) for x in sec.location.values]
    dists = [str(x) for x in dis.location.values]
    df = pd.read_csv(DATASET); df["location"] = df["location"].astype(str)
    l2d = dict(zip(df.location, df.district.astype(str)))
    sec_dist = [l2d[s] for s in secs]

    # align time periods across levels
    tp = [str(t) for t in sec.time_period.values]
    H = sec.sizes["horizon_distance"]; nS = len(secs)

    # base means/observed per node, per (tp,h)
    sec_mean = sec.forecast.mean("sample").values          # (nS, T, H)
    dis_mean = dis.forecast.mean("sample").sel(time_period=sec.time_period).values  # (nD,T,H)
    nat_mean = nat.forecast.mean("sample").sel(time_period=sec.time_period).values  # (1,T,H)
    sec_obs = sec.observed.values; dis_obs = dis.observed.sel(time_period=sec.time_period).values
    nat_obs = nat.observed.sel(time_period=sec.time_period).values

    # node order: [national, districts, sectors]
    di = {d: i for i, d in enumerate(dists)}
    # summing matrix S: (1+nD+nS) x nS
    S = np.zeros((1 + len(dists) + nS, nS))
    S[0, :] = 1.0
    for j, d in enumerate(sec_dist):
        S[1 + di[d], j] = 1.0
    S[1 + len(dists):, :] = np.eye(nS)

    # base-error matrix E: (nodes, cells) for W; cells = (tp,h) with all levels finite
    base_mean_nodes, obs_nodes = [], []
    base_mean_nodes.append(nat_mean.reshape(1, -1))       # national row
    base_mean_nodes.append(dis_mean.reshape(len(dists), -1))
    base_mean_nodes.append(sec_mean.reshape(nS, -1))
    Ybase = np.vstack(base_mean_nodes)                    # (nodes, T*H)
    obs_nodes = np.vstack([nat_obs.reshape(1, -1).repeat(H, 0).reshape(1, -1) if False else
                           np.repeat(nat_obs, H).reshape(1, -1),
                           np.repeat(dis_obs, H).reshape(len(dists), -1),
                           np.repeat(sec_obs, H).reshape(nS, -1)])
    E = Ybase - obs_nodes
    # interior cells = those where the forecast exists at every node (drops ragged backtest
    # edges). Scattered missing *observeds* within a cell are left as NaN and handled by the
    # nan-aware covariance below -- we must NOT require all 406 sector observeds present.
    valid = np.isfinite(Ybase).all(0)
    Ev = E[:, valid]

    # W and G
    nN = S.shape[0]
    if a.method == "ols":
        Winv = np.eye(nN)
    elif a.method == "wls":
        v = np.nanvar(Ev, 1); v[v <= 0] = np.nanmean(v[v > 0]); Winv = np.diag(1.0 / v)
    else:
        W = shrink_cov(Ev); Winv = np.linalg.pinv(W)
    G = np.linalg.solve(S.T @ Winv @ S + 1e-6 * np.eye(nS), S.T @ Winv)   # (nS, nodes)
    print(f"method={a.method}  nodes={nN}  bottom={nS}  cells={valid.sum()}")

    # reconcile each cell's mean; build reconciled sector samples by mean-shift
    sec_samp = sec.forecast.values                         # (nS, T, H, n)
    rng_idx = 0
    rows = []
    for ti in range(len(tp)):
        for h in range(H):
            yb = Ybase[:, ti * H + h]
            if not np.isfinite(yb).all(): continue
            b_recon = G @ yb                                # (nS,) reconciled sector means
            base_sec = sec_mean[:, ti, h]
            shift = b_recon - base_sec
            for si in range(nS):
                y = sec_obs[si, ti]
                if not np.isfinite(y): continue
                bs = sec_samp[si, ti, h]
                ms = np.clip(bs + shift[si], 0, None)
                rows.append((secs[si], sec_dist[si], ti, h, y,
                             crps(bs, y, log=True), crps(ms, y, log=True),
                             crps(bs, y), crps(ms, y)))
    R = pd.DataFrame(rows, columns=["sector", "district", "ti", "h", "y",
                                    "base_lc", "mint_lc", "base_crps", "mint_crps"])
    print("\n=== SECTOR level (champion/base vs MinT) ===")
    print(f"  log-CRPS  base={R.base_lc.mean():.4f}  MinT={R.mint_lc.mean():.4f}  "
          f"delta={R.mint_lc.mean()-R.base_lc.mean():+.4f}")
    print(f"  CRPS      base={R.base_crps.mean():.2f}  MinT={R.mint_crps.mean():.2f}  "
          f"delta={R.mint_crps.mean()-R.base_crps.mean():+.2f}")

    # district level: bottom-up sum of base sector samples vs sum of MinT sector samples vs independent
    print("\n=== DISTRICT level (bottom-up vs MinT vs independent base) ===")
    bu, mt, ind = [], [], []
    for ti in range(len(tp)):
        for h in range(H):
            yb = Ybase[:, ti * H + h]
            if not np.isfinite(yb).all(): continue
            b_recon = G @ yb; shift = b_recon - sec_mean[:, ti, h]
            for dd in dists:
                idx = [i for i, s in enumerate(sec_dist) if s == dd]
                yo = dis_obs[di[dd], ti]
                if not np.isfinite(yo): continue
                bu_s = np.nansum(sec_samp[idx, ti, h, :], 0)
                mt_s = np.nansum(np.clip(sec_samp[idx, ti, h, :] + shift[idx][:, None], 0, None), 0)
                bu.append(crps(bu_s, yo, log=True)); mt.append(crps(mt_s, yo, log=True))
                ind.append(crps(dis.forecast.values[di[dd], ti, h], yo, log=True))
    print(f"  log-CRPS  bottom-up={np.nanmean(bu):.4f}  MinT={np.nanmean(mt):.4f}  "
          f"independent={np.nanmean(ind):.4f}")


if __name__ == "__main__":
    main()
