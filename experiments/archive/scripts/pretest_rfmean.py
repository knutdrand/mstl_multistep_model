"""Pre-test for the ARIMA<->RF role inversion (dynamic regression).

Decision question: can a recursive RF *mean* forecast the deseasonalised D as well as the
champion's shared ARIMA(0,1,2)? The series is near-random-walk, and RFs extrapolate poorly,
so this could be dead on arrival. We compare multi-step (h=1..3) point-forecast error on D:
  (a) ARIMA(0,1,2) mean      (b) recursive RF mean (D-lags + climate lags)   (c) random walk
RMSE/MAE in deseasonalised log space, per horizon. If (b) is clearly worse than (a), stop.

Usage: uv run python scripts/pretest_rfmean.py [--n-locations 120] [--n-splits 6]
"""
from __future__ import annotations
import argparse
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestRegressor
from statsforecast.models import ARIMA
from mstl_multistep import decomposition as dec
from mstl_multistep.io_utils import detect_frequency, period_to_timestamp

DATASET = "/Users/knutdr/Data/CH/chap_data_level5_irs_allocated_monthly.csv"
TARGET = "disease_cases"
CLIM = ["rainfall_era5", "mean_temperature", "relative_humidity"]
H = 3
NLAG_D = 3
CLIM_LAGS = [1, 2, 3]


def deseason_panel(hist):
    freq = detect_frequency(hist)
    des, decomps = dec.decompose_panel(hist, freq, TARGET, [12], True)
    des["_ts"] = des["time_period"].apply(lambda p: period_to_timestamp(p, freq))
    return des, decomps, freq


def build_train(des, hist, freq):
    """Pooled rows: target D_t, features [D_{t-1..3}, clim lags]. Raw climate (pretest proxy)."""
    h = hist.copy(); h["_ts"] = h["time_period"].apply(lambda p: period_to_timestamp(p, freq))
    rows = []
    for loc, g in des.groupby("location", sort=False):
        g = g.sort_values("_ts"); D = g[TARGET].to_numpy(float)
        hc = h[h.location == loc].sort_values("_ts")
        clim = {c: pd.to_numeric(hc[c], errors="coerce").to_numpy(float) for c in CLIM}
        n = len(D)
        for t in range(n):
            feat = []
            ok = True
            for k in range(1, NLAG_D + 1):
                feat.append(D[t - k] if t - k >= 0 else np.nan)
            for c in CLIM:
                for k in CLIM_LAGS:
                    feat.append(clim[c][t - k] if t - k >= 0 else np.nan)
            rows.append((D[t], feat))
    X = np.array([r[1] for r in rows], float); y = np.array([r[0] for r in rows], float)
    m = ~(np.isnan(X).any(1) | np.isnan(y))
    return X[m], y[m]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-locations", type=int, default=120); ap.add_argument("--n-splits", type=int, default=6)
    a = ap.parse_args()
    df = pd.read_csv(DATASET); df["location"] = df["location"].astype(str)
    step = max(1, df.location.nunique() // a.n_locations)
    df = df[df.location.isin(sorted(df.location.unique())[::step])].copy()
    freq = detect_frequency(df)
    times = sorted(df.time_period.unique(), key=lambda p: period_to_timestamp(p, freq))

    err = {m: {h: [] for h in range(1, H + 1)} for m in ("arima", "rf", "rw")}
    for k in range(a.n_splits):
        o = len(times) - H - k
        if o < 24: break
        tr, fu = times[:o], times[o:o + H]
        hist = df[df.time_period.isin(tr)]; fut = df[df.time_period.isin(fu)]
        des, decomps, _ = deseason_panel(hist)
        X, y = build_train(des, hist, freq)
        rf = RandomForestRegressor(n_estimators=200, min_samples_leaf=3, max_features="sqrt",
                                   random_state=42, n_jobs=-1).fit(X, y)
        hc_all = hist.copy(); hc_all["_ts"] = hc_all.time_period.apply(lambda p: period_to_timestamp(p, freq))
        fc_all = fut.copy(); fc_all["_ts"] = fc_all.time_period.apply(lambda p: period_to_timestamp(p, freq))
        for loc, g in des.groupby("location", sort=False):
            g = g.sort_values("_ts"); D = g[TARGET].to_numpy(float)
            if np.isfinite(D).sum() < 24: continue
            decomp = decomps[loc]
            # actual future D = log1p(y_fut) - seasonal_naive (from history only) -> leak-free
            fg = fc_all[fc_all.location == loc].sort_values("_ts")
            if len(fg) < H: continue
            yf = pd.to_numeric(fg[TARGET], errors="coerce").to_numpy(float)
            S = dec.extrapolate_seasonal(decomp, [12], H)
            Dtrue = np.log1p(np.clip(yf, 0, None)) - S
            # (a) ARIMA(0,1,2) mean
            clean = pd.Series(D).interpolate(limit_direction="both").ffill().bfill().to_numpy()
            try: mean_h = np.asarray(ARIMA(order=(0,1,2), season_length=1).forecast(y=clean, h=H)["mean"], float)
            except Exception: mean_h = np.full(H, clean[-1])
            # (c) random walk
            rw = np.full(H, clean[-1])
            # (b) recursive RF mean
            climf = {c: pd.to_numeric(hc_all[hc_all.location==loc].sort_values("_ts")[c], errors="coerce").to_numpy(float) for c in CLIM}
            climfut = {c: pd.to_numeric(fg[c], errors="coerce").to_numpy(float) for c in CLIM}
            dwin = list(clean[-NLAG_D:])
            rfp = []
            for hh in range(H):
                feat = [dwin[-k] for k in range(1, NLAG_D + 1)]
                for c in CLIM:
                    series = np.concatenate([climf[c], climfut[c]])
                    idx = len(climf[c]) + hh
                    for kk in CLIM_LAGS:
                        feat.append(series[idx - kk] if idx - kk >= 0 else np.nan)
                p = rf.predict(np.nan_to_num(np.array(feat, float))[None, :])[0]
                rfp.append(p); dwin.append(p)
            rfp = np.array(rfp)
            for hh in range(H):
                if not np.isfinite(Dtrue[hh]): continue
                err["arima"][hh+1].append(abs(mean_h[hh]-Dtrue[hh]))
                err["rf"][hh+1].append(abs(rfp[hh]-Dtrue[hh]))
                err["rw"][hh+1].append(abs(rw[hh]-Dtrue[hh]))
    print(f"locations={df.location.nunique()} splits<= {a.n_splits}  (MAE on deseasonalised log D)")
    print(f"{'horizon':>8} | {'ARIMA(0,1,2)':>13} | {'recursive RF':>13} | {'random walk':>12}")
    for hh in range(1, H+1):
        ma=np.mean(err['arima'][hh]); mr=np.mean(err['rf'][hh]); mw=np.mean(err['rw'][hh])
        print(f"{hh:>8} | {ma:>13.4f} | {mr:>13.4f} | {mw:>12.4f}   RF/ARIMA={mr/ma:.3f}")
    allp=lambda m: np.mean([v for hh in range(1,H+1) for v in err[m][hh]])
    print(f"\npooled MAE  ARIMA={allp('arima'):.4f}  RF={allp('rf'):.4f}  RW={allp('rw'):.4f}")


if __name__ == "__main__":
    main()
