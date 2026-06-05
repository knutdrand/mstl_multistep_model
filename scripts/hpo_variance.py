"""Subset screen for the location-scale coordinate-descent variance head.

Champion draws a symmetric ARIMA Gaussian and adds the RF correction as a point with
no uncertainty. This screens widening the spread to sqrt(sigma_ARIMA^2 + scale*v(x))
with v(x) from RF inter-tree variance ("tree") or a GBM on squared OOB residuals
("model"), optionally with IRLS reweighting. Reports log-CRPS + coverage(10-90) +
frac>q90/q99 (the tail axis this targets). Confirm winners on the full frozen harness.

Usage: uv run python scripts/hpo_variance.py [--n-locations 100] [--n-splits 8]
"""
from __future__ import annotations
import argparse, time
import numpy as np, pandas as pd
from mstl_multistep.pipeline import build_chap_model
from mstl_multistep.run_config import load_model_configuration
from mstl_multistep.io_utils import detect_frequency, period_to_timestamp

DATASET = "/Users/knutdr/Data/CH/chap_data_level5_irs_allocated_monthly.csv"
BASE = "config_tgtlag3.yaml"   # champion: IRS + per-cov lags + rf_target_lags=3
TARGET = "disease_cases"


def crps(s, y):
    s = np.sort(np.asarray(s, float)); m = len(s)
    if m == 0 or not np.isfinite(y): return np.nan
    i = np.arange(m)
    return float(np.mean(np.abs(s - y)) - np.sum((2 * i - (m - 1)) * s) / (m * m))


def rolling(times, n_splits, n_periods=3, stride=1):
    T = len(times)
    for k in range(n_splits):
        o = T - n_periods - k * stride
        if o < 24: break
        yield times[:o], times[o:o + n_periods]


def score(df, cfg, cov, splits):
    lcs, covs, q90, q99 = [], [], [], []
    for tr, fu in splits:
        h = df[df.time_period.isin(tr)]; f = df[df.time_period.isin(fu)]
        if f.empty: continue
        m = build_chap_model(cfg, cov); m.fit(h); p = m.predict(h, f)
        sc = [c for c in p.columns if c.startswith("sample_")]
        merged = p.merge(f[["time_period", "location", TARGET]].astype({"location": str}),
                         on=["time_period", "location"], how="inner")
        for _, r in merged.iterrows():
            y = pd.to_numeric(r[TARGET], errors="coerce")
            if not np.isfinite(y): continue
            s = r[sc].to_numpy(float)
            lcs.append(crps(np.log1p(np.clip(s, 0, None)), float(np.log1p(max(y, 0)))))
            ql, qh = np.quantile(s, [.1, .9]); covs.append(float(ql <= y <= qh))
            q90.append(float(y > np.quantile(s, .9)))
            q99.append(float(y > np.quantile(s, .99)))
    agg = lambda a: float(np.nanmean(a)) if a else np.nan
    return agg(lcs), agg(covs), agg(q90), agg(q99)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-locations", type=int, default=100)
    ap.add_argument("--n-splits", type=int, default=8)
    ap.add_argument("--n-periods", type=int, default=3)
    args = ap.parse_args()
    df = pd.read_csv(DATASET); df["location"] = df["location"].astype(str)
    step = max(1, df["location"].nunique() // args.n_locations)
    df = df[df["location"].isin(sorted(df["location"].unique())[::step])].copy()
    freq = detect_frequency(df)
    times = sorted(df["time_period"].unique(), key=lambda p: period_to_timestamp(p, freq))
    splits = list(rolling(times, args.n_splits, n_periods=args.n_periods))
    base = load_model_configuration(BASE); cov = base.additional_continuous_covariates
    print(f"locations={df.location.nunique()} splits={len(splits)}")

    configs = [("champion (none)", {})]
    for sc in (0.5, 1.0, 2.0):
        configs.append((f"tree  scale={sc}", {"residual_variance": "tree", "residual_variance_scale": sc}))
    for sc in (1.0, 2.0):
        configs.append((f"model scale={sc}", {"residual_variance": "model", "residual_variance_scale": sc}))

    res = []
    for label, over in configs:
        cfg = base.user_option_values.model_copy(update=over)
        t0 = time.time()
        lc, cv, f90, f99 = score(df, cfg, cov, splits)
        res.append((lc, cv, f90, f99, label))
        print(f"  {label:22s} log_crps={lc:.4f}  cov={cv:.3f}  f>q90={f90:.3f}  f>q99={f99:.3f}  ({time.time()-t0:.0f}s)")
    res.sort(key=lambda r: (np.inf if np.isnan(r[0]) else r[0]))
    print("\n=== ranked by log-CRPS (subset proxy) ===")
    for lc, cv, f90, f99, label in res:
        print(f"{lc:.4f}  cov={cv:.3f}  f>q90={f90:.3f}  f>q99={f99:.3f}  {label}")


if __name__ == "__main__":
    main()
