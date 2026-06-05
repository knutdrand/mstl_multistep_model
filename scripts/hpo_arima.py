"""Subset screen: shared fixed ARIMA order vs per-location AutoARIMA (champion).

Overrides arima_order on the champion (config_tgtlag3_var). Tests the data's modal order and a
couple of neighbours, with/without drift. Reports log-CRPS + CRPS + coverage. Full harness decides.

Usage: uv run python scripts/hpo_arima.py [--n-locations 120] [--n-splits 6]
"""
from __future__ import annotations
import argparse, time
import numpy as np, pandas as pd
from mstl_multistep.pipeline import build_chap_model
from mstl_multistep.run_config import load_model_configuration
from mstl_multistep.io_utils import detect_frequency, period_to_timestamp

DATASET = "/Users/knutdr/Data/CH/chap_data_level5_irs_allocated_monthly.csv"
BASE = "config_tgtlag3_var.yaml"
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
    lcs, crs, covs = [], [], []
    for tr, fu in splits:
        h = df[df.time_period.isin(tr)]; f = df[df.time_period.isin(fu)]
        if f.empty: continue
        m = build_chap_model(cfg, cov); m.fit(h); p = m.predict(h, f)
        sc = [c for c in p.columns if c.startswith("sample_")]
        mg = p.merge(f[["time_period", "location", TARGET]].astype({"location": str}),
                     on=["time_period", "location"], how="inner")
        for _, r in mg.iterrows():
            y = pd.to_numeric(r[TARGET], errors="coerce")
            if not np.isfinite(y): continue
            s = r[sc].to_numpy(float)
            lcs.append(crps(np.log1p(np.clip(s, 0, None)), float(np.log1p(max(y, 0)))))
            crs.append(crps(s, float(y)))
            ql, qh = np.quantile(s, [.1, .9]); covs.append(float(ql <= y <= qh))
    agg = lambda a: float(np.nanmean(a)) if a else np.nan
    return agg(lcs), agg(crs), agg(covs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-locations", type=int, default=120)
    ap.add_argument("--n-splits", type=int, default=6)
    args = ap.parse_args()
    df = pd.read_csv(DATASET); df["location"] = df["location"].astype(str)
    step = max(1, df["location"].nunique() // args.n_locations)
    df = df[df["location"].isin(sorted(df["location"].unique())[::step])].copy()
    freq = detect_frequency(df)
    times = sorted(df["time_period"].unique(), key=lambda p: period_to_timestamp(p, freq))
    splits = list(rolling(times, args.n_splits))
    base = load_model_configuration(BASE); cov = base.additional_continuous_covariates
    print(f"locations={df.location.nunique()} splits={len(splits)}")

    configs = [
        ("AutoARIMA (champion)", {}),
        ("fixed (0,1,1)", {"arima_order": [0, 1, 1]}),
        ("fixed (0,1,1)+drift", {"arima_order": [0, 1, 1], "arima_include_drift": True}),
        ("fixed (0,1,2)", {"arima_order": [0, 1, 2]}),
        ("fixed (1,1,1)", {"arima_order": [1, 1, 1]}),
    ]
    res = []
    for label, over in configs:
        cfg = base.user_option_values.model_copy(update=over)
        t0 = time.time()
        lc, cr, cv = score(df, cfg, cov, splits)
        res.append((lc, cr, cv, label))
        print(f"  {label:22s} log_crps={lc:.4f}  crps={cr:.2f}  cov={cv:.3f}  ({time.time()-t0:.0f}s)")
    res.sort(key=lambda r: (np.inf if np.isnan(r[0]) else r[0]))
    print("\n=== ranked by log-CRPS (subset proxy) ===")
    for lc, cr, cv, label in res:
        print(f"{lc:.4f}  crps={cr:.2f}  cov={cv:.3f}  {label}")


if __name__ == "__main__":
    main()
