"""Subset screen for spatial neighbor-target features (leave-one-out district mean of D).

Subsets by whole DISTRICTS (not locations) so neighbour structure is preserved. Compares
the champion (config_tgtlag3_var) against adding neighbor_target_lags=1/2/3. Reports
log-CRPS + CRPS + coverage(10-90). Confirm any winner on the full frozen harness.

Usage: uv run python scripts/hpo_neighbor.py [--n-districts 12] [--n-splits 6] [--n-periods 3]
"""
from __future__ import annotations
import argparse, time
import numpy as np, pandas as pd
from mstl_multistep.pipeline import build_chap_model
from mstl_multistep.run_config import load_model_configuration
from mstl_multistep.io_utils import detect_frequency, period_to_timestamp

DATASET = "/Users/knutdr/Data/CH/chap_data_level5_irs_allocated_monthly.csv"
BASE = "config_tgtlag3_var.yaml"   # champion: variance head + target lags
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
        merged = p.merge(f[["time_period", "location", TARGET]].astype({"location": str}),
                         on=["time_period", "location"], how="inner")
        for _, r in merged.iterrows():
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
    ap.add_argument("--n-districts", type=int, default=12)
    ap.add_argument("--n-splits", type=int, default=6)
    ap.add_argument("--n-periods", type=int, default=3)
    args = ap.parse_args()
    df = pd.read_csv(DATASET); df["location"] = df["location"].astype(str)
    dists = sorted(df["district"].unique())
    step = max(1, len(dists) // args.n_districts)
    keep = dists[::step]
    df = df[df["district"].isin(keep)].copy()
    freq = detect_frequency(df)
    times = sorted(df["time_period"].unique(), key=lambda p: period_to_timestamp(p, freq))
    splits = list(rolling(times, args.n_splits, n_periods=args.n_periods))
    base = load_model_configuration(BASE); cov = base.additional_continuous_covariates
    print(f"districts={df.district.nunique()} locations={df.location.nunique()} splits={len(splits)}")

    configs = [("champion (no nbr)", {})]
    for k in (1, 2, 3):
        configs.append((f"nbr_lags={k}", {"neighbor_group_col": "district", "neighbor_target_lags": k}))

    res = []
    for label, over in configs:
        cfg = base.user_option_values.model_copy(update=over)
        t0 = time.time()
        lc, cr, cv = score(df, cfg, cov, splits)
        res.append((lc, cr, cv, label))
        print(f"  {label:18s} log_crps={lc:.4f}  crps={cr:.3f}  cov={cv:.3f}  ({time.time()-t0:.0f}s)")
    res.sort(key=lambda r: (np.inf if np.isnan(r[0]) else r[0]))
    print("\n=== ranked by log-CRPS (subset proxy) ===")
    for lc, cr, cv, label in res:
        print(f"{lc:.4f}  crps={cr:.3f}  cov={cv:.3f}  {label}")


if __name__ == "__main__":
    main()
