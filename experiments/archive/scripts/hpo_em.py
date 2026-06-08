"""Subset screen for the EM/backfitting variant: log-CRPS + coverage vs em_iterations/damping.

Coverage is reported because the suspected EM failure mode is ARIMA's sigma tightening on
the covariate-cleaned residual. Confirm winners on the full frozen harness afterwards.

Usage: uv run python scripts/hpo_em.py [--n-locations 100] [--n-splits 8]
"""
from __future__ import annotations
import argparse, time
import numpy as np, pandas as pd
from mstl_multistep.pipeline import build_chap_model
from mstl_multistep.run_config import load_model_configuration
from mstl_multistep.io_utils import detect_frequency, period_to_timestamp

DATASET = "/Users/knutdr/Data/CH/chap_data_level5_irs_allocated_monthly.csv"
BASE = "config_irs_lags.yaml"   # per-covariate-lags champion, rf_target_lags=0
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
    lcs, covs = [], []
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
    return float(np.nanmean(lcs)) if lcs else np.nan, float(np.nanmean(covs)) if covs else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-locations", type=int, default=100)
    ap.add_argument("--n-splits", type=int, default=8)
    args = ap.parse_args()
    df = pd.read_csv(DATASET); df["location"] = df["location"].astype(str)
    step = max(1, df["location"].nunique() // args.n_locations)
    df = df[df["location"].isin(sorted(df["location"].unique())[::step])].copy()
    freq = detect_frequency(df)
    times = sorted(df["time_period"].unique(), key=lambda p: period_to_timestamp(p, freq))
    splits = list(rolling(times, args.n_splits))
    base = load_model_configuration(BASE); cov = base.additional_continuous_covariates
    print(f"locations={df.location.nunique()} splits={len(splits)}")

    configs = [("em=1 (current)", {"em_iterations": 1})]
    for d in (1.0, 0.5):
        for s in (1.0, 1.2, 1.4):
            configs.append((f"em=2 damp={d} sig={s}",
                            {"em_iterations": 2, "em_damping": d, "em_sigma_scale": s}))

    res = []
    for label, over in configs:
        cfg = base.user_option_values.model_copy(update=over)
        t0 = time.time()
        lc, cv = score(df, cfg, cov, splits)
        res.append((lc, cv, label))
        print(f"  {label:18s} log_crps={lc:.4f}  cov10-90={cv:.3f}  ({time.time()-t0:.0f}s)")
    res.sort(key=lambda r: (np.inf if np.isnan(r[0]) else r[0]))
    print("\n=== ranked by log-CRPS (subset proxy) ===")
    for lc, cv, label in res:
        print(f"{lc:.4f}  cov={cv:.3f}  {label}")


if __name__ == "__main__":
    main()
