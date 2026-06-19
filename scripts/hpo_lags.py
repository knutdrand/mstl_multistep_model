"""In-process subset HPO over per-covariate lags + rf_target_lags (rf_residual).

Screens lag configurations on a *location subset* with a rolling-origin log-CRPS
backtest (same metric chap reports as crps_log1p), reusing the champion config
(config_irs_lags.yaml) as the base. The subset keeps it fast; confirm the top
candidates afterwards on the full frozen chap harness.

Usage:
    uv run python scripts/hpo_lags.py [--n-locations 100] [--n-splits 8]
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from mstl_multistep.pipeline import build_chap_model
from mstl_multistep.run_config import load_model_configuration
from mstl_multistep.io_utils import detect_frequency, period_to_timestamp

DATASET = "/Users/knutdr/Data/CH/chap_data_level5_irs_allocated_monthly.csv"
BASE_CONFIG = "config_irs_lags.yaml"
TARGET = "disease_cases"


def crps_sample(samples, y):
    s = np.sort(np.asarray(samples, dtype=float))
    m = len(s)
    if m == 0 or not np.isfinite(y):
        return np.nan
    term1 = np.mean(np.abs(s - y))
    i = np.arange(m)
    term2 = np.sum((2 * i - (m - 1)) * s) / (m * m)
    return float(term1 - term2)


def log_crps(preds, truth):
    sc = [c for c in preds.columns if c.startswith("sample_")]
    merged = preds.merge(
        truth[["time_period", "location", TARGET]].astype({"location": str}),
        on=["time_period", "location"], how="inner",
    )
    out = []
    for _, r in merged.iterrows():
        y = pd.to_numeric(r[TARGET], errors="coerce")
        if not np.isfinite(y):
            continue
        s = r[sc].to_numpy(dtype=float)
        out.append(crps_sample(np.log1p(np.clip(s, 0, None)), float(np.log1p(max(y, 0)))))
    return float(np.nanmean(out)) if out else np.nan


def rolling_splits(times, n_splits, n_periods=3, stride=1):
    T = len(times)
    for k in range(n_splits):
        origin = T - n_periods - k * stride
        if origin < 24:
            break
        yield times[:origin], times[origin: origin + n_periods]


def score(df, cfg, cov, splits):
    per = []
    for tr, fu in splits:
        h = df[df.time_period.isin(tr)]
        f = df[df.time_period.isin(fu)]
        if f.empty:
            continue
        m = build_chap_model(cfg, cov)
        m.fit(h)
        per.append(log_crps(m.predict(h, f), f))
    return float(np.nanmean(per)) if per else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-locations", type=int, default=100)
    ap.add_argument("--n-splits", type=int, default=8)
    args = ap.parse_args()

    df = pd.read_csv(DATASET)
    df["location"] = df["location"].astype(str)
    locs = sorted(df["location"].unique())[:: max(1, len(set(df["location"])) // args.n_locations)]
    df = df[df["location"].isin(locs)].copy()
    freq = detect_frequency(df)
    times = sorted(df["time_period"].unique(), key=lambda p: period_to_timestamp(p, freq))
    splits = list(rolling_splits(times, args.n_splits))
    base = load_model_configuration(BASE_CONFIG)
    cov = base.additional_continuous_covariates
    print(f"locations={len(locs)} splits={len(splits)} covariates={cov}")

    # champion lag config for reference
    CHAMP = {"rainfall_era5": [1, 6], "relative_humidity": [1, 4], "mean_temperature": [1, 2]}

    configs = []
    # --- Phase A: per-covariate lag grid (rf_target_lags=0) ---
    for rmax in (4, 6, 9):
        for hmax in (3, 4, 6):
            for tmax in (2, 4):
                cl = {"rainfall_era5": [1, rmax], "relative_humidity": [1, hmax],
                      "mean_temperature": [1, tmax]}
                configs.append((f"lags r{rmax} h{hmax} t{tmax}",
                                {"covariate_lags": cl, "rf_target_lags": 0}))
    # --- Phase B: target lags at champion lag config ---
    for k in (2, 3, 6):
        configs.append((f"champlags + tgtlag{k}",
                        {"covariate_lags": CHAMP, "rf_target_lags": k}))

    results = []
    for label, over in configs:
        cfg = base.user_option_values.model_copy(update=over)
        t0 = time.time()
        sc = score(df, cfg, cov, splits)
        results.append((sc, label))
        print(f"  {label:28s} log_crps={sc:.4f}  ({time.time()-t0:.0f}s)")

    results.sort(key=lambda r: (np.inf if np.isnan(r[0]) else r[0]))
    print("\n=== ranked (best first), subset proxy ===")
    for sc, label in results:
        print(f"{sc:.4f}  {label}")


if __name__ == "__main__":
    main()
