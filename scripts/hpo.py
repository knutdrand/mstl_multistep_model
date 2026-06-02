"""Fast in-process hyperparameter search by rolling-origin log-CRPS.

Runs a rolling-origin backtest (mimicking ``chap eval``'s n_splits / n_periods
/ stride) entirely in-process and scores each config by **log-CRPS** — CRPS of
the sample forecast computed on ``log1p`` values, the metric chap reports as
``crps_log1p``. This avoids the per-config ``uv``/``chap`` subprocess overhead,
so a dozen configs over a dozen splits run in a few minutes. Confirm the winner
afterwards with a real tracked ``chap eval``.

Usage::

    uv run python scripts/hpo.py DATASET_CSV [--n-splits 12] [--n-periods 3] [--stride 1]
"""

from __future__ import annotations

import argparse
import itertools
import time

import numpy as np
import pandas as pd

from mstl_multistep import RunConfig, build_chap_model
from mstl_multistep.io_utils import detect_frequency, period_to_timestamp


def crps_sample(samples: np.ndarray, y: float) -> float:
    """Empirical CRPS for one observation from a 1-d sample vector."""
    s = np.sort(np.asarray(samples, dtype=float))
    m = len(s)
    if m == 0 or not np.isfinite(y):
        return np.nan
    term1 = np.mean(np.abs(s - y))
    # Σ_{i<j}(s_j - s_i) = Σ_i (2i-(m-1)) s_i for ascending sorted s (i from 0).
    i = np.arange(m)
    pairwise = np.sum((2 * i - (m - 1)) * s)
    term2 = pairwise / (m * m)
    return float(term1 - term2)


def log_crps_over_window(preds: pd.DataFrame, truth: pd.DataFrame, target: str) -> float:
    """Mean log-CRPS across all (location, time_period) rows in a window."""
    sample_cols = [c for c in preds.columns if c.startswith("sample_")]
    merged = preds.merge(
        truth[["time_period", "location", target]].astype({"location": str}),
        on=["time_period", "location"],
        how="inner",
    )
    scores = []
    for _, row in merged.iterrows():
        y = pd.to_numeric(row[target], errors="coerce")
        if not np.isfinite(y):
            continue
        s = row[sample_cols].to_numpy(dtype=float)
        scores.append(crps_sample(np.log1p(np.clip(s, 0, None)), float(np.log1p(max(y, 0)))))
    return float(np.nanmean(scores)) if scores else np.nan


def rolling_splits(times: list[str], n_splits: int, n_periods: int, stride: int):
    """Yield (train_times, future_times) windows, most-recent split first."""
    T = len(times)
    for k in range(n_splits):
        origin = T - n_periods - k * stride
        if origin < 24:  # need >= ~2 seasons of monthly history for MSTL
            break
        yield times[:origin], times[origin : origin + n_periods]


def score_config(
    df: pd.DataFrame,
    cfg: RunConfig,
    feature_columns: list[str],
    splits,
    target: str,
) -> float:
    """Mean log-CRPS of one config across all backtest splits."""
    per_split = []
    for train_times, future_times in splits:
        historic = df[df["time_period"].isin(train_times)].copy()
        future = df[df["time_period"].isin(future_times)].copy()
        if future.empty:
            continue
        model = build_chap_model(cfg, feature_columns)
        model.fit(historic)
        preds = model.predict(historic, future)
        per_split.append(log_crps_over_window(preds, future, target))
    return float(np.nanmean(per_split)) if per_split else np.nan


# Curated search space. Kept small on purpose; expand as needed.
COVARIATES = ["rainfall", "mean_temperature", "mean_relative_humidity"]

GRID = {
    "prob_model": ["arima_residual", "recursive_residual"],
    "n_target_lags": [4, 6, 9],
    "rf_max_depth": [10, None],
    "rf_min_samples_leaf": [5],
}


def build_configs():
    keys = list(GRID)
    for combo in itertools.product(*(GRID[k] for k in keys)):
        d = dict(zip(keys, combo))
        cfg = RunConfig(
            prob_model=d["prob_model"],
            n_target_lags=d["n_target_lags"],
            log_transform=True,
            deseasonalize_covariates=True,
            n_samples=100,
            feature_min_lag=1,
            feature_max_lag=3,
            arima_level=68,
            rf={
                "n_estimators": 200,
                "max_depth": d["rf_max_depth"],
                "min_samples_leaf": d["rf_min_samples_leaf"],
                "max_features": "sqrt",
                "random_state": 42,
            },
        )
        label = (
            f"{d['prob_model'][:5]} lags={d['n_target_lags']} "
            f"depth={d['rf_max_depth']} leaf={d['rf_min_samples_leaf']}"
        )
        yield label, cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset_csv")
    ap.add_argument("--n-splits", type=int, default=12)
    ap.add_argument("--n-periods", type=int, default=3)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--target", default="disease_cases")
    args = ap.parse_args()

    df = pd.read_csv(args.dataset_csv)
    df["location"] = df["location"].astype(str)
    freq = detect_frequency(df)
    times = sorted(df["time_period"].unique(), key=lambda p: period_to_timestamp(p, freq))
    splits = list(rolling_splits(times, args.n_splits, args.n_periods, args.stride))
    feature_columns = [c for c in COVARIATES if c in df.columns]
    print(f"dataset={args.dataset_csv} freq={freq} splits={len(splits)} covariates={feature_columns}")

    results = []
    for label, cfg in build_configs():
        t0 = time.time()
        score = score_config(df, cfg, feature_columns, splits, args.target)
        results.append((score, label, cfg))
        print(f"  {label:40s} log_crps={score:.4f}  ({time.time()-t0:.0f}s)")

    results.sort(key=lambda r: (np.inf if np.isnan(r[0]) else r[0]))
    print("\n=== ranked (best first) ===")
    for score, label, _ in results:
        print(f"{score:.4f}  {label}")
    best_score, best_label, best_cfg = results[0]
    print(f"\nBEST: {best_label}  log_crps={best_score:.4f}")
    print("best user_option_values:")
    print(best_cfg.model_dump())


if __name__ == "__main__":
    main()
