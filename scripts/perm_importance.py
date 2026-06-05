"""Permutation importance of the IRS-bank features on the RF residual target.

Reconstructs the RF training design for config_irs_bank (same way fit() builds it), then ranks
features by permutation importance (sklearn) on a held-out split. Highlights the IRS bank
features so we can see which channels/decays/recency terms the RF actually uses, and prune.

Usage: uv run python scripts/perm_importance.py [--n-locations 200]
"""
from __future__ import annotations
import argparse
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

from mstl_multistep import decomposition as dec
from mstl_multistep.features import INDEX_COLS, build_model_features
from mstl_multistep.irs_features import build_irs_features
from mstl_multistep.pipeline import build_chap_model
from mstl_multistep.run_config import load_model_configuration
from mstl_multistep.io_utils import detect_frequency, period_to_timestamp

DATASET = "/Users/knutdr/Data/CH/chap_data_level5_irs_allocated_monthly.csv"
CONFIG = "config_irs_bank.yaml"
H = 3


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n-locations", type=int, default=200)
    args = ap.parse_args()
    df = pd.read_csv(DATASET); df["location"] = df["location"].astype(str)
    step = max(1, df["location"].nunique() // args.n_locations)
    df = df[df["location"].isin(sorted(df["location"].unique())[::step])].copy()
    freq = detect_frequency(df)
    gt = sorted(df["time_period"].unique(), key=lambda p: period_to_timestamp(p, freq))
    hist = df[df["time_period"].isin(gt[:len(gt) - H])].copy()

    mc = load_model_configuration(CONFIG); cfg = mc.user_option_values
    model = build_chap_model(cfg, mc.additional_continuous_covariates)
    model.fit(hist)                      # sets _freq/_season_lengths/_feat_cols
    sl = model._season_lengths; tgt = cfg.target_variable
    feat_cols = model._feat_cols

    # rebuild the labeled design exactly as fit() does
    deseason, _ = dec.decompose_panel(hist, freq, tgt, sl, cfg.log_transform)
    feats = build_model_features(hist, None, mc.additional_continuous_covariates, freq, sl,
                                 cfg.feature_min_lag, cfg.feature_max_lag,
                                 model._sector_dummies_flag(), cfg.deseasonalize_covariates,
                                 cfg.lags_by_col())
    irs, irs_cols = build_irs_features(hist, None, cfg.irs_column, cfg.irs_features,
                                       cfg.irs_halflife, chem_column=cfg.irs_chemical_column)
    feats = feats.merge(irs, on=INDEX_COLS, how="left")
    for c in irs_cols:
        feats[c] = feats[c].fillna(0.0)
    deseason = deseason.copy()
    deseason["_ts"] = deseason["time_period"].apply(lambda p: period_to_timestamp(p, freq))
    blocks = []
    for loc, g in deseason.groupby("location", sort=False):
        g = g.sort_values("_ts"); y = g[tgt].to_numpy(float)
        fit_, _, _ = model._arima(y, h=1, want_fitted=True)
        b = g[INDEX_COLS].copy(); b["_resid"] = y - fit_; blocks.append(b)
    resid = pd.concat(blocks, ignore_index=True)
    tl = model._target_lag_frame(deseason, tgt, cfg.rf_target_lags)
    feats = feats.merge(tl, on=INDEX_COLS, how="left")
    design = feats.merge(resid, on=INDEX_COLS, how="left")

    X = design[feat_cols].to_numpy(float)
    y = design["_resid"].to_numpy(float)
    mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
    X, y = X[mask], y[mask]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0)
    rf = RandomForestRegressor(n_estimators=cfg.rf.n_estimators, max_depth=cfg.rf.max_depth,
                               min_samples_leaf=cfg.rf.min_samples_leaf,
                               max_features=cfg.rf.max_features, random_state=42, n_jobs=-1)
    rf.fit(Xtr, ytr)
    print(f"locations={df.location.nunique()} design_rows={len(y)} features={len(feat_cols)} "
          f"test_R2={rf.score(Xte, yte):.4f}")
    pi = permutation_importance(rf, Xte, yte, n_repeats=8, random_state=0, n_jobs=-1)
    imp = pd.Series(pi.importances_mean, index=feat_cols).sort_values(ascending=False)

    irs_feats = [c for c in feat_cols if c.startswith("irs_")]
    print("\n=== IRS bank features by permutation importance (rank among ALL features) ===")
    ranks = {c: r for r, c in enumerate(imp.index, 1)}
    for c in sorted(irs_feats, key=lambda c: ranks[c]):
        print(f"  rank {ranks[c]:3d}/{len(feat_cols)}  imp={imp[c]:+.5f}  {c}")
    print("\n=== top 12 features overall ===")
    for c, v in imp.head(12).items():
        print(f"  {v:+.5f}  {c}")


if __name__ == "__main__":
    main()
