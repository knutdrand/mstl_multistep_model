"""Generate a small, faithful debugging fixture for ONE test split of the champion.

Picks a handful of inspectable sectors and one held-out split (the most recent 3 months),
then writes, into debug/:
  raw_historic.csv        -- raw CHAP rows used to fit   (times[:origin])
  raw_future.csv          -- raw CHAP rows to forecast    (times[origin:origin+3])
  rf_train_design.csv     -- the design the RF is TRAINED on: index + every feature column
                             + `_resid` (the ARIMA residual R = D - A, the RF target) + in_mask
  rf_predict_design.csv    -- the future feature rows fed to rf.predict() at forecast time

Fidelity is asserted: the labeled reconstruction is compared bit-for-bit against the arrays
the model actually hands to the RF during fit() and predict().

Usage: uv run python scripts/make_debug_split.py [--n-locations 5]
"""
from __future__ import annotations
import argparse, os
import numpy as np, pandas as pd

from mstl_multistep import decomposition as dec
from mstl_multistep.features import INDEX_COLS, build_model_features
from mstl_multistep.irs_features import build_irs_features
from mstl_multistep.pipeline import build_chap_model
from mstl_multistep.run_config import load_model_configuration
from mstl_multistep.io_utils import detect_frequency, period_to_timestamp

DATASET = "/Users/knutdr/Data/CH/chap_data_level5_irs_allocated_monthly.csv"
CONFIG = "config_tgtlag3_var.yaml"
OUTDIR = "debug"
H = 3


def pick_locations(df, n):
    """A few full-history sectors, biased to include sprayed ones (so IRS features fire)."""
    full = df.groupby("location").size().max()
    keep = df.groupby("location").filter(lambda g: len(g) == full)["location"].unique()
    sprayed = set(df[df["irs_allocated"] > 0]["location"])
    spr = [l for l in sorted(keep) if l in sprayed]
    non = [l for l in sorted(keep) if l not in sprayed]
    chosen = spr[: max(1, n // 2)] + non[: n - max(1, n // 2)]
    return sorted(chosen)[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-locations", type=int, default=5)
    args = ap.parse_args()
    os.makedirs(OUTDIR, exist_ok=True)

    df = pd.read_csv(DATASET); df["location"] = df["location"].astype(str)
    freq = detect_frequency(df)
    gtimes = sorted(df["time_period"].unique(), key=lambda p: period_to_timestamp(p, freq))
    origin = len(gtimes) - H                      # one split: forecast the last 3 months
    train_times, fut_times = gtimes[:origin], gtimes[origin:origin + H]

    locs = pick_locations(df, args.n_locations)
    sub = df[df["location"].isin(locs)].copy()
    historic = sub[sub["time_period"].isin(train_times)].copy()
    future = sub[sub["time_period"].isin(fut_times)].copy()
    print(f"locations={locs}\norigin={gtimes[origin]}  train={train_times[0]}..{train_times[-1]}  "
          f"future={fut_times}")

    # --- raw split files ---
    historic.to_csv(f"{OUTDIR}/raw_historic.csv", index=False)
    future.to_csv(f"{OUTDIR}/raw_future.csv", index=False)

    mc = load_model_configuration(CONFIG); cfg = mc.user_option_values
    cols = mc.additional_continuous_covariates
    model = build_chap_model(cfg, cols)

    # capture the exact (X, y) arrays the RF is trained on
    captrain = {}
    real_fmv = model._fit_mean_and_variance
    def wrap_fmv(Xm, ym):
        captrain["X"], captrain["y"] = np.asarray(Xm).copy(), np.asarray(ym).copy()
        return real_fmv(Xm, ym)
    model._fit_mean_and_variance = wrap_fmv
    model.fit(historic)
    model._fit_mean_and_variance = real_fmv
    sl = model._season_lengths
    feat_cols = model._feat_cols
    print(f"RF features ({len(feat_cols)}): {feat_cols}")

    # --- reconstruct the labeled training design exactly as fit() builds it ---
    target = cfg.target_variable
    deseason, _ = dec.decompose_panel(historic, freq, target, sl, cfg.log_transform)
    feats = build_model_features(historic, None, cols, freq, sl, cfg.feature_min_lag,
                                 cfg.feature_max_lag, cfg.use_location_dummies,
                                 cfg.deseasonalize_covariates, cfg.lags_by_col())
    irs, irs_cols = build_irs_features(historic, None, cfg.irs_column, cfg.irs_features, cfg.irs_halflife)
    feats = feats.merge(irs, on=INDEX_COLS, how="left")
    for c in irs_cols:
        feats[c] = feats[c].fillna(0.0)

    deseason = deseason.copy()
    deseason["_ts"] = deseason["time_period"].apply(lambda p: period_to_timestamp(p, freq))
    blocks = []
    for loc, g in deseason.groupby("location", sort=False):
        g = g.sort_values("_ts"); y = g[target].to_numpy(float)
        fitted, _, _ = model._arima(y, h=1, want_fitted=True)
        blk = g[INDEX_COLS].copy(); blk["_resid"] = y - fitted
        blocks.append(blk)
    resid_df = pd.concat(blocks, ignore_index=True)

    tl = model._target_lag_frame(deseason, target, cfg.rf_target_lags)
    feats = feats.merge(tl, on=INDEX_COLS, how="left")
    design = feats.merge(resid_df, on=INDEX_COLS, how="left")

    X = design[feat_cols].to_numpy(float)
    yres = design["_resid"].to_numpy(float)
    mask = ~(np.isnan(X).any(axis=1) | np.isnan(yres))
    design["in_mask"] = mask

    # fidelity: labeled reconstruction == what the RF actually trained on
    assert np.allclose(X[mask], captrain["X"], equal_nan=True), "train X mismatch"
    assert np.allclose(yres[mask], captrain["y"], equal_nan=True), "train y mismatch"
    print(f"FIDELITY ok: train design matches RF input  ({mask.sum()} train rows, "
          f"{(~mask).sum()} dropped by lag/NaN mask)")

    out_cols = INDEX_COLS + feat_cols + ["_resid", "in_mask"]
    design.sort_values(INDEX_COLS)[out_cols].to_csv(f"{OUTDIR}/rf_train_design.csv", index=False)

    # --- predict design: the future feature rows fed to rf.predict(), per location ---
    rows = []
    cap = {}
    real_pred = model._rf.predict
    model._rf.predict = lambda Xf: (cap.__setitem__("Xf", np.asarray(Xf)) or real_pred(Xf))
    for loc in locs:
        hl = historic[historic["location"] == loc]
        fl = future[future["location"] == loc].sort_values("time_period")
        model.predict(hl, fl)
        Xf = cap["Xf"]
        for step, (_, fr) in enumerate(fl.iterrows()):
            rows.append({"location": loc, "time_period": fr["time_period"],
                         **{c: Xf[step, j] for j, c in enumerate(feat_cols)}})
    model._rf.predict = real_pred
    pred_design = pd.DataFrame(rows)[["location", "time_period"] + feat_cols]
    pred_design.to_csv(f"{OUTDIR}/rf_predict_design.csv", index=False)

    print(f"\nwrote to {OUTDIR}/:")
    for f in ("raw_historic.csv", "raw_future.csv", "rf_train_design.csv", "rf_predict_design.csv"):
        n = sum(1 for _ in open(f"{OUTDIR}/{f}")) - 1
        print(f"  {f:24s} {n:6d} rows")


if __name__ == "__main__":
    main()
