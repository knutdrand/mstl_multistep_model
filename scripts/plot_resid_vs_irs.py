"""Scatter the RF target (ARIMA residual R) vs each IRS feature, coloured by district.

The 5-sector debug fixture is too sparse for a by-district view (≈1 sector/district), so this
computes R + IRS features for a richer slice -- the most spray-active districts, several sectors
each -- using the SAME residual definition as rf_train_design (R = D - ARIMA in-sample fit) and
the same IRS feature builder. One split (history = all but the last 3 months).

Writes debug/plots/resid_vs_irs.png.

Usage: uv run python scripts/plot_resid_vs_irs.py [--n-districts 6]
"""
from __future__ import annotations
import argparse, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mstl_multistep import decomposition as dec
from mstl_multistep.features import INDEX_COLS
from mstl_multistep.irs_features import build_irs_features
from mstl_multistep.pipeline import build_chap_model
from mstl_multistep.run_config import load_model_configuration
from mstl_multistep.io_utils import detect_frequency, period_to_timestamp

DATASET = "/Users/knutdr/Data/CH/chap_data_level5_irs_allocated_monthly.csv"
CONFIG = "config_tgtlag3_var.yaml"
OUT = "debug/plots"
H = 3
IRS_FEATS = [("irs_level", "irs_level (coverage)"), ("irs_decay", "irs_decay (persistence)"),
             ("irs_since", "irs_since (months)"), ("irs_cumulative", "irs_cumulative (stock)")]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n-districts", type=int, default=6)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    df = pd.read_csv(DATASET); df["location"] = df["location"].astype(str)
    freq = detect_frequency(df)
    gtimes = sorted(df["time_period"].unique(), key=lambda p: period_to_timestamp(p, freq))
    hist = df[df["time_period"].isin(gtimes[:len(gtimes) - H])].copy()

    # pick districts with the most ever-sprayed sectors (so by-district has multiplicity)
    sprayed_loc = set(df[df["irs_allocated"] > 0]["location"])
    spr_per_dist = (df[df["location"].isin(sprayed_loc)].drop_duplicates("location")
                    .groupby("district").size().sort_values(ascending=False))
    districts = spr_per_dist.head(args.n_districts).index.tolist()
    loc2dist = dict(zip(df["location"], df["district"]))
    locs = [l for l in sprayed_loc if loc2dist[l] in districts]
    hist = hist[hist["location"].isin(locs)].copy()
    print(f"districts={districts}\nsprayed sectors in them: {len(locs)}")

    mc = load_model_configuration(CONFIG); cfg = mc.user_option_values
    model = build_chap_model(cfg, mc.additional_continuous_covariates)
    model._freq = freq
    model._season_lengths = model._season_lengths_for(freq)
    sl = model._season_lengths
    target = cfg.target_variable

    # R = D - ARIMA in-sample fit (same as fit() builds _resid)
    deseason, _ = dec.decompose_panel(hist, freq, target, sl, cfg.log_transform)
    deseason["_ts"] = deseason["time_period"].apply(lambda p: period_to_timestamp(p, freq))
    blocks = []
    for loc, g in deseason.groupby("location", sort=False):
        g = g.sort_values("_ts"); y = g[target].to_numpy(float)
        fitted, _, _ = model._arima(y, h=1, want_fitted=True)
        blk = g[INDEX_COLS].copy(); blk["_resid"] = y - fitted
        blocks.append(blk)
    resid = pd.concat(blocks, ignore_index=True)

    irs, _ = build_irs_features(hist, None, cfg.irs_column, cfg.irs_features, cfg.irs_halflife)
    d = resid.merge(irs, on=INDEX_COLS, how="left")
    d["district"] = d["location"].map(loc2dist)
    d = d[np.isfinite(d["_resid"])]

    # ---- plot: 2x2, R vs each IRS feature, coloured by district ----
    cmap = plt.get_cmap("tab10")
    dlabels = {dist: f"D{i+1}" for i, dist in enumerate(districts)}
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.0), sharey=True)
    for ax, (col, lab) in zip(axes.ravel(), IRS_FEATS):
        for i, dist in enumerate(districts):
            sd = d[d["district"] == dist]
            ax.scatter(sd[col], sd["_resid"], s=10, alpha=0.35, color=cmap(i),
                       edgecolors="none", label=dlabels[dist])
        ax.axhline(0, color="0.6", lw=0.7)
        ax.set_xlabel(lab); ax.set_ylabel("$R = D-\\hat A$")
    handles = [plt.Line2D([], [], marker="o", ls="", color=cmap(i), label=dlabels[d_])
               for i, d_ in enumerate(districts)]
    fig.legend(handles=handles, loc="upper center", ncol=len(districts), fontsize=8,
               frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("RF residual $R$ vs IRS features, by district", y=0.97, x=0.5, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(f"{OUT}/resid_vs_irs.png"); plt.close(fig)
    print(f"wrote {OUT}/resid_vs_irs.png   ({len(d)} points)")
    print("district key:", {dlabels[k]: k for k in districts})


if __name__ == "__main__":
    main()
