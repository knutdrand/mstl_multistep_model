"""irs_decay vs RF residual R, faceted by district, points coloured by sector.

Same residual definition as rf_train_design (R = D - ARIMA in-sample fit) and the same IRS
feature builder; one split (history = all but the last 3 months). One subplot per district;
within a subplot each sector gets its own colour (no global legend -- 79 sectors -- a thin grey
line shows the per-district linear fit of R on irs_decay).

Writes debug/plots/resid_decay_facet.png.

Usage: uv run python scripts/plot_resid_decay_facet.py [--n-districts 6]
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


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n-districts", type=int, default=6)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    df = pd.read_csv(DATASET); df["location"] = df["location"].astype(str)
    freq = detect_frequency(df)
    gtimes = sorted(df["time_period"].unique(), key=lambda p: period_to_timestamp(p, freq))
    hist = df[df["time_period"].isin(gtimes[:len(gtimes) - H])].copy()

    sprayed_loc = set(df[df["irs_allocated"] > 0]["location"])
    spr_per_dist = (df[df["location"].isin(sprayed_loc)].drop_duplicates("location")
                    .groupby("district").size().sort_values(ascending=False))
    districts = spr_per_dist.head(args.n_districts).index.tolist()
    loc2dist = dict(zip(df["location"], df["district"]))
    locs = [l for l in sprayed_loc if loc2dist[l] in districts]
    hist = hist[hist["location"].isin(locs)].copy()

    mc = load_model_configuration(CONFIG); cfg = mc.user_option_values
    model = build_chap_model(cfg, mc.additional_continuous_covariates)
    model._freq = freq; model._season_lengths = model._season_lengths_for(freq)
    sl = model._season_lengths; target = cfg.target_variable

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
    d = d[np.isfinite(d["_resid"]) & np.isfinite(d["irs_decay"])]

    ncol = 3
    nrow = int(np.ceil(len(districts) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 2.6 * nrow), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()
    cmap = plt.get_cmap("tab20")
    for k, dist in enumerate(districts):
        ax = axes[k]
        sd = d[d["district"] == dist]
        secs = sorted(sd["location"].unique())
        for i, loc in enumerate(secs):
            sl_ = sd[sd["location"] == loc]
            ax.scatter(sl_["irs_decay"], sl_["_resid"], s=9, alpha=0.45,
                       color=cmap(i % 20), edgecolors="none")
        # per-district linear fit of R on irs_decay
        x, yv = sd["irs_decay"].to_numpy(), sd["_resid"].to_numpy()
        if len(x) > 10 and np.ptp(x) > 0:
            b, a = np.polyfit(x, yv, 1)
            xx = np.linspace(x.min(), x.max(), 50)
            ax.plot(xx, a + b * xx, color="0.25", lw=1.3)
            ax.text(0.96, 0.04, f"slope={b:+.2f}", transform=ax.transAxes, fontsize=7,
                    ha="right", va="bottom", color="0.25")
        ax.axhline(0, color="0.7", lw=0.6)
        ax.set_title(f"D{k+1}  ({len(secs)} sectors)", fontsize=9, loc="left")
    for k in range(len(districts), len(axes)):
        axes[k].axis("off")
    for k in range(len(districts)):
        if k % ncol == 0:
            axes[k].set_ylabel("$R = D-\\hat A$")
        if k >= len(districts) - ncol:
            axes[k].set_xlabel("irs_decay")
    fig.suptitle("RF residual $R$ vs irs_decay — faceted by district, coloured by sector",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(f"{OUT}/resid_decay_facet.png"); plt.close(fig)
    print(f"wrote {OUT}/resid_decay_facet.png  ({len(d)} pts, {len(locs)} sectors, "
          f"{len(districts)} districts)")
    print("district key:", {f"D{i+1}": dd for i, dd in enumerate(districts)})


if __name__ == "__main__":
    main()
