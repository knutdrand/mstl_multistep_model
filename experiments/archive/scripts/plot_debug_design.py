"""Plot the RF training design (debug/rf_train_design.csv) for inspection.

Writes to debug/plots/:
  design_resid.png         -- the RF target (ARIMA residual R) per sector over time, masked rows shaded
  design_feature_corr.png  -- correlation of each feature with R (signal check), coloured by block
  design_features_sector.png -- feature-construction sanity for one sprayed sector
                               (R, rainfall anomaly, IRS features, target lags over time)

Usage: uv run python scripts/plot_debug_design.py
"""
from __future__ import annotations
import os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

IN = "debug/rf_train_design.csv"
OUT = "debug/plots"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 130, "savefig.bbox": "tight"})
BLOCK_C = {"climate": "#1f77b4", "irs": "#9467bd", "tgt": "#2ca44c", "loc": "#999999"}


def block_of(col):
    if col.startswith("loc_"):
        return "loc"
    if col.startswith("irs_"):
        return "irs"
    if col.startswith("tgt_lag"):
        return "tgt"
    return "climate"


def main():
    os.makedirs(OUT, exist_ok=True)
    d = pd.read_csv(IN)
    d["t"] = pd.PeriodIndex(d["time_period"].astype(str), freq="M").to_timestamp()
    feat_cols = [c for c in d.columns if c not in ("time_period", "location", "t", "_resid", "in_mask")]
    locs = sorted(d["location"].unique())

    # ---------- 1) RF target (residual R) per sector ----------
    fig, axes = plt.subplots(len(locs), 1, figsize=(6.8, 1.25 * len(locs)), sharex=True)
    for ax, loc in zip(np.atleast_1d(axes), locs):
        g = d[d.location == loc].sort_values("t")
        ax.axhline(0, color="0.8", lw=0.7)
        ax.plot(g["t"], g["_resid"], color="#2ca44c", lw=1.0)
        # shade each contiguous run of dropped (masked) rows, not start..last-masked
        m = (~g.in_mask).to_numpy()
        tt = g["t"].to_numpy()
        pad = np.timedelta64(15, "D")
        i = 0
        while i < len(m):
            if m[i]:
                j = i
                while j + 1 < len(m) and m[j + 1]:
                    j += 1
                ax.axvspan(tt[i] - pad, tt[j] + pad, color="0.85", alpha=0.6, lw=0)
                i = j + 1
            else:
                i += 1
        ax.set_ylabel(loc[:8], fontsize=7)
        ax.margins(x=0.01)
    axes[0].set_title("RF target: ARIMA residual $R=D-\\hat A$ per sector "
                      "(grey = rows dropped by lag/NaN mask)", fontsize=10, loc="left")
    fig.tight_layout(); fig.savefig(f"{OUT}/design_resid.png"); plt.close(fig)

    # ---------- 2) feature <-> target correlation ----------
    dm = d[d.in_mask].copy()
    corr = {c: np.corrcoef(dm[c], dm["_resid"])[0, 1] for c in feat_cols
            if dm[c].std() > 0}
    s = pd.Series(corr).sort_values()
    fig, ax = plt.subplots(figsize=(6.8, 0.28 * len(s) + 0.8))
    colors = [BLOCK_C[block_of(c)] for c in s.index]
    ax.barh(range(len(s)), s.values, color=colors)
    ax.set_yticks(range(len(s))); ax.set_yticklabels(s.index, fontsize=7)
    ax.axvline(0, color="0.5", lw=0.8)
    ax.set_xlabel("correlation with $R$ (on in-mask rows)")
    ax.set_title("Which features carry signal for the residual", fontsize=10, loc="left")
    handles = [plt.Rectangle((0, 0), 1, 1, color=BLOCK_C[b]) for b in BLOCK_C]
    ax.legend(handles, BLOCK_C.keys(), fontsize=7, loc="lower right", ncol=2)
    fig.tight_layout(); fig.savefig(f"{OUT}/design_feature_corr.png"); plt.close(fig)

    # ---------- 3) feature construction sanity, one sprayed sector ----------
    spr = dm.groupby("location")["irs_decay"].max()
    loc = spr.idxmax() if spr.max() > 0 else locs[0]
    g = d[d.location == loc].sort_values("t")
    fig, ax = plt.subplots(4, 1, figsize=(6.8, 6.2), sharex=True)
    ax[0].axhline(0, color="0.8", lw=0.7)
    ax[0].plot(g["t"], g["_resid"], color="#2ca44c", lw=1.0); ax[0].set_ylabel("$R$ (target)")
    ax[0].set_title(f"Feature construction for sector {loc}", fontsize=10, loc="left")
    ax[1].axhline(0, color="0.8", lw=0.7)
    ax[1].plot(g["t"], g["rainfall_era5_lag1"], color="#1f77b4", lw=1.0)
    ax[1].set_ylabel("rainfall\nanomaly lag1")
    a2 = ax[2]
    a2.plot(g["t"], g["irs_level"], color="#d62728", lw=1.2, label="irs_level")
    a2.plot(g["t"], g["irs_decay"], color="#9467bd", lw=1.2, label="irs_decay")
    a2.set_ylabel("IRS level / decay"); a2.legend(fontsize=7, loc="upper left")
    a2b = a2.twinx(); a2b.spines["right"].set_visible(True)
    a2b.plot(g["t"], g["irs_since"], color="0.5", lw=0.8, ls=":")
    a2b.set_ylabel("irs_since (mo)", color="0.4", fontsize=8)
    for j, c in enumerate(["tgt_lag1", "tgt_lag2", "tgt_lag3"]):
        ax[3].plot(g["t"], g[c], lw=1.0, label=c)
    ax[3].set_ylabel("target lags $D_{t-k}$"); ax[3].legend(fontsize=7, ncol=3, loc="upper left")
    for a in ax:
        a.margins(x=0.01)
    fig.tight_layout(); fig.savefig(f"{OUT}/design_features_sector.png"); plt.close(fig)

    print("wrote 3 plots to", OUT)
    print("top |corr| with R:")
    print(s.reindex(s.abs().sort_values(ascending=False).index).head(6).round(3).to_string())


if __name__ == "__main__":
    main()
