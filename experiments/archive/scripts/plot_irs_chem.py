"""Scatter plots for the chemical-aware IRS decay feature.

Computes R (= D - ARIMA in-sample fit, the RF target) and the chemical-aware irs_decay for all
ever-sprayed sectors, plus the active insecticide per row, then writes to debug/plots/:
  irs_chem_decay_curves.png -- the literature decay curve per chemical (half-life illustration)
  irs_chem_resid_scatter.png -- R vs chemical-aware irs_decay, coloured by active chemical,
                                with per-chemical linear fits (do longer-lasting products differ?)
  irs_chem_resid_facet.png   -- R vs chemical-aware irs_decay faceted by district, coloured by sector

Usage: uv run python scripts/plot_irs_chem.py
"""
from __future__ import annotations
import os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mstl_multistep import decomposition as dec
from mstl_multistep.features import INDEX_COLS
from mstl_multistep.irs_features import build_irs_features, chem_halflife, CHEM_HALFLIFE_MONTHS
from mstl_multistep.pipeline import build_chap_model
from mstl_multistep.run_config import load_model_configuration
from mstl_multistep.io_utils import detect_frequency, period_to_timestamp

DATASET = "/Users/knutdr/Data/CH/chap_data_level5_irs_allocated_monthly.csv"
CONFIG = "config_irs_chem.yaml"
OUT = "debug/plots"
H = 3
CHEM_COL = "irs_insecticide_used"
# display label + colour per chemical class (by half-life)
CLASS = {2.0: ("Bendiocarb (2mo)", "#d62728"), 3.0: ("Deltamethrin (3mo)", "#ff7f0e"),
         5.0: ("Actellic/Pirimiphos (5mo)", "#1f77b4"), 8.0: ("Fludora (8mo)", "#2ca44c"),
         4.0: ("other/unknown (4mo)", "#999999")}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 130, "savefig.bbox": "tight"})


def main():
    os.makedirs(OUT, exist_ok=True)
    df = pd.read_csv(DATASET); df["location"] = df["location"].astype(str)
    freq = detect_frequency(df)
    gtimes = sorted(df["time_period"].unique(), key=lambda p: period_to_timestamp(p, freq))
    hist = df[df["time_period"].isin(gtimes[:len(gtimes) - H])].copy()
    sprayed = sorted(set(df[df["irs_allocated"] > 0]["location"]))
    hist = hist[hist["location"].isin(sprayed)].copy()
    loc2dist = dict(zip(df["location"], df["district"]))

    mc = load_model_configuration(CONFIG); cfg = mc.user_option_values
    model = build_chap_model(cfg, mc.additional_continuous_covariates)
    model._freq = freq; model._season_lengths = model._season_lengths_for(freq)
    sl = model._season_lengths; target = cfg.target_variable

    # R = D - ARIMA fit
    deseason, _ = dec.decompose_panel(hist, freq, target, sl, cfg.log_transform)
    deseason["_ts"] = deseason["time_period"].apply(lambda p: period_to_timestamp(p, freq))
    blocks = []
    for loc, g in deseason.groupby("location", sort=False):
        g = g.sort_values("_ts"); y = g[target].to_numpy(float)
        fitted, _, _ = model._arima(y, h=1, want_fitted=True)
        blk = g[INDEX_COLS].copy(); blk["_resid"] = y - fitted
        blocks.append(blk)
    resid = pd.concat(blocks, ignore_index=True)

    # chemical-aware decay + active chemical per row
    irs, _ = build_irs_features(hist, None, cfg.irs_column, ["decay"], cfg.irs_halflife,
                                chem_column=CHEM_COL)
    h = hist[INDEX_COLS + ["irs_allocated", CHEM_COL]].copy()
    h["_hl"] = h[CHEM_COL].apply(lambda s: chem_halflife(s, cfg.irs_halflife))
    h["_ts"] = h["time_period"].apply(lambda p: period_to_timestamp(p, freq))
    # active half-life = most recent campaign's half-life (ffill where allocated>0)
    h = h.sort_values(["location", "_ts"])
    h["_camp_hl"] = h["_hl"].where(h["irs_allocated"] > 0)
    h["_active_hl"] = h.groupby("location")["_camp_hl"].ffill()

    d = resid.merge(irs, on=INDEX_COLS).merge(
        h[INDEX_COLS + ["_active_hl"]], on=INDEX_COLS)
    d["district"] = d["location"].map(loc2dist)
    d = d[np.isfinite(d["_resid"])]
    print(f"sprayed sectors={len(sprayed)}  rows={len(d)}  under-protection rows (decay>0.05)="
          f"{int((d['irs_decay'] > 0.05).sum())}")

    # ---- 1) literature decay curves ----
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    t = np.arange(0, 13)
    for hl in (2.0, 3.0, 5.0, 8.0):
        lab, col = CLASS[hl]
        ax.plot(t, 0.5 ** (t / hl), color=col, lw=1.8, marker="o", ms=3, label=lab)
    ax.plot(t, 0.5 ** (t / 4.0), color="0.5", lw=1.2, ls="--", label="fixed champion (4mo)")
    ax.set_xlabel("months since campaign"); ax.set_ylabel("protection (decay feature)")
    ax.set_title("Literature residual-decay per insecticide", fontsize=10, loc="left")
    ax.legend(fontsize=7.5)
    fig.tight_layout(); fig.savefig(f"{OUT}/irs_chem_decay_curves.png"); plt.close(fig)

    # ---- 2) R vs chemical-aware decay, coloured by active chemical, per-chemical fits ----
    dp = d[d["irs_decay"] > 0.05]
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    for hl in (2.0, 3.0, 5.0, 8.0):
        sd = dp[np.isclose(dp["_active_hl"], hl)]
        if len(sd) < 5:
            continue
        lab, col = CLASS[hl]
        ax.scatter(sd["irs_decay"], sd["_resid"], s=10, alpha=0.35, color=col, edgecolors="none")
        b, a = np.polyfit(sd["irs_decay"], sd["_resid"], 1)
        xx = np.linspace(sd["irs_decay"].min(), sd["irs_decay"].max(), 30)
        ax.plot(xx, a + b * xx, color=col, lw=2.0, label=f"{lab}  slope={b:+.2f} (n={len(sd)})")
    ax.axhline(0, color="0.6", lw=0.7)
    ax.set_xlabel("chemical-aware irs_decay"); ax.set_ylabel("$R = D-\\hat A$")
    ax.set_title("RF residual $R$ vs chemical-aware IRS decay, by insecticide", fontsize=10, loc="left")
    ax.legend(fontsize=7.5, loc="upper right")
    fig.tight_layout(); fig.savefig(f"{OUT}/irs_chem_resid_scatter.png"); plt.close(fig)

    # ---- 3) faceted by district (top-6 spray-active), coloured by sector ----
    spr_per = (df[df["location"].isin(sprayed)].drop_duplicates("location")
               .groupby("district").size().sort_values(ascending=False))
    districts = spr_per.head(6).index.tolist()
    cmap = plt.get_cmap("tab20")
    fig, axes = plt.subplots(2, 3, figsize=(9.0, 5.2), sharex=True, sharey=True)
    axes = axes.ravel()
    for k, dist in enumerate(districts):
        ax = axes[k]; sd = d[d["district"] == dist]
        for i, loc in enumerate(sorted(sd["location"].unique())):
            sl_ = sd[sd["location"] == loc]
            ax.scatter(sl_["irs_decay"], sl_["_resid"], s=8, alpha=0.4, color=cmap(i % 20), edgecolors="none")
        x, yv = sd["irs_decay"].to_numpy(), sd["_resid"].to_numpy()
        if len(x) > 10 and np.ptp(x) > 0:
            b, a = np.polyfit(x, yv, 1)
            xx = np.linspace(x.min(), x.max(), 30); ax.plot(xx, a + b * xx, color="0.25", lw=1.3)
            ax.text(0.96, 0.04, f"slope={b:+.2f}", transform=ax.transAxes, fontsize=7, ha="right", color="0.25")
        ax.axhline(0, color="0.7", lw=0.6); ax.set_title(f"D{k+1} ({sd['location'].nunique()} sec)", fontsize=9, loc="left")
    axes[0].set_ylabel("$R$"); axes[3].set_ylabel("$R$")
    for k in (3, 4, 5):
        axes[k].set_xlabel("chemical-aware irs_decay")
    fig.suptitle("R vs chemical-aware IRS decay — faceted by district, coloured by sector", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96]); fig.savefig(f"{OUT}/irs_chem_resid_facet.png"); plt.close(fig)

    print("wrote 3 plots to", OUT)
    print("per-chemical pooled slope (R on chemical-decay):")
    for hl in (2.0, 3.0, 5.0, 8.0):
        sd = dp[np.isclose(dp["_active_hl"], hl)]
        if len(sd) >= 5:
            b = np.polyfit(sd["irs_decay"], sd["_resid"], 1)[0]
            print(f"  {CLASS[hl][0]:28s} slope={b:+.3f}  n={len(sd)}")


if __name__ == "__main__":
    main()
