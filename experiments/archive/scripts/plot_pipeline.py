"""Generate per-step figures of the champion pipeline for single sectors, for the PDF.

Fits the champion (config_tgtlag3_var) once on a real training window (all locations), then:
  fig1_decomposition.pdf -- Step 1: log1p(cases) = seasonal S + deseasonalized D (MSTL)
  fig2_arima_rf.pdf      -- Steps 2-4: ARIMA fit/forecast on D, RF correction, variance head
  fig3_forecast.pdf      -- Step 5: reconstructed fan chart vs actual, TWO panels:
                            (a) a sector where the forecast tracks the actual,
                            (b) an explosive outbreak the forecast under-predicts.
Steps 1-4 are shown for the "works" sector. Single fit, two locations at the same origin.

Usage: uv run python scripts/plot_pipeline.py
"""
from __future__ import annotations
import os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from mstl_multistep import decomposition as dec
from mstl_multistep.pipeline import build_chap_model
from mstl_multistep.run_config import load_model_configuration
from mstl_multistep.io_utils import detect_frequency, period_to_timestamp

DATASET = "/Users/knutdr/Data/CH/chap_data_level5_irs_allocated_monthly.csv"
CONFIG = "config_tgtlag3_var.yaml"
TARGET = "disease_cases"
OUTDIR = "docs/figs"
H = 3
ORIGIN = 153  # 2025-10; forecast Oct-Dec 2025

C = dict(seasonal="#1f77b4", arima="#d62728", rf="#2ca44c", var="#9467bd",
         data="#222222", band="#d62728")
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 130, "savefig.bbox": "tight"})


def intermediates(model, df, train_df, loc, gtimes, freq, sl, cfg):
    """Run + capture every pipeline intermediate for one (loc, ORIGIN)."""
    sub = df[df["location"] == loc].copy()
    sub["_ts"] = sub["time_period"].apply(lambda p: period_to_timestamp(p, freq))
    sub = sub.sort_values("_ts")
    y_all = pd.to_numeric(sub[TARGET], errors="coerce").to_numpy(float)
    ts_all = sub["_ts"].to_numpy()
    fut_times = gtimes[ORIGIN:ORIGIN + H]

    hist_loc = train_df[train_df["location"] == loc].copy()
    fut_loc = df[(df["location"] == loc) & (df["time_period"].isin(fut_times))].copy()

    cap = {}
    orig = model._rf.predict
    model._rf.predict = lambda X: (cap.__setitem__("Xf", np.asarray(X)) or orig(X))
    wide = model.predict(hist_loc, fut_loc)
    model._rf.predict = orig

    deseason_hist, decomps = dec.decompose_panel(hist_loc, freq, TARGET, sl, cfg.log_transform)
    decomp = decomps[loc]
    L = decomp["data"].to_numpy(float); S = dec.seasonal_component(decomp); D = L - S
    fitted, mean_h, sigma_h = model._arima(D, h=H, want_fitted=True)
    g = model._rf.predict(cap["Xf"])
    v = model._tree_variance(model._rf, cap["Xf"]) if model._var_mode != "none" else np.zeros(H)
    sigma_eff = np.sqrt(sigma_h ** 2 + cfg.residual_variance_scale * v)
    sc = [c for c in wide.columns if c.startswith("sample_")]
    Smp = wide.sort_values("time_period")[sc].to_numpy(float)
    q = {p: np.quantile(Smp, p, axis=1) for p in (.1, .25, .5, .75, .9)}
    name = sub["location_name"].iloc[0] if "location_name" in sub else loc
    return dict(name=name, th=ts_all[:ORIGIN], tf=ts_all[ORIGIN:ORIGIN + H],
                y_hist=y_all[:ORIGIN], y_fut=y_all[ORIGIN:ORIGIN + H], L=L, S=S, D=D,
                fitted=fitted, mean_h=mean_h, sigma_h=sigma_h, sigma_eff=sigma_eff, g=g, q=q)


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    df = pd.read_csv(DATASET); df["location"] = df["location"].astype(str)
    freq = detect_frequency(df)
    gtimes = sorted(df["time_period"].unique(), key=lambda p: period_to_timestamp(p, freq))
    train_times = gtimes[:ORIGIN]
    train_df = df[df["time_period"].isin(train_times)].copy()

    mc = load_model_configuration(CONFIG); cfg = mc.user_option_values
    model = build_chap_model(cfg, mc.additional_continuous_covariates)
    print("fitting champion on training window ...")
    model.fit(train_df)
    sl = model._season_lengths

    # candidate sectors: enough history, non-trivial counts
    st = df.groupby("location")[TARGET].agg(["mean", "count"])
    cands = st[(st["count"] >= 130) & (st["mean"] >= 15)].index.tolist()
    print(f"scoring {len(cands)} candidate sectors at origin {gtimes[ORIGIN]} ...")

    rows = []
    for loc in cands:
        try:
            d = intermediates(model, df, train_df, loc, gtimes, freq, sl, cfg)
        except Exception:
            continue
        yf, q = d["y_fut"], d["q"]
        if not np.isfinite(yf).all():
            continue
        cov = float(np.mean((q[.1] <= yf) & (yf <= q[.9])))
        rel = float(np.median(np.abs(q[.5] - yf) / (yf + 1)))
        dyn = float(np.max(yf) - np.min(yf))
        ratio = float(np.max(yf) / max(np.max(q[.9]), 1))  # >1 => actual above band
        rows.append((loc, cov, rel, dyn, ratio, d))
    # "works": full coverage, low relative error, some dynamic range
    works = sorted([r for r in rows if r[1] >= 0.99 and r[3] >= 20], key=lambda r: r[2])
    # "miss": actual far above the 90% band (under-forecast outbreak)
    miss = sorted(rows, key=lambda r: -r[4])
    dw = works[0][5] if works else sorted(rows, key=lambda r: r[2])[0][5]
    dm = miss[0][5]
    print(f"works = {dw['name']}  (cov ok, rel={works[0][2]:.2f})" if works else "works=fallback")
    print(f"miss  = {dm['name']}  (actual/band={miss[0][4]:.1f}x)")
    for tag, d in (("works", dw), ("miss", dm)):
        print(f"  [{tag}] {d['name']}: median={np.round(d['q'][.5],0)} actual={d['y_fut']}")

    nb = 54
    # ===================== Figure 1: MSTL decomposition (works) =====================
    d = dw
    fig, ax = plt.subplots(3, 1, figsize=(6.6, 4.6), sharex=True)
    ax[0].plot(d["th"], d["L"], color=C["data"], lw=1.1); ax[0].set_ylabel("log1p(cases)\n$L$")
    ax[0].set_title(f"Step 1 — MSTL decomposition  (sector {d['name']})", fontsize=10, loc="left")
    ax[1].plot(d["th"], d["S"], color=C["seasonal"], lw=1.1); ax[1].set_ylabel("seasonal\n$S$")
    ax[2].plot(d["th"], d["D"], color=C["rf"], lw=1.1); ax[2].set_ylabel("deseasonalized\n$D=L-S$")
    for a in ax: a.margins(x=0.01)
    fig.tight_layout(); fig.savefig(f"{OUTDIR}/fig1_decomposition.pdf"); plt.close(fig)

    # ============ Figure 2: ARIMA base + RF correction + variance head (works) ============
    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    ax.plot(d["th"][-nb:], d["D"][-nb:], color=C["rf"], lw=1.2, label="$D$ (deseasonalized)")
    ax.plot(d["th"][-nb:], d["fitted"][-nb:], color=C["arima"], lw=1.0, ls="--",
            label="ARIMA in-sample fit $\\hat A$")
    ax.axvline(d["tf"][0], color="0.6", lw=0.8, ls=":")
    ax.fill_between(d["tf"], d["mean_h"] - d["sigma_eff"], d["mean_h"] + d["sigma_eff"],
                    color=C["var"], alpha=0.18, label="$\\pm\\sigma_{\\mathrm{eff}}$ (variance head)")
    ax.fill_between(d["tf"], d["mean_h"] - d["sigma_h"], d["mean_h"] + d["sigma_h"],
                    color=C["arima"], alpha=0.18, label="$\\pm\\sigma_{\\mathrm{ARIMA}}$")
    ax.plot(d["tf"], d["mean_h"], color=C["arima"], lw=1.4, marker="o", ms=3, label="ARIMA mean $\\mu_h$")
    ax.plot(d["tf"], d["mean_h"] + d["g"], color=C["rf"], lw=1.4, marker="s", ms=3,
            label="$\\mu_h+g(x)$ (RF-corrected)")
    ax.set_ylabel("deseasonalized (log1p)")
    ax.set_title(f"Steps 2–4 — ARIMA base, RF residual correction, variance head  (sector {d['name']})",
                 fontsize=10, loc="left")
    ax.legend(fontsize=7, ncol=2, loc="best", framealpha=0.9); ax.margins(x=0.01)
    fig.tight_layout(); fig.savefig(f"{OUTDIR}/fig2_arima_rf.pdf"); plt.close(fig)

    # ============ Figure 3: forecast vs actual — works | miss ============
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.1))
    for ax, d, lab in ((axes[0], dw, "(a) forecast tracks actual"),
                       (axes[1], dm, "(b) outbreak — forecast misses")):
        q = d["q"]
        ax.plot(d["th"][-nb:], d["y_hist"][-nb:], color=C["data"], lw=1.1, label="observed")
        ax.fill_between(d["tf"], q[.1], q[.9], color=C["band"], alpha=0.15, label="forecast 10–90%")
        ax.fill_between(d["tf"], q[.25], q[.75], color=C["band"], alpha=0.30, label="forecast 25–75%")
        ax.plot(d["tf"], q[.5], color=C["band"], lw=1.3, marker="o", ms=3, label="forecast median")
        ax.plot(np.concatenate([[d["th"][-1]], d["tf"]]),
                np.concatenate([[d["y_hist"][-1]], d["y_fut"]]),
                color=C["data"], lw=1.1, ls="--", marker="D", ms=4, label="actual (held out)")
        ax.axvline(d["tf"][0], color="0.6", lw=0.8, ls=":")
        ax.set_title(f"{lab}\nsector {d['name']}", fontsize=8.5, loc="left")
        ax.margins(x=0.02)
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.tick_params(axis="x", labelsize=7)
    axes[0].set_ylabel("disease cases")
    axes[0].legend(fontsize=6.5, loc="upper left", framealpha=0.9)
    fig.suptitle("Step 5 — reconstructed probabilistic forecast vs. actual", fontsize=10, x=0.02, ha="left")
    fig.tight_layout(); fig.savefig(f"{OUTDIR}/fig3_forecast.pdf"); plt.close(fig)
    print("wrote 3 figures to", OUTDIR)


if __name__ == "__main__":
    main()
