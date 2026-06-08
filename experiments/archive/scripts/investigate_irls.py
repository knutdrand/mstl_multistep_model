"""Investigate WHY the variance head regresses under IRLS iterations (residual_variance_iterations>1).

Champion = config_tgtlag3_var (residual_variance=tree, scale=0.5, iterations=1). The IRLS loop
refits the mean RF weighted by 1/v and re-estimates v. We instrument that loop to test the
hypothesis: reweighting by 1/v downweights the high-residual (outbreak) points, biasing the
mean correction toward easy points -- exactly the points log-CRPS rewards.

Captures per iteration: weight-vs-|R| relationship, OOB MSE overall vs on the top-decile |R|
points, and the change in the mean correction. Then scores held-out log-CRPS/CRPS/coverage at
iterations 1/2/3, split by low/high truth.

Usage: uv run python scripts/investigate_irls.py [--n-locations 80] [--n-splits 3]
"""
from __future__ import annotations
import argparse, time
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from mstl_multistep.rf_residual import ArimaBaseRFResidualModel
from mstl_multistep.run_config import load_model_configuration
from mstl_multistep.io_utils import detect_frequency, period_to_timestamp

DATASET = "/Users/knutdr/Data/CH/chap_data_level5_irs_allocated_monthly.csv"
BASE = "config_tgtlag3_var.yaml"
TARGET = "disease_cases"


class Instrumented(ArimaBaseRFResidualModel):
    """Override the IRLS fit to record per-iteration diagnostics into self.diag."""

    def _fit_mean_and_variance(self, Xm, ym):
        cfg = self.cfg
        mode = cfg.residual_variance
        self._var_mode = mode
        self._varmodel = None
        iters = max(1, int(cfg.residual_variance_iterations))
        self.diag = []
        weight = None
        prev_pred = None
        absR = np.abs(ym)
        hi = absR >= np.quantile(absR, 0.90)   # top-decile |R| = the big residuals/outbreaks
        for it in range(iters):
            rf = RandomForestRegressor(
                n_estimators=cfg.rf.n_estimators, max_depth=cfg.rf.max_depth,
                min_samples_leaf=cfg.rf.min_samples_leaf, max_features=cfg.rf.max_features,
                random_state=cfg.rf.random_state, n_jobs=-1, oob_score=(mode != "none"),
            )
            rf.fit(Xm, ym, sample_weight=weight)
            self._rf = rf
            pred = rf.predict(Xm)
            oob = rf.oob_prediction_ if mode != "none" else pred
            ok = np.isfinite(oob)
            e2 = np.where(ok, (ym - oob) ** 2, np.nan)
            rec = {
                "it": it,
                "oob_mse_all": float(np.nanmean(e2)),
                "oob_mse_hi": float(np.nanmean(e2[hi])),   # error on big-residual points
                "oob_mse_lo": float(np.nanmean(e2[~hi])),
                "mean_abs_corr": float(np.mean(np.abs(pred))),          # size of the correction
                "mean_abs_corr_hi": float(np.mean(np.abs(pred[hi]))),  # correction size on outbreaks
                "delta_pred": float(np.mean(np.abs(pred - prev_pred))) if prev_pred is not None else 0.0,
            }
            # weight used to FIT THIS iteration (None on it=0)
            if weight is not None:
                rec["w_vs_absR_corr"] = float(np.corrcoef(weight, absR)[0, 1])
                rec["w_hi/w_lo"] = float(weight[hi].mean() / weight[~hi].mean())
            self.diag.append(rec)
            prev_pred = pred
            if mode == "model":
                vm = HistGradientBoostingRegressor(loss="squared_error", max_iter=200,
                                                   learning_rate=0.05, random_state=cfg.rf.random_state)
                m = ok & np.isfinite(e2)
                vm.fit(Xm[m], np.log(e2[m] + 1e-6)); self._varmodel = vm
                vtrain = np.exp(vm.predict(Xm))
            else:
                vtrain = self._tree_variance(rf, Xm)
            if it < iters - 1:
                weight = 1.0 / np.clip(vtrain, 1e-6, None)


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-locations", type=int, default=80)
    ap.add_argument("--n-splits", type=int, default=3)
    args = ap.parse_args()
    df = pd.read_csv(DATASET); df["location"] = df["location"].astype(str)
    step = max(1, df["location"].nunique() // args.n_locations)
    df = df[df["location"].isin(sorted(df["location"].unique())[::step])].copy()
    freq = detect_frequency(df)
    times = sorted(df["time_period"].unique(), key=lambda p: period_to_timestamp(p, freq))
    splits = list(rolling(times, args.n_splits))
    base = load_model_configuration(BASE); cov = base.additional_continuous_covariates
    print(f"locations={df.location.nunique()} splits={len(splits)}\n")

    # threshold for "high truth" outbreak points (eval split), from full data
    hi_thr = float(np.nanquantile(pd.to_numeric(df[TARGET], errors="coerce"), 0.75))

    for iters in (1, 2, 3):
        cfg = base.user_option_values.model_copy(
            update={"residual_variance": "tree", "residual_variance_scale": 0.5,
                    "residual_variance_iterations": iters})
        lc, cr, cov_, lc_hi, lc_lo = [], [], [], [], []
        last_diag = None
        t0 = time.time()
        for tr, fu in splits:
            h = df[df.time_period.isin(tr)]; f = df[df.time_period.isin(fu)]
            if f.empty: continue
            m = Instrumented(cfg, cov); m.fit(h); last_diag = m.diag
            p = m.predict(h, f)
            sc = [c for c in p.columns if c.startswith("sample_")]
            mg = p.merge(f[["time_period", "location", TARGET]].astype({"location": str}),
                         on=["time_period", "location"], how="inner")
            for _, r in mg.iterrows():
                y = pd.to_numeric(r[TARGET], errors="coerce")
                if not np.isfinite(y): continue
                s = r[sc].to_numpy(float)
                lcl = crps(np.log1p(np.clip(s, 0, None)), float(np.log1p(max(y, 0))))
                lc.append(lcl); cr.append(crps(s, float(y)))
                ql, qh = np.quantile(s, [.1, .9]); cov_.append(float(ql <= y <= qh))
                (lc_hi if y >= hi_thr else lc_lo).append(lcl)
        agg = lambda a: float(np.nanmean(a)) if a else np.nan
        print(f"=== iterations={iters} ===  ({time.time()-t0:.0f}s)")
        print(f"  log-CRPS={agg(lc):.4f}  CRPS={agg(cr):.2f}  cov={agg(cov_):.3f}"
              f"   log-CRPS[hi y]={agg(lc_hi):.4f}  [lo y]={agg(lc_lo):.4f}")
        for d in (last_diag or []):
            extra = ""
            if "w_vs_absR_corr" in d:
                extra = f"  w~|R|corr={d['w_vs_absR_corr']:+.3f}  w_hi/w_lo={d['w_hi/w_lo']:.3f}"
            print(f"    it{d['it']}: oob_mse all={d['oob_mse_all']:.4f} hi={d['oob_mse_hi']:.4f} "
                  f"lo={d['oob_mse_lo']:.4f} | |corr|={d['mean_abs_corr']:.4f} "
                  f"|corr|hi={d['mean_abs_corr_hi']:.4f} dpred={d['delta_pred']:.4f}{extra}")
        print()


if __name__ == "__main__":
    main()
