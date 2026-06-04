"""MSTL → ARIMA base → RandomForest-on-residual hybrid.

The mirror of :mod:`mstl_multistep.arima_residual`: here **AutoARIMA is the
base** forecaster on the MSTL-deseasonalized series (it carries the
autoregressive dynamics *and* the predictive uncertainty), and a
**RandomForest models the ARIMA residual** as a deterministic point correction,
exploiting nonlinear / climate-anomaly structure the univariate ARIMA cannot.

Reconstruction (log1p space when ``log_transform``)::

    final = seasonal_naive + ARIMA_draws(mu_h, sigma_h) + RF_residual_point

ARIMA supplies the spread (``sigma_h`` grows with horizon); RF nudges the mean
using lagged climate anomalies and location identity. ARIMA's in-sample
residuals are well-behaved (unlike RF's, which is why the inverse mode needs
OOB), so no out-of-bag trick is required here — RF's forecast on the unseen
future feature rows is genuinely out-of-sample.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from statsforecast.models import AutoARIMA

from mstl_multistep import decomposition as dec
from mstl_multistep.features import INDEX_COLS, build_model_features, build_seasonal_features
from mstl_multistep.irs_features import build_irs_features
from mstl_multistep.io_utils import detect_frequency, period_to_timestamp
from mstl_multistep.run_config import RunConfig

logger = logging.getLogger(__name__)


class ArimaBaseRFResidualModel:
    def __init__(self, cfg: RunConfig, feature_columns: list[str]):
        self.cfg = cfg
        self.feature_columns = list(feature_columns)
        self._rf: RandomForestRegressor | None = None
        self._qmodels: dict | None = None  # {level: quantile GBM} for residual_quantile mode
        self._feat_cols: list[str] | None = None
        self._cov_cols: list[str] | None = None  # covariate/IRS/dummy features
        self._tgt_cols: list[str] = []           # deseasonalized-target lag features
        self._freq: str | None = None
        self._season_lengths: list[int] | None = None

    def _target_lag_frame(self, frame: pd.DataFrame, value_col: str, k: int) -> pd.DataFrame:
        """``[*INDEX_COLS, tgt_lag1..tgt_lagk]`` of ``value_col`` (D or ARIMA residual), per location."""
        g = frame.copy()
        g["_ts"] = g["time_period"].apply(lambda p: period_to_timestamp(p, self._freq))
        g = g.sort_values(["location", "_ts"])
        out = g[INDEX_COLS].copy()
        for j in range(1, k + 1):
            out[f"tgt_lag{j}"] = g.groupby("location")[value_col].shift(j).to_numpy()
        return out

    def _season_lengths_for(self, freq: str) -> list[int]:
        sl = (
            self.cfg.season_length_weekly
            if freq.startswith("W")
            else self.cfg.season_length_monthly
        )
        return [int(sl)]

    def _arima(self, y: np.ndarray, h: int, want_fitted: bool):
        """Return (fitted, mean, sigma) from AutoARIMA on a deseasonalized series.

        Cleans the input internally (interpolate + ffill/bfill) so AutoARIMA
        always sees a finite series. A series with no finite values at all (a
        location with no history yet in this window) gets a zero-mean baseline.
        Any ``fitted`` array is length ``len(y)`` so the caller can subtract it
        from the original (NaN-bearing) series to form residuals.
        """
        cfg = self.cfg
        h = max(int(h), 1)
        y = np.asarray(y, dtype=float)
        clean = pd.Series(y).interpolate(limit_direction="both").ffill().bfill().to_numpy()

        if not np.isfinite(clean).all():
            fitted = np.zeros(len(y)) if want_fitted else None
            return fitted, np.zeros(h), np.ones(h)

        z = norm.ppf(0.5 + cfg.arima_level / 200.0)
        try:
            model = AutoARIMA(
                season_length=1,  # the series is already deseasonalized
                approximation=cfg.arima_approximation,
                stepwise=cfg.arima_stepwise,
            )
            res = model.forecast(y=clean, h=h, level=[cfg.arima_level], fitted=want_fitted)
            fitted = np.asarray(res["fitted"], dtype=float) if want_fitted else None
            mean = np.asarray(res["mean"], dtype=float)
            lo = np.asarray(res[f"lo-{cfg.arima_level}"], dtype=float)
            hi = np.asarray(res[f"hi-{cfg.arima_level}"], dtype=float)
            sigma = (hi - lo) / (2.0 * z)
            sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-3)
            return fitted, mean, sigma
        except Exception as e:  # pragma: no cover - defensive
            logger.info("AutoARIMA failed (%s); using constant-mean fallback", type(e).__name__)
            mu = float(np.mean(clean))
            sd = float(np.std(clean)) or 1.0
            fitted = np.full(len(y), mu) if want_fitted else None
            return fitted, np.full(h, mu), np.full(h, sd)

    def fit(self, historic_df: pd.DataFrame) -> "ArimaBaseRFResidualModel":
        cfg = self.cfg
        self._freq = detect_frequency(historic_df)
        self._season_lengths = self._season_lengths_for(self._freq)
        if cfg.em_iterations and cfg.em_iterations > 1:
            return self._fit_em(historic_df)
        if cfg.rf_horizon_feature:
            return self._fit_horizon(historic_df)
        target = cfg.target_variable

        deseason, _ = dec.decompose_panel(
            historic_df, self._freq, target, self._season_lengths, cfg.log_transform
        )
        feats = build_model_features(
            historic_df,
            None,
            self.feature_columns,
            self._freq,
            self._season_lengths,
            cfg.feature_min_lag,
            cfg.feature_max_lag,
            cfg.use_location_dummies,
            cfg.deseasonalize_covariates,
            cfg.lags_by_col(),
        )
        if cfg.irs_column and cfg.irs_features:
            irs, irs_cols = build_irs_features(
                historic_df, None, cfg.irs_column, cfg.irs_features, cfg.irs_halflife
            )
            feats = feats.merge(irs, on=INDEX_COLS, how="left")
            for c in irs_cols:
                feats[c] = feats[c].fillna(0.0)
        if cfg.seasonal_fourier_order > 0:
            seas, _ = build_seasonal_features(historic_df, None, self._freq, cfg.seasonal_fourier_order)
            feats = feats.merge(seas, on=INDEX_COLS, how="left")
        self._cov_cols = [c for c in feats.columns if c not in INDEX_COLS]

        deseason = deseason.copy()
        deseason["_ts"] = deseason["time_period"].apply(
            lambda p: period_to_timestamp(p, self._freq)
        )

        # ARIMA residual = deseason - ARIMA in-sample fitted, per location.
        resid_blocks = []
        for loc, g in deseason.groupby("location", sort=False):
            g = g.sort_values("_ts")
            y = g[target].to_numpy(dtype=float)
            fitted, _, _ = self._arima(y, h=1, want_fitted=True)
            blk = g[INDEX_COLS].copy()
            blk["_resid"] = y - fitted  # keeps NaN where y was NaN
            resid_blocks.append(blk)
        resid_df = pd.concat(resid_blocks, ignore_index=True)

        # Target lags: of the deseasonalized target D, or of the ARIMA residual R.
        self._tgt_cols = []
        if cfg.rf_target_lags > 0:
            if cfg.rf_target_lag_source == "residual":
                tl = self._target_lag_frame(resid_df, "_resid", cfg.rf_target_lags)
            else:
                tl = self._target_lag_frame(deseason, target, cfg.rf_target_lags)
            feats = feats.merge(tl, on=INDEX_COLS, how="left")
            self._tgt_cols = [f"tgt_lag{j}" for j in range(1, cfg.rf_target_lags + 1)]
        self._feat_cols = self._cov_cols + self._tgt_cols

        design = feats.merge(resid_df, on=INDEX_COLS, how="left")
        X = design[self._feat_cols].to_numpy(dtype=float)
        yres = design["_resid"].to_numpy(dtype=float)
        mask = ~(np.isnan(X).any(axis=1) | np.isnan(yres))

        if cfg.residual_quantile:
            # Quantile GBMs on the ARIMA residual: one per level, for an asymmetric spread.
            self._qmodels = {}
            for q in cfg.residual_quantile_levels:
                m = HistGradientBoostingRegressor(
                    loss="quantile", quantile=q, max_iter=200, learning_rate=0.05,
                    max_depth=None, random_state=cfg.rf.random_state,
                )
                m.fit(X[mask], yres[mask])
                self._qmodels[float(q)] = m
            return self

        self._rf = RandomForestRegressor(
            n_estimators=cfg.rf.n_estimators,
            max_depth=cfg.rf.max_depth,
            min_samples_leaf=cfg.rf.min_samples_leaf,
            max_features=cfg.rf.max_features,
            random_state=cfg.rf.random_state,
            n_jobs=-1,
        )
        self._rf.fit(X[mask], yres[mask])
        return self

    def predict(self, historic_df: pd.DataFrame, future_df: pd.DataFrame) -> pd.DataFrame:
        if self._rf is None and self._qmodels is None:
            raise RuntimeError("Call fit() before predict()")
        cfg = self.cfg
        if cfg.em_iterations and cfg.em_iterations > 1:
            return self._predict_em(historic_df, future_df)
        if cfg.rf_horizon_feature:
            return self._predict_horizon(historic_df, future_df)
        freq = self._freq
        target = cfg.target_variable
        rng = np.random.default_rng(cfg.random_seed)

        deseason_hist, decomps = dec.decompose_panel(
            historic_df, freq, target, self._season_lengths, cfg.log_transform
        )
        deseason_hist = deseason_hist.copy()
        deseason_hist["_ts"] = deseason_hist["time_period"].apply(
            lambda p: period_to_timestamp(p, freq)
        )

        feats_all = build_model_features(
            historic_df,
            future_df,
            self.feature_columns,
            freq,
            self._season_lengths,
            cfg.feature_min_lag,
            cfg.feature_max_lag,
            cfg.use_location_dummies,
            cfg.deseasonalize_covariates,
            cfg.lags_by_col(),
        )
        if cfg.irs_column and cfg.irs_features:
            irs, _ = build_irs_features(
                historic_df, future_df, cfg.irs_column, cfg.irs_features, cfg.irs_halflife
            )
            feats_all = feats_all.merge(irs, on=INDEX_COLS, how="left")
        if cfg.seasonal_fourier_order > 0:
            seas, _ = build_seasonal_features(historic_df, future_df, freq, cfg.seasonal_fourier_order)
            feats_all = feats_all.merge(seas, on=INDEX_COLS, how="left")
        feats_all = feats_all.reindex(columns=INDEX_COLS + self._cov_cols, fill_value=0.0)
        feats_all["_ts"] = feats_all["time_period"].apply(lambda p: period_to_timestamp(p, freq))

        future = future_df.copy()
        future["_ts"] = future["time_period"].apply(lambda p: period_to_timestamp(p, freq))

        rows = []
        for loc, fg in future.groupby("location", sort=False):
            loc_str = str(loc)
            fg = fg.sort_values("_ts")
            times = fg["time_period"].astype(str).tolist()
            h = len(fg)

            y_hist = (
                deseason_hist[deseason_hist["location"].astype(str) == loc_str]
                .sort_values("_ts")[target]
                .to_numpy(dtype=float)
            )
            want_fitted = bool(self._tgt_cols) and cfg.rf_target_lag_source == "residual"
            fitted_h, mean_h, sigma_h = self._arima(y_hist, h=h, want_fitted=want_fitted)
            arima_draws = rng.normal(
                loc=mean_h[:, None], scale=sigma_h[:, None], size=(h, cfg.n_samples)
            )

            fut_feat = (
                feats_all[feats_all["location"].astype(str) == loc_str]
                .sort_values("_ts")
                .tail(h)[self._cov_cols]
                .to_numpy(dtype=float)
            )
            if self._tgt_cols:
                # Non-recursive bridge for future lags; tgt_lag{j} at step t = series[lh+t-j].
                if cfg.rf_target_lag_source == "residual":
                    # lag the ARIMA residual R = D - fitted; future residuals ~ 0.
                    r_hist = y_hist - fitted_h
                    s = np.concatenate([r_hist, np.zeros(h)])
                else:
                    # lag the deseasonalized target D; bridge future with the ARIMA mean.
                    s = np.concatenate([y_hist, mean_h])
                lh = len(y_hist)
                tl = np.empty((h, len(self._tgt_cols)), dtype=float)
                for t in range(h):
                    for ji, j in enumerate(range(1, len(self._tgt_cols) + 1)):
                        idx = lh + t - j
                        tl[t, ji] = s[idx] if idx >= 0 else np.nan
                fut_feat = np.hstack([fut_feat, tl])
            seasonal = dec.extrapolate_seasonal(decomps[loc_str], self._season_lengths, h)
            Xf = np.nan_to_num(fut_feat)

            if self._qmodels is not None:
                # Asymmetric residual: inverse-CDF interpolation of predicted quantiles,
                # added to the ARIMA mean; deviations from the median scale by sigma_h/sigma_1.
                levels = np.array(sorted(self._qmodels), dtype=float)
                Q = np.column_stack([self._qmodels[float(q)].predict(Xf) for q in levels])  # (h, L)
                Q.sort(axis=1)  # enforce monotone quantiles (no crossing)
                s1 = sigma_h[0] if sigma_h[0] > 0 else 1.0
                u = rng.random((h, cfg.n_samples))
                resid = np.empty((h, cfg.n_samples))
                for t in range(h):
                    med_t = np.interp(0.5, levels, Q[t])
                    draws = np.interp(u[t], levels, Q[t])
                    resid[t] = med_t + (draws - med_t) * (sigma_h[t] / s1)
                final = seasonal[:, None] + mean_h[:, None] + resid  # (h, n)
            else:
                rf_resid = self._rf.predict(Xf)  # (h,)
                final = seasonal[:, None] + arima_draws + rf_resid[:, None]  # (h, n)
            if cfg.log_transform:
                final = np.expm1(final)
            final = np.clip(final, a_min=0.0, a_max=None)
            if cfg.discretize_samples:
                final = rng.poisson(final).astype(float)

            for step in range(h):
                row = {"time_period": times[step], "location": loc_str}
                for i, v in enumerate(final[step]):
                    row[f"sample_{i}"] = float(v)
                rows.append(row)

        sample_cols = [f"sample_{i}" for i in range(cfg.n_samples)]
        return pd.DataFrame(rows)[INDEX_COLS + sample_cols]

    # -- Per-horizon residual variant (rf_horizon_feature) -----------------

    def _deseason_by_loc(self, df):
        """{loc: (times list, D array)} deseasonalized target, time-sorted."""
        target = self.cfg.target_variable
        deseason, decomps = dec.decompose_panel(
            df, self._freq, target, self._season_lengths, self.cfg.log_transform
        )
        deseason["_ts"] = deseason["time_period"].apply(lambda p: period_to_timestamp(p, self._freq))
        out = {}
        for loc, g in deseason.groupby("location", sort=False):
            g = g.sort_values("_ts")
            out[str(loc)] = (g["time_period"].astype(str).tolist(), g[target].to_numpy(dtype=float))
        return out, decomps

    def _fit_horizon(self, historic_df: pd.DataFrame) -> "ArimaBaseRFResidualModel":
        cfg = self.cfg
        if cfg.rf_target_lags:
            raise ValueError("rf_horizon_feature requires rf_target_lags=0")
        H = int(cfg.rf_horizon_max)

        feats = self._cov_features(historic_df, None)
        self._cov_cols = [c for c in feats.columns if c not in INDEX_COLS]
        self._tgt_cols = []
        self._feat_cols = self._cov_cols + ["_horizon"]
        fmap = {
            (str(l), str(t)): row
            for l, t, row in zip(
                feats["location"], feats["time_period"],
                feats[self._cov_cols].to_numpy(dtype=float),
            )
        }

        ser, _ = self._deseason_by_loc(historic_df)
        Xrows, yrows = [], []
        for loc, (times, D) in ser.items():
            T = len(D)
            valid = [o for o in range(24, T) if np.isfinite(D[:o]).sum() >= 24]
            for o in valid[-cfg.rf_horizon_origins:]:
                _, mean_h, _ = self._arima(D[:o], h=H, want_fitted=False)
                for hh in range(1, H + 1):
                    ti = o + hh - 1
                    if ti >= T or not np.isfinite(D[ti]):
                        continue
                    fr = fmap.get((loc, times[ti]))
                    if fr is None:
                        continue
                    Xrows.append(np.concatenate([fr, [float(hh)]]))
                    yrows.append(D[ti] - mean_h[hh - 1])
        X = np.asarray(Xrows, dtype=float)
        y = np.asarray(yrows, dtype=float)
        mask = ~np.isnan(X).any(axis=1) & np.isfinite(y)
        self._rf = RandomForestRegressor(
            n_estimators=cfg.rf.n_estimators, max_depth=cfg.rf.max_depth,
            min_samples_leaf=cfg.rf.min_samples_leaf, max_features=cfg.rf.max_features,
            random_state=cfg.rf.random_state, n_jobs=-1,
        )
        self._rf.fit(X[mask], y[mask])
        return self

    def _predict_horizon(self, historic_df: pd.DataFrame, future_df: pd.DataFrame) -> pd.DataFrame:
        cfg = self.cfg
        freq = self._freq
        rng = np.random.default_rng(cfg.random_seed)

        feats_all = self._cov_features(historic_df, future_df)
        feats_all = feats_all.reindex(columns=INDEX_COLS + self._cov_cols, fill_value=0.0)
        cmap = {
            (str(l), str(t)): row
            for l, t, row in zip(
                feats_all["location"], feats_all["time_period"],
                feats_all[self._cov_cols].to_numpy(dtype=float),
            )
        }
        ser, decomps = self._deseason_by_loc(historic_df)

        future = future_df.copy()
        future["_ts"] = future["time_period"].apply(lambda p: period_to_timestamp(p, freq))
        rows = []
        for loc, fg in future.groupby("location", sort=False):
            loc_str = str(loc)
            fg = fg.sort_values("_ts")
            times = fg["time_period"].astype(str).tolist()
            h = len(fg)
            y_hist = ser.get(loc_str, ([], np.array([])))[1]
            _, mean_h, sigma_h = self._arima(y_hist, h=h, want_fitted=False)
            seasonal = dec.extrapolate_seasonal(decomps[loc_str], self._season_lengths, h)
            arima_draws = rng.normal(mean_h[:, None], sigma_h[:, None], size=(h, cfg.n_samples))

            rf_resid = np.zeros(h)
            for step in range(h):
                fr = cmap.get((loc_str, times[step]))
                if fr is None:
                    continue
                x = np.concatenate([fr, [float(step + 1)]])
                rf_resid[step] = self._rf.predict(np.nan_to_num(x)[None, :])[0]

            final = seasonal[:, None] + arima_draws + rf_resid[:, None]
            if cfg.log_transform:
                final = np.expm1(final)
            final = np.clip(final, a_min=0.0, a_max=None)
            if cfg.discretize_samples:
                final = rng.poisson(final).astype(float)
            for step in range(h):
                row = {"time_period": times[step], "location": loc_str}
                for i, v in enumerate(final[step]):
                    row[f"sample_{i}"] = float(v)
                rows.append(row)

        sample_cols = [f"sample_{i}" for i in range(cfg.n_samples)]
        return pd.DataFrame(rows)[INDEX_COLS + sample_cols]

    # -- EM / backfitting variant (em_iterations > 1) ----------------------

    def _cov_features(self, historic_df, future_df):
        """Covariate + IRS feature frame (no target lags — EM keeps predict non-circular)."""
        cfg = self.cfg
        feats = build_model_features(
            historic_df, future_df, self.feature_columns, self._freq,
            self._season_lengths, cfg.feature_min_lag, cfg.feature_max_lag,
            cfg.use_location_dummies, cfg.deseasonalize_covariates, cfg.lags_by_col(),
        )
        if cfg.irs_column and cfg.irs_features:
            irs, irs_cols = build_irs_features(
                historic_df, future_df, cfg.irs_column, cfg.irs_features, cfg.irs_halflife
            )
            feats = feats.merge(irs, on=INDEX_COLS, how="left")
            for c in irs_cols:
                feats[c] = feats[c].fillna(0.0)
        if cfg.seasonal_fourier_order > 0:
            seas, _ = build_seasonal_features(historic_df, future_df, self._freq, cfg.seasonal_fourier_order)
            feats = feats.merge(seas, on=INDEX_COLS, how="left")
        return feats

    def _new_rf(self, oob: bool):
        cfg = self.cfg
        return RandomForestRegressor(
            n_estimators=cfg.rf.n_estimators, max_depth=cfg.rf.max_depth,
            min_samples_leaf=cfg.rf.min_samples_leaf, max_features=cfg.rf.max_features,
            random_state=cfg.rf.random_state, n_jobs=-1, bootstrap=True, oob_score=oob,
        )

    def _fit_em(self, historic_df: pd.DataFrame) -> "ArimaBaseRFResidualModel":
        cfg = self.cfg
        target = cfg.target_variable
        if cfg.rf_target_lags:
            raise ValueError("em_iterations>1 requires rf_target_lags=0")

        feats = self._cov_features(historic_df, None)
        self._cov_cols = [c for c in feats.columns if c not in INDEX_COLS]
        self._tgt_cols = []
        self._feat_cols = self._cov_cols

        base = historic_df[INDEX_COLS].copy()
        base["_L"] = np.where(
            cfg.log_transform,
            np.log1p(np.clip(pd.to_numeric(historic_df[target], errors="coerce"), 0, None)),
            pd.to_numeric(historic_df[target], errors="coerce"),
        )
        design = feats.merge(base, on=INDEX_COLS, how="left")
        design["_ts"] = design["time_period"].apply(lambda p: period_to_timestamp(p, self._freq))
        X = np.nan_to_num(design[self._cov_cols].to_numpy(dtype=float))
        Xnan = design[self._cov_cols].to_numpy(dtype=float)
        Lvec = design["_L"].to_numpy(dtype=float)
        n = len(design)

        loc_pos: dict[str, np.ndarray] = {}
        for loc, g in design.sort_values(["location", "_ts"]).groupby("location", sort=False):
            loc_pos[str(loc)] = g.index.to_numpy()

        C = np.zeros(n)
        rf = None
        K = int(cfg.em_iterations)
        for it in range(K):
            resid_target = np.full(n, np.nan)
            for loc, pos in loc_pos.items():
                Lc = Lvec[pos] - C[pos]
                decomp = dec.decompose(Lc, self._season_lengths)
                Dclean = Lc - dec.seasonal_component(decomp)
                A, _, _ = self._arima(Dclean, h=1, want_fitted=True)
                resid_target[pos] = (Dclean - A) + C[pos]   # = L - S - A
            mask = np.isfinite(resid_target) & ~np.isnan(Xnan).any(axis=1)
            last = it == K - 1
            rf = self._new_rf(oob=not last)
            rf.fit(X[mask], resid_target[mask])
            if not last:
                oob = np.full(n, np.nan)
                oob[np.where(mask)[0]] = rf.oob_prediction_
                newC = np.where(np.isfinite(oob), oob, rf.predict(X))
                C = cfg.em_damping * newC + (1.0 - cfg.em_damping) * C
        self._rf = rf
        return self

    def _predict_em(self, historic_df: pd.DataFrame, future_df: pd.DataFrame) -> pd.DataFrame:
        cfg = self.cfg
        target = cfg.target_variable
        rng = np.random.default_rng(cfg.random_seed)

        feats_all = self._cov_features(historic_df, future_df)
        feats_all = feats_all.reindex(columns=INDEX_COLS + self._cov_cols, fill_value=0.0)
        Call = self._rf.predict(np.nan_to_num(feats_all[self._cov_cols].to_numpy(dtype=float)))
        cmap = {
            (str(l), str(t)): c
            for l, t, c in zip(feats_all["location"], feats_all["time_period"], Call)
        }

        hist = historic_df.copy()
        hist["_ts"] = hist["time_period"].apply(lambda p: period_to_timestamp(p, self._freq))
        future = future_df.copy()
        future["_ts"] = future["time_period"].apply(lambda p: period_to_timestamp(p, self._freq))

        rows = []
        for loc, fg in future.groupby("location", sort=False):
            loc_str = str(loc)
            fg = fg.sort_values("_ts")
            times = fg["time_period"].astype(str).tolist()
            h = len(fg)
            hg = hist[hist["location"].astype(str) == loc_str].sort_values("_ts")
            y = pd.to_numeric(hg[target], errors="coerce").to_numpy(dtype=float)
            L = np.log1p(np.clip(y, 0, None)) if cfg.log_transform else y
            C_hist = np.array([cmap.get((loc_str, str(t)), 0.0) for t in hg["time_period"]])
            Lc = L - C_hist
            decomp = dec.decompose(Lc, self._season_lengths)
            Dclean = Lc - dec.seasonal_component(decomp)
            _, mean_h, sigma_h = self._arima(Dclean, h=h, want_fitted=False)
            sigma_h = sigma_h * cfg.em_sigma_scale
            seasonal = dec.extrapolate_seasonal(decomp, self._season_lengths, h)
            C_fut = np.array([cmap.get((loc_str, t), 0.0) for t in times])
            arima_draws = rng.normal(mean_h[:, None], sigma_h[:, None], size=(h, cfg.n_samples))

            final = seasonal[:, None] + arima_draws + C_fut[:, None]
            if cfg.log_transform:
                final = np.expm1(final)
            final = np.clip(final, a_min=0.0, a_max=None)
            if cfg.discretize_samples:
                final = rng.poisson(final).astype(float)
            for step in range(h):
                row = {"time_period": times[step], "location": loc_str}
                for i, v in enumerate(final[step]):
                    row[f"sample_{i}"] = float(v)
                rows.append(row)

        sample_cols = [f"sample_{i}" for i in range(cfg.n_samples)]
        return pd.DataFrame(rows)[INDEX_COLS + sample_cols]
