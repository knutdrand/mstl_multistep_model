"""MSTL → ARIMA base → RandomForest-on-residual hybrid.

A fixed-order ARIMA (shared across all locations) is the base forecaster on the
MSTL-deseasonalized series — it carries the autoregressive dynamics *and* the
predictive uncertainty — and a pooled RandomForest models the ARIMA residual as a
deterministic point correction, exploiting nonlinear / climate-anomaly structure the
univariate ARIMA cannot. A location-scale variance head widens the spread by the
forest's own (inter-tree) uncertainty.

Reconstruction (always in log1p space)::

    final = seasonal_naive + draws(mu_h, sigma_eff) + RF_residual_point
    sigma_eff^2 = sigma_ARIMA(h)^2 + residual_variance_scale * v(x)

See docs/champion_model.pdf.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.ensemble import RandomForestRegressor
from statsforecast.models import ARIMA

from mstl_multistep import decomposition as dec
from mstl_multistep.features import INDEX_COLS, build_model_features
from mstl_multistep.irs_features import build_irs_features
from mstl_multistep.io_utils import detect_frequency, period_to_timestamp
from mstl_multistep.run_config import RunConfig

logger = logging.getLogger(__name__)


class ArimaBaseRFResidualModel:
    def __init__(self, cfg: RunConfig, feature_columns: list[str]):
        self.cfg = cfg
        self.feature_columns = list(feature_columns)
        # Every covariate is MSTL-deseasonalized before lagging, except the IRS columns
        # (handled by the IRS feature bank, not the generic lag path).
        irs_cols = {c for c in (cfg.irs_column, cfg.irs_chemical_column) if c}
        self._deseason_cols = [c for c in self.feature_columns if c not in irs_cols]
        self._rf: RandomForestRegressor | None = None
        self._var_mode: str = "none"      # cached cfg.residual_variance for predict
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

    def _fill_missing_target(self, historic_df: pd.DataFrame) -> pd.DataFrame:
        """Treat NaN target as 0 reported cases when ``cfg.treat_missing_as_zero`` is set.

        Returns the frame unchanged (no copy) when the flag is off, so the default path —
        and the monthly champion — is untouched.
        """
        if not self.cfg.treat_missing_as_zero:
            return historic_df
        target = self.cfg.target_variable
        out = historic_df.copy()
        out[target] = pd.to_numeric(out[target], errors="coerce").fillna(0.0)
        return out

    def _season_lengths_for(self, freq: str) -> list[int]:
        sl = (
            self.cfg.season_length_weekly
            if freq.startswith("W")
            else self.cfg.season_length_monthly
        )
        return [int(sl)]

    def _arima(self, y: np.ndarray, h: int, want_fitted: bool):
        """Return (fitted, mean, sigma) from a fixed-order ARIMA on a deseasonalized series.

        All locations share ``arima_order`` (only coefficients are fit per series). Cleans the
        input internally (interpolate + ffill/bfill) so ARIMA always sees a finite series; a
        series with no finite values at all gets a zero-mean baseline. Any ``fitted`` array is
        length ``len(y)`` so the caller can subtract it to form residuals.
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
            model = ARIMA(order=tuple(int(o) for o in cfg.arima_order), season_length=1)
            res = model.forecast(y=clean, h=h, level=[cfg.arima_level], fitted=want_fitted)
            fitted = np.asarray(res["fitted"], dtype=float) if want_fitted else None
            mean = np.asarray(res["mean"], dtype=float)
            lo = np.asarray(res[f"lo-{cfg.arima_level}"], dtype=float)
            hi = np.asarray(res[f"hi-{cfg.arima_level}"], dtype=float)
            sigma = (hi - lo) / (2.0 * z)
            sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-3)
            return fitted, mean, sigma
        except Exception as e:  # pragma: no cover - defensive
            logger.info("ARIMA failed (%s); using constant-mean fallback", type(e).__name__)
            mu = float(np.mean(clean))
            sd = float(np.std(clean)) or 1.0
            fitted = np.full(len(y), mu) if want_fitted else None
            return fitted, np.full(h, mu), np.full(h, sd)

    def fit(self, historic_df: pd.DataFrame) -> "ArimaBaseRFResidualModel":
        cfg = self.cfg
        self._freq = detect_frequency(historic_df)
        historic_df = self._fill_missing_target(historic_df)
        self._season_lengths = self._season_lengths_for(self._freq)
        target = cfg.target_variable

        deseason, _ = dec.decompose_panel(
            historic_df, self._freq, target, self._season_lengths
        )
        feats = build_model_features(
            historic_df, None, self.feature_columns, self._freq, self._season_lengths,
            cfg.feature_min_lag, cfg.feature_max_lag, cfg.use_location_dummies,
            self._deseason_cols, cfg.lags_by_col(),
        )
        if cfg.irs_column and cfg.irs_features:
            irs, irs_cols = build_irs_features(
                historic_df, None, cfg.irs_column, cfg.irs_features, cfg.irs_halflife,
                chem_column=cfg.irs_chemical_column,
            )
            feats = feats.merge(irs, on=INDEX_COLS, how="left")
            for c in irs_cols:
                feats[c] = feats[c].fillna(0.0)
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

        # Lagged deseasonalized target fed to the RF (future lags bridged with the ARIMA mean).
        self._tgt_cols = []
        if cfg.rf_target_lags > 0:
            tl = self._target_lag_frame(deseason, target, cfg.rf_target_lags)
            feats = feats.merge(tl, on=INDEX_COLS, how="left")
            self._tgt_cols = [f"tgt_lag{j}" for j in range(1, cfg.rf_target_lags + 1)]
        self._feat_cols = self._cov_cols + self._tgt_cols

        design = feats.merge(resid_df, on=INDEX_COLS, how="left")
        X = design[self._feat_cols].to_numpy(dtype=float)
        yres = design["_resid"].to_numpy(dtype=float)
        mask = ~(np.isnan(X).any(axis=1) | np.isnan(yres))
        self._fit_mean_and_variance(X[mask], yres[mask])
        return self

    @staticmethod
    def _tree_variance(rf: RandomForestRegressor, X: np.ndarray) -> np.ndarray:
        """Inter-tree variance of the RF mean estimate — epistemic uncertainty per row."""
        preds = np.stack([est.predict(X) for est in rf.estimators_], axis=0)
        return preds.var(axis=0)

    def _fit_mean_and_variance(self, Xm: np.ndarray, ym: np.ndarray) -> None:
        """Fit the pooled mean RF on the ARIMA residual.

        With ``residual_variance="tree"`` the forest is fit out-of-bag so its inter-tree
        variance v(x) can widen the predictive spread in predict; ``"none"`` skips that.
        """
        cfg = self.cfg
        self._var_mode = cfg.residual_variance
        rf = RandomForestRegressor(
            n_estimators=cfg.rf.n_estimators,
            max_depth=cfg.rf.max_depth,
            min_samples_leaf=cfg.rf.min_samples_leaf,
            max_features=cfg.rf.max_features,
            random_state=cfg.rf.random_state,
            n_jobs=-1,
            oob_score=(self._var_mode != "none"),
        )
        rf.fit(Xm, ym)
        self._rf = rf

    def _residual_variance(self, Xf: np.ndarray) -> np.ndarray:
        """Inter-tree variance v(x) of the RF correction on forecast rows."""
        return self._tree_variance(self._rf, Xf)

    def predict(self, historic_df: pd.DataFrame, future_df: pd.DataFrame) -> pd.DataFrame:
        if self._rf is None:
            raise RuntimeError("Call fit() before predict()")
        cfg = self.cfg
        freq = self._freq
        target = cfg.target_variable
        rng = np.random.default_rng(cfg.random_seed)
        historic_df = self._fill_missing_target(historic_df)

        deseason_hist, decomps = dec.decompose_panel(
            historic_df, freq, target, self._season_lengths
        )
        deseason_hist = deseason_hist.copy()
        deseason_hist["_ts"] = deseason_hist["time_period"].apply(
            lambda p: period_to_timestamp(p, freq)
        )

        feats_all = build_model_features(
            historic_df, future_df, self.feature_columns, freq, self._season_lengths,
            cfg.feature_min_lag, cfg.feature_max_lag, cfg.use_location_dummies,
            self._deseason_cols, cfg.lags_by_col(),
        )
        if cfg.irs_column and cfg.irs_features:
            irs, _ = build_irs_features(
                historic_df, future_df, cfg.irs_column, cfg.irs_features, cfg.irs_halflife,
                chem_column=cfg.irs_chemical_column,
            )
            feats_all = feats_all.merge(irs, on=INDEX_COLS, how="left")
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
            _, mean_h, sigma_h = self._arima(y_hist, h=h, want_fitted=False)
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
                # Lag the deseasonalized target D; bridge future steps with the ARIMA mean
                # (non-recursive). tgt_lag{j} at step t = series[lh+t-j].
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

            rf_resid = self._rf.predict(Xf)  # (h,)
            if self._var_mode != "none":
                # Honest predictive variance: ARIMA spread + variance of the RF correction.
                v = self._residual_variance(Xf)  # (h,)
                sigma_eff = np.sqrt(sigma_h ** 2 + cfg.residual_variance_scale * v)
                draws = rng.normal(
                    loc=mean_h[:, None], scale=sigma_eff[:, None], size=(h, cfg.n_samples)
                )
                final = seasonal[:, None] + draws + rf_resid[:, None]  # (h, n)
            else:
                final = seasonal[:, None] + arima_draws + rf_resid[:, None]  # (h, n)
            final = np.expm1(final)
            final = np.clip(final, a_min=0.0, a_max=None)

            for step in range(h):
                row = {"time_period": times[step], "location": loc_str}
                for i, v in enumerate(final[step]):
                    row[f"sample_{i}"] = float(v)
                rows.append(row)

        sample_cols = [f"sample_{i}" for i in range(cfg.n_samples)]
        return pd.DataFrame(rows)[INDEX_COLS + sample_cols]

