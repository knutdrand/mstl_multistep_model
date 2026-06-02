"""MSTL → RandomForest → ARIMA-residual hybrid.

The inverse of chap_nixtla's ``mstl_arima_residual`` (which uses ARIMA as the
base and a neural net on the residuals): here a RandomForest is the
*deterministic point* model on the MSTL-deseasonalized series, and AutoARIMA
on the RF's **out-of-bag** residuals supplies the predictive uncertainty.

Why OOB residuals: an RF fits its training rows almost perfectly, so naive
in-sample residuals are far too small and ARIMA would learn a near-zero
variance — leaving the same under-dispersion we set out to fix. Out-of-bag
predictions are honest held-out estimates, so their residuals carry the real
forecast error for ARIMA to model.

Forecast reconstruction (all in log1p space when ``log_transform``)::

    final = seasonal_naive + RF_point(recursive) + ARIMA_residual_samples

where ARIMA contributes ``N(mu_h, sigma_h)`` draws whose sigma grows with the
horizon, and the seasonal/trend pieces are deterministic.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.ensemble import RandomForestRegressor
from statsforecast.models import AutoARIMA

from mstl_multistep import decomposition as dec
from mstl_multistep.features import INDEX_COLS, build_model_features
from mstl_multistep.io_utils import detect_frequency, period_to_timestamp
from mstl_multistep.run_config import RunConfig

logger = logging.getLogger(__name__)


def _lag_names(n_lags: int) -> list[str]:
    """Target-lag feature names, ``tlag_1`` (most recent) .. ``tlag_n``."""
    return [f"tlag_{k}" for k in range(1, n_lags + 1)]


class MSTLArimaResidualModel:
    def __init__(self, cfg: RunConfig, feature_columns: list[str]):
        self.cfg = cfg
        self.feature_columns = list(feature_columns)
        self._rf: RandomForestRegressor | None = None
        self._feat_cols: list[str] | None = None  # covariate/dummy columns (fixed order)
        self._order: list[str] | None = None       # full design order = feat_cols + lag names
        self._resid: dict[str, np.ndarray] = {}     # loc -> residual series (time order)
        self._resid_pool: np.ndarray | None = None  # global one-step OOB residuals (recursive mode)
        self._freq: str | None = None
        self._season_lengths: list[int] | None = None

    # -- internals ---------------------------------------------------------

    def _season_lengths_for(self, freq: str) -> list[int]:
        sl = (
            self.cfg.season_length_weekly
            if freq.startswith("W")
            else self.cfg.season_length_monthly
        )
        return [int(sl)]

    def _design(self, deseason: pd.DataFrame, feats: pd.DataFrame):
        """Build the pooled one-step design frame.

        Returns a frame sorted by (location, _ts) with the index columns,
        ``_ts``, the target column, the covariate/dummy feature columns and
        the target-lag columns. Lagging is done per location.
        """
        cfg = self.cfg
        target = cfg.target_variable
        L = cfg.n_target_lags

        df = deseason.merge(feats, on=INDEX_COLS, how="left")
        df["_ts"] = df["time_period"].apply(lambda p: period_to_timestamp(p, self._freq))
        df = df.sort_values(["location", "_ts"]).reset_index(drop=True)

        lag_cols = _lag_names(L)
        for k in range(1, L + 1):
            df[f"tlag_{k}"] = df.groupby("location")[target].shift(k)
        return df, lag_cols

    # -- public API --------------------------------------------------------

    def fit(self, historic_df: pd.DataFrame) -> "MSTLArimaResidualModel":
        cfg = self.cfg
        self._freq = detect_frequency(historic_df)
        self._season_lengths = self._season_lengths_for(self._freq)
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
        )
        self._feat_cols = [c for c in feats.columns if c not in INDEX_COLS]

        design, lag_cols = self._design(deseason, feats)
        self._order = self._feat_cols + lag_cols

        X = design[self._order].to_numpy(dtype=float)
        y = design[target].to_numpy(dtype=float)
        mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X_fit, y_fit = X[mask], y[mask]

        rf = RandomForestRegressor(
            n_estimators=cfg.rf.n_estimators,
            max_depth=cfg.rf.max_depth,
            min_samples_leaf=cfg.rf.min_samples_leaf,
            max_features=cfg.rf.max_features,
            random_state=cfg.rf.random_state,
            bootstrap=True,
            oob_score=True,
            n_jobs=-1,
        )
        rf.fit(X_fit, y_fit)
        self._rf = rf

        # Out-of-bag residuals (honest held-out error). Samples never left OOB
        # get NaN -> filled per-series by interpolation below.
        oob = np.full(len(y), np.nan)
        oob[mask] = rf.oob_prediction_
        design = design.copy()
        design["_resid"] = design[target].to_numpy() - oob

        # Global pool of honest one-step residuals (used by recursive_residual).
        self._resid_pool = (design["_resid"].to_numpy(dtype=float))
        self._resid_pool = self._resid_pool[np.isfinite(self._resid_pool)]

        for loc, g in design.groupby("location", sort=False):
            g = g.sort_values("_ts")
            r = pd.Series(g["_resid"].to_numpy(), dtype=float)
            r = r.interpolate(limit_direction="both")
            self._resid[str(loc)] = r.dropna().to_numpy(dtype=float)
        return self

    def _recursive_forecast_samples(
        self,
        seed_lags: np.ndarray,
        future_feats: np.ndarray,
        h: int,
        n: int,
        rng,
    ) -> np.ndarray:
        """Recursive RF forecast with one-step OOB residuals resampled per step.

        Each of the ``n`` paths feeds its own noisy prediction back into the lag
        window, so the residual variance compounds with horizon. Returns the
        deseasonalized draws, shape ``(h, n)`` (point + accumulated noise).
        """
        L = self.cfg.n_target_lags
        pool = self._resid_pool if self._resid_pool is not None and len(self._resid_pool) else np.array([0.0])
        recent = np.tile(seed_lags[-L:].astype(float), (n, 1))  # (n, L) oldest..newest
        out = np.empty((h, n))
        for step in range(h):
            # design lag order is tlag_1 (newest) .. tlag_L (oldest) == recent reversed
            lag_block = recent[:, ::-1]
            X = np.column_stack([np.tile(future_feats[step], (n, 1)), lag_block])
            point = self._rf.predict(X)
            eps = rng.choice(pool, size=n, replace=True)
            y = point + eps
            out[step] = y
            recent = np.column_stack([recent[:, 1:], y[:, None]])
        return out

    def _rf_point_forecast(
        self,
        loc: str,
        seed_lags: np.ndarray,
        future_feats: np.ndarray,
        h: int,
    ) -> np.ndarray:
        """Recursive deterministic RF forecast of the deseasonalized series.

        ``seed_lags`` holds the last ``n_target_lags`` deseasonalized values
        (oldest first); ``future_feats`` is ``(h, len(feat_cols))``.
        """
        L = self.cfg.n_target_lags
        recent = list(seed_lags[-L:].astype(float))
        preds = np.empty(h)
        for step in range(h):
            # lag_k = recent[-k] (tlag_1 most recent .. tlag_L oldest)
            lags = [recent[-k] for k in range(1, L + 1)]
            row = np.concatenate([future_feats[step], np.asarray(lags, dtype=float)])
            p = float(self._rf.predict(row.reshape(1, -1))[0])
            preds[step] = p
            recent.append(p)
            recent = recent[-L:]
        return preds

    def _arima_forecast(self, loc: str, h: int, rng):
        """AutoARIMA on the residual series -> (h, n_samples) residual draws."""
        cfg = self.cfg
        n = cfg.n_samples
        resid = self._resid.get(loc)
        if resid is None or len(resid) < 5 or not np.isfinite(resid).all():
            sd = float(np.nanstd(resid)) if resid is not None and len(resid) else 1.0
            sd = sd if np.isfinite(sd) and sd > 0 else 1.0
            return rng.normal(0.0, sd, size=(h, n))

        z = norm.ppf(0.5 + cfg.arima_level / 200.0)
        try:
            model = AutoARIMA(
                season_length=1,  # residuals are already deseasonalized
                approximation=cfg.arima_approximation,
                stepwise=cfg.arima_stepwise,
            )
            res = model.forecast(y=resid, h=h, level=[cfg.arima_level])
            mean_arr = np.asarray(res["mean"], dtype=float)
            lo = np.asarray(res[f"lo-{cfg.arima_level}"], dtype=float)
            hi = np.asarray(res[f"hi-{cfg.arima_level}"], dtype=float)
            sigma = (hi - lo) / (2.0 * z)
            sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-3)
        except Exception as e:  # pragma: no cover - defensive
            logger.info("AutoARIMA failed for %s (%s); using residual std", loc, type(e).__name__)
            sd = float(np.std(resid)) or 1.0
            mean_arr = np.zeros(h)
            sigma = np.full(h, sd)

        draws = rng.normal(
            loc=mean_arr[:, None], scale=sigma[:, None], size=(h, n)
        )
        return draws

    def predict(self, historic_df: pd.DataFrame, future_df: pd.DataFrame) -> pd.DataFrame:
        if self._rf is None:
            raise RuntimeError("Call fit() before predict()")
        cfg = self.cfg
        freq = self._freq
        target = cfg.target_variable
        rng = np.random.default_rng(cfg.random_seed)

        deseason_hist, decomps = dec.decompose_panel(
            historic_df, freq, target, self._season_lengths, cfg.log_transform
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
        )
        feats_all = feats_all.reindex(columns=INDEX_COLS + self._feat_cols, fill_value=0.0)
        feats_all["_ts"] = feats_all["time_period"].apply(lambda p: period_to_timestamp(p, freq))

        # Seed lags from the deseasonalized history (ffill the tail).
        deseason_hist = deseason_hist.copy()
        deseason_hist["_ts"] = deseason_hist["time_period"].apply(
            lambda p: period_to_timestamp(p, freq)
        )

        future = future_df.copy()
        future["_ts"] = future["time_period"].apply(lambda p: period_to_timestamp(p, freq))

        rows = []
        for loc, fg in future.groupby("location", sort=False):
            loc_str = str(loc)
            fg = fg.sort_values("_ts")
            times = fg["time_period"].astype(str).tolist()
            h = len(fg)

            seed = (
                deseason_hist[deseason_hist["location"].astype(str) == loc_str]
                .sort_values("_ts")[target]
                .ffill()
                .to_numpy(dtype=float)
            )
            if len(seed) < cfg.n_target_lags:
                seed = np.concatenate(
                    [np.full(cfg.n_target_lags - len(seed), seed[-1] if len(seed) else 0.0), seed]
                )

            fut_feat_rows = (
                feats_all[feats_all["location"].astype(str) == loc_str]
                .sort_values("_ts")
                .tail(h)[self._feat_cols]
                .to_numpy(dtype=float)
            )

            seasonal = dec.extrapolate_seasonal(decomps[loc_str], self._season_lengths, h)

            if cfg.prob_model == "recursive_residual":
                deseason_draws = self._recursive_forecast_samples(
                    seed, fut_feat_rows, h, cfg.n_samples, rng
                )  # (h, n) already point + compounded noise
                final = seasonal[:, None] + deseason_draws
            else:
                rf_point = self._rf_point_forecast(loc_str, seed, fut_feat_rows, h)
                resid_draws = self._arima_forecast(loc_str, h, rng)  # (h, n_samples)
                final = seasonal[:, None] + rf_point[:, None] + resid_draws  # (h, n)
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
