"""MSTL + multistep trend-model orchestrator.

Training
--------
1. Per location, MSTL-decompose ``log1p(target)`` (or raw target if
   ``log_transform`` is off) into seasonal + deseasonalized.
2. Fit the reused ``MultistepModel`` from :mod:`simple_multistep_model` on
   the pooled panel of deseasonalized series (target lags + optional
   covariate lags + optional location dummies). This is the *trend model*.

Prediction
----------
3. Re-decompose the historic context the same way to obtain the seasonal
   shape and to seed the recursive lag window with deseasonalized values.
4. Recursively sample the deseasonalized forecast from the trend model.
5. Add the seasonal-naive extrapolation back to every sample, then expm1
   (if log-transformed), clip at 0, and optionally Poisson-discretize.

The trend model is used exactly as it is standalone — same regressor, same
probabilistic wrapper, same lag handling — only the series it forecasts is
the MSTL-deseasonalized one rather than the raw target.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from simple_multistep_model import (
    BucketCalculator,
    BucketedResidualBootstrapModel,
    FixedMapieCrossConformalRegressor,
    SkproWrapper,
    features_to_xarray,
    target_to_xarray,
)
from simple_multistep_model.multistep import MultistepModel
from skpro.regression.bootstrap import BootstrapRegressor
from sklearn.ensemble import RandomForestRegressor

from mstl_multistep import decomposition as dec
from mstl_multistep.features import INDEX_COLS, build_features
from mstl_multistep.io_utils import detect_frequency, period_to_timestamp
from mstl_multistep.run_config import RunConfig

logger = logging.getLogger(__name__)


def _build_regressor(cfg: RunConfig) -> RandomForestRegressor:
    rf = cfg.rf
    return RandomForestRegressor(
        n_estimators=rf.n_estimators,
        max_depth=rf.max_depth,
        min_samples_leaf=rf.min_samples_leaf,
        max_features=rf.max_features,
        random_state=rf.random_state,
    )


def _build_one_step(cfg: RunConfig):
    """Return ``(one_step_model, bucket_calculator_or_None)`` for the trend model."""
    regressor = _build_regressor(cfg)
    if cfg.prob_wrapper == "bucketedresidual":
        return (
            BucketedResidualBootstrapModel(regressor),
            BucketCalculator(min_bucket_size=cfg.min_bucket_size),
        )
    if cfg.prob_wrapper == "bootstrap":
        return SkproWrapper(BootstrapRegressor(regressor)), None
    if cfg.prob_wrapper == "cross-conformal":
        return SkproWrapper(FixedMapieCrossConformalRegressor(regressor)), None
    raise ValueError(f"Unknown prob_wrapper: {cfg.prob_wrapper!r}")


def _predictions_to_wide(
    predictions, ordered_future: dict[str, list[str]]
) -> pd.DataFrame:
    """Convert (location, trajectory, step) samples to a wide sample_* frame."""
    rows = []
    for loc in predictions.coords["location"].values:
        loc_str = str(loc)
        loc_preds = predictions.sel(location=loc)
        times = ordered_future[loc_str]
        for step_idx in range(loc_preds.sizes["step"]):
            samples = loc_preds.isel(step=step_idx).values
            row = {"time_period": times[step_idx], "location": loc_str}
            for i, s in enumerate(samples):
                row[f"sample_{i}"] = float(s)
            rows.append(row)
    return pd.DataFrame(rows)


class MSTLMultistepModel:
    """MSTL decomposition wrapped around a recursive multistep trend model."""

    def __init__(self, cfg: RunConfig, feature_columns: list[str]):
        self.cfg = cfg
        self.feature_columns = list(feature_columns)
        self._model: MultistepModel | None = None
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

    def _decompose_panel(self, df: pd.DataFrame, freq: str):
        """Return (deseasonalized_target_df, {loc: decomp_frame}).

        The returned frame has the index columns plus the (de-seasonalized,
        and log-transformed if configured) target column. Decompositions are
        kept per location for seasonal extrapolation at predict time.
        """
        target = self.cfg.target_variable
        df = df.copy()
        df["_ts"] = df["time_period"].apply(lambda p: period_to_timestamp(p, freq))

        blocks: list[pd.DataFrame] = []
        decomps: dict[str, pd.DataFrame] = {}
        for loc, g in df.groupby("location", sort=False):
            g = g.sort_values("_ts")
            y = pd.to_numeric(g[target], errors="coerce").to_numpy(dtype=float)
            y_t = np.log1p(np.clip(y, a_min=0.0, a_max=None)) if self.cfg.log_transform else y
            decomp = dec.decompose(y_t, self._season_lengths)
            decomps[str(loc)] = decomp
            sub = g[INDEX_COLS].copy()
            sub[target] = dec.deseasonalize(decomp)
            blocks.append(sub)

        deseason = pd.concat(blocks, ignore_index=True)
        return deseason, decomps

    # -- public API --------------------------------------------------------

    def fit(self, historic_df: pd.DataFrame) -> "MSTLMultistepModel":
        cfg = self.cfg
        self._freq = detect_frequency(historic_df)
        self._season_lengths = self._season_lengths_for(self._freq)

        deseason, _ = self._decompose_panel(historic_df, self._freq)
        feats = build_features(
            historic_df,
            self.feature_columns,
            cfg.feature_min_lag,
            cfg.feature_max_lag,
            cfg.use_location_dummies,
        )

        one_step, bucket = _build_one_step(cfg)
        self._model = MultistepModel(one_step, cfg.n_target_lags, bucket_calculator=bucket)

        y_xr = target_to_xarray(deseason, cfg.target_variable)
        X_xr = features_to_xarray(feats)  # None when no feature columns
        self._model.fit_multi(y_xr, X_xr)
        return self

    def predict(self, historic_df: pd.DataFrame, future_df: pd.DataFrame) -> pd.DataFrame:
        if self._model is None:
            raise RuntimeError("Call fit() before predict()")
        cfg = self.cfg
        freq = self._freq
        target = cfg.target_variable
        n_steps = int(future_df.groupby("location").size().max())

        # Re-decompose the historic context: gives the seasonal shape and the
        # deseasonalized values used to seed the recursive lag window.
        deseason_hist, decomps = self._decompose_panel(historic_df, freq)

        feats = build_features(
            pd.concat([historic_df, future_df], ignore_index=True),
            self.feature_columns,
            cfg.feature_min_lag,
            cfg.feature_max_lag,
            cfg.use_location_dummies,
        )

        y_xr = target_to_xarray(deseason_hist, target, ffill=True)
        previous_y = y_xr.isel(time=slice(-cfg.n_target_lags, None))

        X_xr = features_to_xarray(feats, ffill=True)
        X_future_xr = (
            X_xr.isel(time=slice(-n_steps, None)).rename({"time": "step"})
            if X_xr is not None
            else None
        )

        # Per-location ordered future time_periods (step order) + bucket times.
        future = future_df.copy()
        future["_ts"] = future["time_period"].apply(lambda p: period_to_timestamp(p, freq))
        ordered_future: dict[str, list[str]] = {}
        for loc, g in future.groupby("location", sort=False):
            ordered_future[str(loc)] = (
                g.sort_values("_ts")["time_period"].astype(str).tolist()
            )
        future_times = ordered_future if self._model.bucket_calculator is not None else None

        predictions = self._model.predict_multi(
            previous_y,
            n_steps,
            cfg.n_samples,
            X_future_xr,
            future_times=future_times,
        )

        wide = _predictions_to_wide(predictions, ordered_future)
        sample_cols = [c for c in wide.columns if c.startswith("sample_")]

        # Seasonal extrapolation per (location, time_period), added back in the
        # same (log or raw) space the trend model forecasts in.
        seas_lookup: dict[tuple[str, str], float] = {}
        for loc, decomp in decomps.items():
            times = ordered_future.get(loc, [])
            seas_vals = dec.extrapolate_seasonal(decomp, self._season_lengths, len(times))
            for tp, s in zip(times, seas_vals):
                seas_lookup[(loc, tp)] = float(s)

        seas = wide.apply(
            lambda r: seas_lookup.get((str(r["location"]), str(r["time_period"])), 0.0),
            axis=1,
        ).to_numpy()
        wide[sample_cols] = wide[sample_cols].to_numpy() + seas[:, None]

        if cfg.log_transform:
            wide[sample_cols] = np.expm1(wide[sample_cols])
        wide[sample_cols] = wide[sample_cols].clip(lower=0.0)

        if cfg.discretize_samples:
            rng = np.random.default_rng(cfg.random_seed)
            wide[sample_cols] = rng.poisson(wide[sample_cols].to_numpy()).astype(float)

        return wide[INDEX_COLS + sample_cols]
