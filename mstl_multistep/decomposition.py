"""Per-series MSTL decomposition helpers.

Each series is decomposed into ``trend + seasonal(s) + remainder``. The
*deseasonalized* series (``trend + remainder``, equivalently
``data - seasonal``) is what the multistep trend model is fit on; the
seasonal piece is held fixed and extrapolated seasonal-naive into the
forecast horizon.
"""

from __future__ import annotations

import logging
from math import trunc

import numpy as np
import pandas as pd
from statsforecast.mstl import mstl

from mstl_multistep.io_utils import period_to_timestamp

logger = logging.getLogger(__name__)

INDEX_COLS = ["time_period", "location"]


def _seasonal_columns(decomp: pd.DataFrame) -> list[str]:
    return [c for c in decomp.columns if c.startswith("seasonal")]


def decompose(
    y: np.ndarray,
    season_lengths: list[int],
    stl_kwargs: dict | None = None,
) -> pd.DataFrame:
    """MSTL-decompose a 1-d series.

    NaNs are linearly interpolated *only to estimate the seasonal shape*;
    callers keep the original NaNs in the deseasonalized target so the
    multistep model's own drop mask handles them. Series shorter than two
    full seasons are treated as all-trend (zero seasonal).
    """
    y = np.asarray(y, dtype=float)
    min_required = 2 * season_lengths[0]
    if np.isfinite(y).sum() < min_required:
        logger.warning(
            "series too short for MSTL (%d finite < 2*season=%d); treating as all-trend",
            int(np.isfinite(y).sum()),
            season_lengths[0],
        )
        return pd.DataFrame({"data": y, "trend": y, "remainder": np.zeros_like(y)})

    y_filled = pd.Series(y).interpolate(limit_direction="both").to_numpy()
    decomp = mstl(x=y_filled, period=season_lengths, stl_kwargs=stl_kwargs or {})
    decomp = decomp.reset_index(drop=True)
    # Restore original data column (pre-imputation) for bookkeeping.
    decomp["data"] = y
    return decomp


def seasonal_component(decomp: pd.DataFrame) -> np.ndarray:
    """Sum of all seasonal columns (zeros if none)."""
    cols = _seasonal_columns(decomp)
    if not cols:
        return np.zeros(len(decomp))
    return decomp[cols].to_numpy().sum(axis=1)


def deseasonalize(decomp: pd.DataFrame) -> np.ndarray:
    """``data - seasonal`` — keeps NaN wherever the original data was NaN."""
    return decomp["data"].to_numpy(dtype=float) - seasonal_component(decomp)


def decompose_panel(
    df: pd.DataFrame,
    freq: str,
    target: str,
    season_lengths: list[int],
    log_transform: bool,
):
    """MSTL-decompose every location's series.

    Returns ``(deseasonalized_df, {loc: decomp_frame})`` where
    ``deseasonalized_df`` has the index columns plus the (de-seasonalized,
    and log1p-transformed if requested) target column. Original NaNs in the
    target are preserved in the deseasonalized output.
    """
    df = df.copy()
    df["_ts"] = df["time_period"].apply(lambda p: period_to_timestamp(p, freq))

    blocks: list[pd.DataFrame] = []
    decomps: dict[str, pd.DataFrame] = {}
    for loc, g in df.groupby("location", sort=False):
        g = g.sort_values("_ts")
        y = pd.to_numeric(g[target], errors="coerce").to_numpy(dtype=float)
        y_t = np.log1p(np.clip(y, a_min=0.0, a_max=None)) if log_transform else y
        decomp = decompose(y_t, season_lengths)
        decomps[str(loc)] = decomp
        sub = g[INDEX_COLS].copy()
        sub[target] = deseasonalize(decomp)
        blocks.append(sub)

    return pd.concat(blocks, ignore_index=True), decomps


def deseasonalize_covariates(
    historic_df: pd.DataFrame,
    future_df: pd.DataFrame | None,
    freq: str,
    covariate_cols: list[str],
    season_lengths: list[int],
) -> pd.DataFrame:
    """Replace each covariate with its MSTL-deseasonalized (trend+remainder) value.

    Returns ``[time_period, location, *covariate_cols]`` spanning the historic
    rows (and the future rows when ``future_df`` is given). Historic values are
    ``raw - seasonal`` from the per-series decomposition; future values are
    ``raw - seasonal_naive_extrapolation``. Covariates are decomposed in their
    natural (raw) space — no log transform — so the result is the climate
    *anomaly* relative to its own seasonal cycle.
    """
    cols = [c for c in covariate_cols if c in historic_df.columns]

    hist = historic_df.copy()
    hist["_ts"] = hist["time_period"].apply(lambda p: period_to_timestamp(p, freq))
    fut = None
    if future_df is not None and len(future_df):
        fut = future_df.copy()
        fut["_ts"] = fut["time_period"].apply(lambda p: period_to_timestamp(p, freq))

    blocks: list[pd.DataFrame] = []
    for loc, hg in hist.groupby("location", sort=False):
        hg = hg.sort_values("_ts")
        h_block = hg[INDEX_COLS].copy()

        fgloc = None
        if fut is not None:
            fgloc = fut[fut["location"].astype(str) == str(loc)].sort_values("_ts")
            fgloc = fgloc if len(fgloc) else None
        f_block = fgloc[INDEX_COLS].copy() if fgloc is not None else None

        for col in cols:
            raw = pd.to_numeric(hg[col], errors="coerce").to_numpy(dtype=float)
            decomp = decompose(raw, season_lengths)
            h_block[col] = raw - seasonal_component(decomp)
            if f_block is not None:
                h_fut = len(fgloc)
                seas_fut = extrapolate_seasonal(decomp, season_lengths, h_fut)
                raw_fut = pd.to_numeric(fgloc[col], errors="coerce").to_numpy(dtype=float)
                f_block[col] = raw_fut - seas_fut

        blocks.append(h_block)
        if f_block is not None:
            blocks.append(f_block)

    return pd.concat(blocks, ignore_index=True)


def extrapolate_seasonal(
    decomp: pd.DataFrame,
    season_lengths: list[int],
    h: int,
) -> np.ndarray:
    """Seasonal-naive forward extrapolation of length ``h``.

    For each seasonal component, take its last ``m`` values and tile them
    forward; sum across components.
    """
    seas_cols = _seasonal_columns(decomp)
    if not seas_cols:
        return np.zeros(h)
    total = np.zeros(h)
    for col, m in zip(seas_cols, season_lengths):
        last_cycle = decomp[col].to_numpy()[-m:]
        tiled = np.tile(last_cycle, trunc(1 + (h - 1) / m))[:h]
        total = total + tiled
    return total
