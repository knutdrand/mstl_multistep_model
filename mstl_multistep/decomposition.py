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

logger = logging.getLogger(__name__)


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
