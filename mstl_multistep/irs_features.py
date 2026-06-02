"""Feature extraction from a raw IRS allocation column.

The raw ``irs_allocated`` column is a 0-1 coverage fraction that fires only in
the months a spray campaign is actually allocated (~2.5% of rows). Its
*protective effect*, however, persists for months after the campaign and is
known ahead of time (allocation is planned), so feeding the raw column through
the ordinary lag machinery mostly feeds the model zeros.

This module turns that sparse event series into a few **dense, contemporaneous**
signals, computed per location over the time-sorted historic(+future) panel:

- ``level``      — the raw allocation coverage this month (0-1).
- ``decay``      — protection that resets to the allocation level on a campaign
                   month and decays geometrically afterwards
                   (``d_t = max(level_t, gamma * d_{t-1})``,
                   ``gamma = 0.5 ** (1 / halflife)``). Dense and bounded in [0, 1].
- ``since``      — months since the last allocation, capped at ``since_cap``
                   (large when never sprayed).
- ``cumulative`` — running count of allocated months: a stock-of-protection /
                   program-intensity proxy.

These are returned at lag 0 (the campaign month itself) because allocation is a
known future covariate — unlike climate, we do not have to lag it to avoid
leakage. The caller merges them onto its design matrix as extra RF features.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

INDEX_COLS = ["time_period", "location"]

IRS_FEATURE_NAMES = ("level", "decay", "since", "cumulative")


def _decay_series(level: np.ndarray, gamma: float) -> np.ndarray:
    """``d_t = max(level_t, gamma * d_{t-1})`` — geometric decay, reset on spray."""
    out = np.zeros(len(level), dtype=float)
    prev = 0.0
    for i, lv in enumerate(level):
        prev = max(float(lv), gamma * prev)
        out[i] = prev
    return out


def build_irs_features(
    historic_df: pd.DataFrame,
    future_df: pd.DataFrame | None,
    column: str,
    features: list[str],
    halflife: float,
    since_cap: int = 24,
) -> tuple[pd.DataFrame, list[str]]:
    """Return ``(frame[INDEX_COLS + irs_cols], irs_cols)`` of engineered IRS features.

    ``frame`` covers every (time_period, location) in historic(+future). Each
    feature is computed per location over the chronologically sorted union so the
    decay/cumulative state flows correctly from history into the forecast window.
    Returns ``(empty-index frame, [])`` when the column is absent or no features
    are requested.
    """
    requested = [f for f in features if f in IRS_FEATURE_NAMES]
    base = historic_df if future_df is None else pd.concat(
        [historic_df, future_df], ignore_index=True
    )
    if not requested or column not in base.columns:
        return base[INDEX_COLS].copy(), []

    gamma = 0.5 ** (1.0 / max(float(halflife), 1e-6))
    src = base[INDEX_COLS + [column]].copy()
    src[column] = pd.to_numeric(src[column], errors="coerce").fillna(0.0)
    src["_ts"] = pd.PeriodIndex(src["time_period"].astype(str), freq="M").to_timestamp()

    out_blocks = []
    for _, g in src.groupby("location", sort=False):
        g = g.sort_values("_ts").copy()
        level = g[column].to_numpy(dtype=float)
        nonzero = level > 0

        feat = {}
        if "level" in requested:
            feat["irs_level"] = level
        if "decay" in requested:
            feat["irs_decay"] = _decay_series(level, gamma)
        if "since" in requested:
            # months since the last nonzero allocation, capped.
            since = np.empty(len(level), dtype=float)
            last = -1
            for i in range(len(level)):
                if nonzero[i]:
                    last = i
                since[i] = since_cap if last < 0 else min(i - last, since_cap)
            feat["irs_since"] = since
        if "cumulative" in requested:
            feat["irs_cumulative"] = np.cumsum(level)

        blk = g[INDEX_COLS].copy()
        for k, v in feat.items():
            blk[k] = v
        out_blocks.append(blk)

    frame = pd.concat(out_blocks, ignore_index=True)
    irs_cols = [c for c in frame.columns if c not in INDEX_COLS]
    return frame, irs_cols
