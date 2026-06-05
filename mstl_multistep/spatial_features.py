"""Spatial neighbor features from an admin-grouping column (e.g. ``district``).

The 406 sectors are nested under 46 districts (every district has >=3 sectors), so
same-district sectors are a proximity proxy in the absence of coordinates. The key
feature is a **leave-one-out district mean of the deseasonalized target** — a spatial
autoregressive signal ("is the surrounding district seeing elevated transmission?"),
the spatial analogue of the per-location target lags. It is lagged (like rf_target_lags)
because in self-forecasting the neighbours' future cases are unknown; the forecast-window
bridge holds the last observed neighbour mean (persistence), which is adequate over the
short horizon.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

INDEX_COLS = ["time_period", "location"]


def build_neighbor_target(
    deseason_df: pd.DataFrame,
    group_map: dict,
    target_col: str,
) -> pd.DataFrame:
    """``[time_period, location, _nbrD]`` leave-one-out group mean of the (deseasonalized) target.

    For each location at time t, ``_nbrD`` is the mean of ``target_col`` over the *other*
    same-group locations at t (NaN where the location has no same-time neighbours with a
    finite value). ``group_map`` maps a location id (str) to its group id (e.g. district).
    """
    df = deseason_df[INDEX_COLS + [target_col]].copy()
    df["_grp"] = df["location"].astype(str).map(group_map)
    v = pd.to_numeric(df[target_col], errors="coerce")
    fin = v.notna()
    df["_vf"] = np.where(fin, v, 0.0)
    df["_nf"] = fin.astype(float)
    by = df.groupby(["_grp", "time_period"], sort=False)
    grp_sum = by["_vf"].transform("sum")
    grp_cnt = by["_nf"].transform("sum")
    # leave-one-out: remove self when self is finite
    own = np.where(fin, v, 0.0)
    own_n = fin.astype(float)
    loo_sum = grp_sum - own
    loo_cnt = grp_cnt - own_n
    out = df[INDEX_COLS].copy()
    out["_nbrD"] = np.where(loo_cnt > 0, loo_sum / loo_cnt, np.nan)
    return out
