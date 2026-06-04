"""Feature engineering for the deseasonalized trend model.

By default *omits* calendar (month sin/cos) features: the target the
multistep model sees has already had its seasonality removed by MSTL, so
calendar features would be redundant. We keep lagged covariates (when the
user supplies any) and optional one-hot location columns so the pooled
model retains per-series identity. Target lags are added by the
``MultistepModel`` itself. Fourier seasonal features can be re-introduced
explicitly via :func:`build_seasonal_features` (``seasonal_fourier_order``),
since MSTL's seasonal-naive extrapolation can leave residual seasonality.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mstl_multistep.io_utils import period_to_timestamp

INDEX_COLS = ["time_period", "location"]


def build_seasonal_features(
    historic_df: pd.DataFrame,
    future_df: pd.DataFrame | None,
    freq: str,
    order: int,
) -> tuple[pd.DataFrame, list[str]]:
    """Return ``([*INDEX_COLS, seas_sin1, seas_cos1, ...], cols)`` Fourier seasonal features.

    Encodes the within-year phase of each period as ``order`` sin/cos harmonics of the
    annual cycle (period 12 for monthly, 52 for weekly). Contemporaneous (lag 0); the
    same calendar features apply to historic and future rows. Empty when ``order<=0``.
    """
    base = historic_df if future_df is None else pd.concat(
        [historic_df, future_df], ignore_index=True
    )
    out = base[INDEX_COLS].copy()
    if order <= 0:
        return out, []
    ts = base["time_period"].apply(lambda p: period_to_timestamp(p, freq))
    if str(freq).startswith("W"):
        pos = ts.dt.isocalendar().week.astype(float) - 1.0
        period = 52.0
    else:
        pos = ts.dt.month.astype(float) - 1.0
        period = 12.0
    cols = []
    for k in range(1, order + 1):
        ang = 2.0 * np.pi * k * pos / period
        out[f"seas_sin{k}"] = np.sin(ang).to_numpy()
        out[f"seas_cos{k}"] = np.cos(ang).to_numpy()
        cols += [f"seas_sin{k}", f"seas_cos{k}"]
    return out, cols


def lag_covariates(
    df: pd.DataFrame,
    min_lag: int,
    max_lag: int,
    lags_by_col: dict[str, tuple[int, int]] | None = None,
) -> pd.DataFrame:
    """Replace each covariate column with its lags ``{col}_lag{k}``, per location.

    Rows are sorted by location then time_period before shifting so lags are
    chronologically correct. With no covariate columns this returns just the
    index columns.

    Each covariate uses the global ``(min_lag, max_lag)`` window unless it is
    named in ``lags_by_col``, which maps a covariate name to its own
    ``(min, max)`` inclusive lag range — letting e.g. rainfall carry a longer
    lag span than temperature.
    """
    lags_by_col = lags_by_col or {}
    feature_cols = [c for c in df.columns if c not in INDEX_COLS]
    df = df.sort_values(["location", "time_period"]).copy()
    for col in feature_cols:
        lo, hi = lags_by_col.get(col, (min_lag, max_lag))
        for lag in range(lo, hi + 1):
            df[f"{col}_lag{lag}"] = df.groupby("location")[col].shift(lag)
    return df.drop(columns=feature_cols)


def one_hot_locations(df: pd.DataFrame) -> pd.DataFrame:
    """Append one-hot ``loc_*`` columns (sorted, so fit/predict columns align)."""
    dummies = pd.get_dummies(df["location"], prefix="loc").astype(float)
    dummies = dummies.reindex(sorted(dummies.columns), axis=1)
    return pd.concat([df, dummies], axis=1)


def build_features(
    df: pd.DataFrame,
    feature_columns: list[str],
    min_lag: int,
    max_lag: int,
    use_location_dummies: bool,
    lags_by_col: dict[str, tuple[int, int]] | None = None,
) -> pd.DataFrame:
    """Build the model feature frame from ``[time_period, location] + covariates``.

    Returns a frame with the two index columns plus engineered feature
    columns. May be just the index columns (pure target-lag autoregression)
    when there are no covariates and dummies are disabled. ``lags_by_col``
    overrides the global lag window per covariate (see :func:`lag_covariates`).
    """
    sub = df[INDEX_COLS + [c for c in feature_columns if c in df.columns]].copy()
    out = lag_covariates(sub, min_lag=min_lag, max_lag=max_lag, lags_by_col=lags_by_col)
    if use_location_dummies:
        out = one_hot_locations(out)
    return out


def build_model_features(
    historic_df: pd.DataFrame,
    future_df: pd.DataFrame | None,
    feature_columns: list[str],
    freq: str,
    season_lengths: list[int],
    min_lag: int,
    max_lag: int,
    use_location_dummies: bool,
    deseasonalize_cols: list[str],
    lags_by_col: dict[str, tuple[int, int]] | None = None,
) -> pd.DataFrame:
    """Build features for fit (``future_df=None``) or predict (future provided).

    Covariates named in ``deseasonalize_cols`` are MSTL-deseasonalized first (so
    the model sees their anomalies, matching the deseasonalized target); all
    other covariates are lagged as-is. This lets climate be deseasonalized while
    seasonal intervention indicators (e.g. ``sprayed_*``) stay raw.
    """
    base = historic_df if future_df is None else pd.concat(
        [historic_df, future_df], ignore_index=True
    )
    present = [c for c in feature_columns if c in base.columns]
    source = base[INDEX_COLS + present].copy()

    to_deseason = [c for c in deseasonalize_cols if c in present]
    if to_deseason:
        from mstl_multistep import decomposition as dec

        des = dec.deseasonalize_covariates(
            historic_df, future_df, freq, to_deseason, season_lengths
        )
        source = source.drop(columns=to_deseason).merge(des, on=INDEX_COLS, how="left")

    return build_features(
        source, feature_columns, min_lag, max_lag, use_location_dummies, lags_by_col
    )
