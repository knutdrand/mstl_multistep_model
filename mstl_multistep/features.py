"""Feature engineering for the deseasonalized trend model.

Deliberately *omits* calendar (month sin/cos) features: the target the
multistep model sees has already had its seasonality removed by MSTL, so
calendar features would be redundant. We keep lagged covariates (when the
user supplies any) and optional one-hot location columns so the pooled
model retains per-series identity. Target lags are added by the
``MultistepModel`` itself.
"""

from __future__ import annotations

import pandas as pd

INDEX_COLS = ["time_period", "location"]


def lag_covariates(df: pd.DataFrame, min_lag: int, max_lag: int) -> pd.DataFrame:
    """Replace each covariate column with its lags ``{col}_lag{k}``, per location.

    Rows are sorted by location then time_period before shifting so lags are
    chronologically correct. With no covariate columns this returns just the
    index columns.
    """
    feature_cols = [c for c in df.columns if c not in INDEX_COLS]
    df = df.sort_values(["location", "time_period"]).copy()
    for col in feature_cols:
        for lag in range(min_lag, max_lag + 1):
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
) -> pd.DataFrame:
    """Build the model feature frame from ``[time_period, location] + covariates``.

    Returns a frame with the two index columns plus engineered feature
    columns. May be just the index columns (pure target-lag autoregression)
    when there are no covariates and dummies are disabled.
    """
    sub = df[INDEX_COLS + [c for c in feature_columns if c in df.columns]].copy()
    out = lag_covariates(sub, min_lag=min_lag, max_lag=max_lag)
    if use_location_dummies:
        out = one_hot_locations(out)
    return out
