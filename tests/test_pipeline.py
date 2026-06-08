"""Smoke + invariant tests for the published MSTL + ARIMA + RF-residual model."""

import numpy as np
import pandas as pd
import pytest

from mstl_multistep import RunConfig, build_chap_model


def _synthetic_panel(n_locations=2, n_months=72, seed=0):
    rng = np.random.default_rng(seed)
    months = pd.period_range("2010-01", periods=n_months, freq="M").astype(str)
    rows = []
    for li in range(n_locations):
        t = np.arange(n_months)
        cases = np.clip(0.1 * t + 10 * li + 5 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 1, n_months), 0, None).round()
        rows.append(pd.DataFrame({
            "time_period": months, "location": f"loc{li}",
            "disease_cases": cases, "rainfall": rng.normal(50, 10, n_months),
        }))
    return pd.concat(rows, ignore_index=True)


def _split(df, horizon=3):
    historic, future = [], []
    for _, g in df.groupby("location", sort=False):
        g = g.sort_values("time_period")
        historic.append(g.iloc[:-horizon]); future.append(g.iloc[-horizon:])
    return pd.concat(historic, ignore_index=True), pd.concat(future, ignore_index=True)


def _cfg(**kw):
    base = dict(n_samples=20, rf={"n_estimators": 30, "random_state": 0})
    base.update(kw)
    return RunConfig(**base)


@pytest.mark.parametrize("covariates", [[], ["rainfall"]])
def test_fit_predict_shapes(covariates):
    df = _synthetic_panel()
    historic, future = _split(df)
    model = build_chap_model(_cfg(), feature_columns=covariates)
    model.fit(historic)
    preds = model.predict(historic, future)
    assert set(preds["location"]) == {"loc0", "loc1"}
    assert len(preds) == len(future)
    sc = [c for c in preds.columns if c.startswith("sample_")]
    assert len(sc) == 20
    vals = preds[sc].to_numpy()
    assert np.isfinite(vals).all() and (vals >= 0).all()


def test_target_lags_and_variance_head():
    """Champion-style config (target lags + tree variance head) fits and yields dispersion."""
    df = _synthetic_panel(n_months=84)
    historic, future = _split(df, horizon=3)
    model = build_chap_model(_cfg(rf_target_lags=3, residual_variance="tree"), ["rainfall"])
    model.fit(historic)
    preds = model.predict(historic, future)
    assert len(preds) == len(future)
    assert preds.filter(like="sample_").std(axis=1).mean() > 0  # genuine spread


def test_missing_location_history_is_finite():
    """A location whose target is entirely NaN must still yield finite forecasts."""
    df = _synthetic_panel(n_locations=2)
    df.loc[df["location"] == "loc1", "disease_cases"] = np.nan
    historic, future = _split(df)
    model = build_chap_model(_cfg(), feature_columns=["rainfall"])
    model.fit(historic)
    preds = model.predict(historic, future)
    assert np.isfinite(preds.filter(like="sample_").to_numpy()).all()


def test_build_model_features_deseasonalizes_only_listed_cols():
    """Listed covariate loses its seasonal cycle; an unlisted one keeps it."""
    from mstl_multistep.features import build_model_features
    df = _synthetic_panel(n_locations=1).copy()
    t = np.arange(len(df))
    df["clim"] = 10 * np.sin(2 * np.pi * t / 12)
    df["spray"] = 10 * np.sin(2 * np.pi * t / 12)
    feats = build_model_features(
        df, None, ["clim", "spray"], "MS", [12],
        min_lag=1, max_lag=1, use_location_dummies=False, deseasonalize_cols=["clim"],
    )
    assert np.nanmax(np.abs(feats["clim_lag1"].to_numpy())) < 2.0
    assert np.nanmax(np.abs(feats["spray_lag1"].to_numpy())) > 5.0


def test_deseasonalize_covariates_removes_seasonal_cycle():
    from mstl_multistep import decomposition as dec
    df = _synthetic_panel(n_locations=1).copy()
    t = np.arange(len(df))
    df["rainfall"] = 10 * np.sin(2 * np.pi * t / 12)
    out = dec.deseasonalize_covariates(df, None, "MS", ["rainfall"], [12])
    assert np.nanmax(np.abs(out["rainfall"].to_numpy())) < 2.0
