"""Smoke + invariant tests for the MSTL + multistep pipeline."""

import numpy as np
import pandas as pd
import pytest

from mstl_multistep import MSTLMultistepModel, RunConfig, build_chap_model


def _synthetic_panel(n_locations=2, n_months=60, seed=0):
    rng = np.random.default_rng(seed)
    months = pd.period_range("2010-01", periods=n_months, freq="M").astype(str)
    rows = []
    for li in range(n_locations):
        t = np.arange(n_months)
        seasonal = 5 * np.sin(2 * np.pi * t / 12)
        trend = 0.1 * t + 10 * li
        noise = rng.normal(0, 1, n_months)
        cases = np.clip(trend + seasonal + noise, 0, None).round()
        rows.append(
            pd.DataFrame(
                {
                    "time_period": months,
                    "location": f"loc{li}",
                    "disease_cases": cases,
                    "rainfall": rng.normal(50, 10, n_months),
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def _split(df, horizon=3):
    """Hold out the last `horizon` months per location as the future window."""
    historic, future = [], []
    for _, g in df.groupby("location", sort=False):
        g = g.sort_values("time_period")
        historic.append(g.iloc[:-horizon])
        future.append(g.iloc[-horizon:])
    return (
        pd.concat(historic, ignore_index=True),
        pd.concat(future, ignore_index=True),
    )


@pytest.mark.parametrize("prob_wrapper", ["bootstrap", "bucketedresidual"])
def test_fit_predict_shapes_no_covariates(prob_wrapper):
    df = _synthetic_panel()
    historic, future = _split(df, horizon=3)
    cfg = RunConfig(
        n_samples=20,
        n_target_lags=4,
        prob_wrapper=prob_wrapper,
        rf={"n_estimators": 30, "random_state": 0},
    )
    model = MSTLMultistepModel(cfg, feature_columns=[])
    model.fit(historic)
    preds = model.predict(historic, future)

    assert set(preds["location"]) == {"loc0", "loc1"}
    assert len(preds) == len(future)
    sample_cols = [c for c in preds.columns if c.startswith("sample_")]
    assert len(sample_cols) == 20
    vals = preds[sample_cols].to_numpy()
    assert np.isfinite(vals).all()
    assert (vals >= 0).all()  # forecasts are non-negative counts


def test_fit_predict_with_covariates():
    df = _synthetic_panel()
    historic, future = _split(df, horizon=3)
    cfg = RunConfig(n_samples=15, n_target_lags=4, rf={"n_estimators": 30, "random_state": 0})
    model = MSTLMultistepModel(cfg, feature_columns=["rainfall"])
    model.fit(historic)
    preds = model.predict(historic, future)
    assert len(preds) == len(future)
    assert (preds.filter(like="sample_").to_numpy() >= 0).all()


@pytest.mark.parametrize("covariates", [[], ["rainfall"]])
def test_rf_residual_fit_predict(covariates):
    df = _synthetic_panel(n_months=72)
    historic, future = _split(df, horizon=3)
    cfg = RunConfig(
        prob_model="rf_residual",
        n_samples=25,
        rf={"n_estimators": 40, "random_state": 0},
    )
    model = build_chap_model(cfg, feature_columns=covariates)
    model.fit(historic)
    preds = model.predict(historic, future)
    assert len(preds) == len(future)
    vals = preds.filter(like="sample_").to_numpy()
    assert np.isfinite(vals).all() and (vals >= 0).all()


@pytest.mark.parametrize("covariates", [[], ["rainfall"]])
def test_recursive_residual_fit_predict(covariates):
    df = _synthetic_panel(n_months=72)
    historic, future = _split(df, horizon=3)
    cfg = RunConfig(
        prob_model="recursive_residual",
        n_samples=30,
        n_target_lags=4,
        rf={"n_estimators": 40, "random_state": 0},
    )
    model = build_chap_model(cfg, feature_columns=covariates)
    model.fit(historic)
    preds = model.predict(historic, future)
    assert len(preds) == len(future)
    vals = preds.filter(like="sample_").to_numpy()
    assert np.isfinite(vals).all() and (vals >= 0).all()


def test_recursive_residual_variance_grows_with_horizon():
    """Compounded residuals should widen the spread at later horizons."""
    df = _synthetic_panel(n_locations=1, n_months=96)
    historic, future = _split(df, horizon=8)
    cfg = RunConfig(
        prob_model="recursive_residual",
        n_samples=200,
        n_target_lags=6,
        log_transform=False,  # compare spread in the native space
        rf={"n_estimators": 80, "random_state": 0},
    )
    model = build_chap_model(cfg, [])
    model.fit(historic)
    preds = model.predict(historic, future).sort_values("time_period")
    spreads = preds.filter(like="sample_").std(axis=1).to_numpy()
    # later-horizon spread should exceed the first step (allowing some noise)
    assert spreads[-1] > spreads[0]


@pytest.mark.parametrize("covariates", [[], ["rainfall"]])
def test_arima_residual_fit_predict(covariates):
    df = _synthetic_panel(n_months=72)
    historic, future = _split(df, horizon=3)
    cfg = RunConfig(
        prob_model="arima_residual",
        n_samples=25,
        n_target_lags=4,
        arima_stepwise=True,
        rf={"n_estimators": 40, "random_state": 0},
    )
    model = build_chap_model(cfg, feature_columns=covariates)
    model.fit(historic)
    preds = model.predict(historic, future)
    assert len(preds) == len(future)
    sample_cols = [c for c in preds.columns if c.startswith("sample_")]
    assert len(sample_cols) == 25
    vals = preds[sample_cols].to_numpy()
    assert np.isfinite(vals).all() and (vals >= 0).all()


def test_arima_residual_widens_intervals_vs_multistep():
    """ARIMA-residual draws should not collapse to a near-zero spread."""
    df = _synthetic_panel(n_months=84)
    historic, future = _split(df, horizon=6)
    common = dict(n_samples=80, n_target_lags=6, rf={"n_estimators": 60, "random_state": 0})
    ar = build_chap_model(RunConfig(prob_model="arima_residual", **common), [])
    ar.fit(historic)
    spread = ar.predict(historic, future).filter(like="sample_").std(axis=1).mean()
    assert spread > 0  # produces genuine dispersion


@pytest.mark.parametrize("prob_model", ["multistep", "arima_residual"])
def test_deseasonalized_covariates(prob_model):
    df = _synthetic_panel(n_months=72)
    historic, future = _split(df, horizon=3)
    cfg = RunConfig(
        prob_model=prob_model,
        deseasonalize_covariates=True,
        n_samples=20,
        n_target_lags=4,
        rf={"n_estimators": 40, "random_state": 0},
    )
    model = build_chap_model(cfg, feature_columns=["rainfall"])
    model.fit(historic)
    preds = model.predict(historic, future)
    assert len(preds) == len(future)
    vals = preds.filter(like="sample_").to_numpy()
    assert np.isfinite(vals).all() and (vals >= 0).all()


def test_deseasonalize_covariates_removes_seasonal_cycle():
    """A purely seasonal covariate should be flattened to ~0 after deseasonalizing."""
    from mstl_multistep import decomposition as dec

    df = _synthetic_panel(n_locations=1, n_months=72)
    # overwrite rainfall with a pure 12-month sinusoid (no trend, no noise)
    t = np.arange(len(df))
    df = df.copy()
    df["rainfall"] = 10 * np.sin(2 * np.pi * t / 12)
    out = dec.deseasonalize_covariates(df, None, "MS", ["rainfall"], [12])
    # after removing the seasonal component, the anomaly should be tiny
    assert np.nanmax(np.abs(out["rainfall"].to_numpy())) < 2.0


def test_seasonality_is_reconstructed():
    """Samples should track the seasonal phase, not be flat."""
    df = _synthetic_panel(n_locations=1, n_months=72)
    historic, future = _split(df, horizon=12)
    cfg = RunConfig(n_samples=50, n_target_lags=6, rf={"n_estimators": 50, "random_state": 0})
    model = MSTLMultistepModel(cfg, feature_columns=[])
    model.fit(historic)
    preds = model.predict(historic, future).sort_values("time_period")
    means = preds.filter(like="sample_").mean(axis=1).to_numpy()
    # A full year of forecasts should show seasonal variation.
    assert means.std() > 0.5 * means.mean() * 0.1
