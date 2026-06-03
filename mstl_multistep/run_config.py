"""Pydantic config for one MSTL + multistep train/predict run.

Same wrapper shape CHAP hands every external model: model knobs under
``user_option_values`` and the list of feature columns under
``additional_continuous_covariates``. The RandomForest hyperparameters are
reused from :mod:`simple_multistep_model` so the trend model is configured
exactly as it is when run standalone.

Example YAML::

    additional_continuous_covariates: []          # pure self-forecasting trend
    user_option_values:
      target_variable: disease_cases
      log_transform: true
      season_length_monthly: 12
      n_target_lags: 6
      n_samples: 100
      prob_wrapper: bootstrap
      rf:
        n_estimators: 200
        max_depth: 10
        min_samples_leaf: 5
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field
from simple_multistep_model import RandomForestConfig

ProbWrapper = Literal["bucketedresidual", "bootstrap", "cross-conformal"]


class RunConfig(BaseModel):
    """All tunable knobs for one MSTL + multistep run."""

    model_config = ConfigDict(extra="forbid")

    target_variable: str = "disease_cases"

    # MSTL decomposition
    log_transform: bool = True
    season_length_monthly: int = 12
    season_length_weekly: int = 52

    # Trend model on the deseasonalized series
    n_target_lags: int = 6
    n_samples: int = 100
    feature_min_lag: int = 1
    feature_max_lag: int = 3
    # Per-covariate lag overrides: maps a covariate name to its own inclusive
    # ``[min, max]`` lag window, overriding ``feature_min_lag``/``feature_max_lag``
    # for that column only (others keep the global window). Lets e.g. rainfall
    # carry a longer lag span than temperature. Names not present are ignored.
    covariate_lags: dict[str, list[int]] = Field(default_factory=dict)
    use_location_dummies: bool = True
    # Names of covariates to MSTL-deseasonalize before lagging, so the model sees
    # their anomalies (matching the deseasonalized target) rather than the raw
    # seasonal series. Covariates not listed are passed through raw — e.g. keep
    # seasonal intervention indicators (sprayed_*) raw while deseasonalizing
    # climate. Names not present in the data are ignored.
    deseasonalize_covariates: list[str] = Field(default_factory=list)

    # How the deseasonalized series is forecast probabilistically:
    #   "multistep"          -> recursive RF from simple_multistep_model + prob_wrapper
    #   "arima_residual"     -> deterministic RF point + AutoARIMA on one-step OOB residuals
    #   "recursive_residual" -> RF point with one-step OOB residuals resampled and
    #                           compounded through the recursion (variance grows with horizon)
    #   "rf_residual"        -> ARIMA base (mean + sigma) with RF modelling the ARIMA
    #                           residual as a deterministic point correction (mirror of
    #                           arima_residual)
    prob_model: Literal[
        "multistep", "arima_residual", "recursive_residual", "rf_residual"
    ] = "multistep"

    # multistep-only
    prob_wrapper: ProbWrapper = "bootstrap"
    min_bucket_size: int = 5

    # arima_residual-only (AutoARIMA on the RF out-of-bag residuals)
    arima_approximation: bool = False
    arima_stepwise: bool = True
    arima_level: int = 68

    # IRS allocation feature extraction (rf_residual mode). When ``irs_column`` is
    # set and ``irs_features`` is non-empty, dense protective-effect features are
    # engineered from the raw (sparse) allocation column and added to the RF at
    # lag 0 — see :mod:`mstl_multistep.irs_features`. ``irs_halflife`` controls the
    # geometric decay (in months) of the ``decay`` feature.
    irs_column: str | None = None
    irs_features: list[str] = Field(default_factory=list)
    irs_halflife: float = 4.0

    # Number of lagged deseasonalized-target values fed to the rf_residual RF as
    # extra features (0 = none, the default — ARIMA alone carries the AR dynamics).
    # Future lags that fall in the forecast window are filled with the ARIMA mean
    # forecast (non-recursive bridge), so the RF can pick up residual autocorrelation
    # the linear ARIMA misses. Distinct from n_target_lags, which only the multistep
    # mode's MultistepModel uses.
    rf_target_lags: int = 0

    # EM / backfitting iterations for rf_residual (1 = current single forward pass,
    # exactly reproducible). When >1, alternately: remove the RF covariate effect
    # from the signal, re-run MSTL+ARIMA on the cleaned series, then re-fit the RF
    # on (signal - seasonal - ARIMA). ARIMA's sigma is naturally re-estimated on the
    # covariate-cleaned residual. The RF covariate effect is formed out-of-bag during
    # iterations so it does not corrupt the re-decomposition. em_damping in [0,1]
    # blends new/old covariate estimates (1 = full update). Requires rf_target_lags=0.
    em_iterations: int = 1
    em_damping: float = 1.0
    # Scale the ARIMA predictive sigma in the EM variant. Removing the covariate
    # effect before ARIMA shrinks its sigma, but the RF effect is added back as a
    # deterministic point — so the spread loses the covariate effect's prediction
    # uncertainty and under-disperses. >1 re-inflates to recover calibration.
    em_sigma_scale: float = 1.0

    # Output post-processing
    discretize_samples: bool = False
    random_seed: int = 42

    rf: RandomForestConfig = Field(default_factory=RandomForestConfig)

    def lags_by_col(self) -> dict[str, tuple[int, int]]:
        """``covariate_lags`` normalized to ``{name: (min, max)}`` tuples.

        Accepts either a 2-element ``[min, max]`` list or a single ``[lag]``
        (treated as ``[lag, lag]``); raises on any other shape.
        """
        out: dict[str, tuple[int, int]] = {}
        for name, span in self.covariate_lags.items():
            if len(span) == 1:
                lo = hi = int(span[0])
            elif len(span) == 2:
                lo, hi = int(span[0]), int(span[1])
            else:
                raise ValueError(
                    f"covariate_lags[{name!r}] must be [min, max] or [lag], got {span!r}"
                )
            if lo > hi:
                raise ValueError(f"covariate_lags[{name!r}] has min > max: {span!r}")
            out[name] = (lo, hi)
        return out


class ChapModelConfiguration(BaseModel):
    """Wrapper YAML that ``chap eval``/``chap forecast`` hands to the model."""

    model_config = ConfigDict(extra="forbid")

    additional_continuous_covariates: list[str] = Field(default_factory=list)
    user_option_values: RunConfig = Field(default_factory=RunConfig)


def load_model_configuration(path: str | Path) -> ChapModelConfiguration:
    """Load and validate a CHAP model-configuration YAML.

    An empty ``{}`` mapping is fine and yields all defaults (no covariates).
    """
    with Path(path).open("r") as f:
        return ChapModelConfiguration.model_validate(yaml.safe_load(f) or {})
