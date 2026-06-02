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
    use_location_dummies: bool = True
    # MSTL-deseasonalize the covariates before lagging, so the RF sees climate
    # anomalies (matching the deseasonalized target) rather than raw seasonal series.
    deseasonalize_covariates: bool = True

    # How the deseasonalized series is forecast probabilistically:
    #   "multistep"          -> recursive RF from simple_multistep_model + prob_wrapper
    #   "arima_residual"     -> deterministic RF point + AutoARIMA on one-step OOB residuals
    #   "recursive_residual" -> RF point with one-step OOB residuals resampled and
    #                           compounded through the recursion (variance grows with horizon)
    prob_model: Literal["multistep", "arima_residual", "recursive_residual"] = "multistep"

    # multistep-only
    prob_wrapper: ProbWrapper = "bootstrap"
    min_bucket_size: int = 5

    # arima_residual-only (AutoARIMA on the RF out-of-bag residuals)
    arima_approximation: bool = False
    arima_stepwise: bool = True
    arima_level: int = 68

    # Output post-processing
    discretize_samples: bool = False
    random_seed: int = 42

    rf: RandomForestConfig = Field(default_factory=RandomForestConfig)


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
