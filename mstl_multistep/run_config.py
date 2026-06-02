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

    # Recursive multistep trend model (fit on the deseasonalized series)
    n_target_lags: int = 6
    n_samples: int = 100
    feature_min_lag: int = 1
    feature_max_lag: int = 3
    prob_wrapper: ProbWrapper = "bootstrap"
    min_bucket_size: int = 5
    use_location_dummies: bool = True

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
