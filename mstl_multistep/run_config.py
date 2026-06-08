"""Pydantic config for one MSTL + ARIMA + RF-residual run.

Same wrapper shape CHAP hands every external model: model knobs under
``user_option_values`` and the list of feature columns under
``additional_continuous_covariates``. Defaults are the published champion; a config
typically only needs to name its covariate / IRS columns and (optionally) tune the lags.

Example YAML::

    additional_continuous_covariates: [rainfall_era5, mean_temperature, relative_humidity]
    user_option_values:
      irs_column: irs_allocated
      irs_features: [level, since, cumulative, chem_channels, decay2, decay8]
      irs_chemical_column: irs_insecticide_used
      covariate_lags: {rainfall_era5: [1, 6], relative_humidity: [1, 4], mean_temperature: [1, 2]}
      deseasonalize_covariates: [rainfall_era5, mean_temperature, relative_humidity]
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class RandomForestConfig(BaseModel):
    """sklearn RandomForestRegressor hyperparameters for the residual model."""

    model_config = ConfigDict(extra="forbid")

    n_estimators: int = 200
    max_depth: int | None = None
    min_samples_leaf: int = 3
    max_features: str | int | float | None = "sqrt"
    random_state: int | None = 42


class RunConfig(BaseModel):
    """Tunable knobs for one MSTL + ARIMA(fixed order) + RF-residual run.

    The pipeline: MSTL deseasonalises ``log1p(target)``; a per-location ARIMA of a shared
    fixed order forecasts the deseasonalised series (mean + spread); a pooled RandomForest
    corrects the ARIMA residual using lagged climate anomalies, engineered IRS features and
    lagged self-history; a location-scale variance head widens the spread by the forest's
    own uncertainty. See docs/champion_model.pdf.
    """

    model_config = ConfigDict(extra="forbid")

    target_variable: str = "disease_cases"
    log_transform: bool = True

    # --- MSTL seasonal decomposition ---
    season_length_monthly: int = 12
    season_length_weekly: int = 52

    # --- features: lagged climate anomalies + self-history ---
    feature_min_lag: int = 1
    feature_max_lag: int = 3
    # Per-covariate lag overrides: ``{name: [min, max]}`` inclusive window for that column,
    # overriding feature_min_lag/feature_max_lag (e.g. rainfall a longer span than temperature).
    covariate_lags: dict[str, list[int]] = Field(default_factory=dict)
    # Covariates to MSTL-deseasonalise before lagging, so the model sees their anomalies
    # (matching the deseasonalised target) rather than the raw seasonal series.
    deseasonalize_covariates: list[str] = Field(default_factory=list)
    # One-hot location identity for the pooled forest.
    use_location_dummies: bool = True
    # Lagged deseasonalised-target values fed to the RF (future lags bridged with the ARIMA mean).
    rf_target_lags: int = 3

    # --- ARIMA base (shared fixed order across all locations) ---
    # All locations share this (p, d, q) order; only the coefficients are fit per series. The
    # data's modal AutoARIMA order is [0, 1, 1] (= simple exponential smoothing); [0, 1, 2] is the
    # tuned default. arima_level is the predictive-interval level used to back out the spread sigma.
    arima_order: list[int] = Field(default_factory=lambda: [0, 1, 2])
    arima_level: int = 68

    # --- IRS (indoor residual spraying) feature extraction ---
    # When irs_column is set and irs_features is non-empty, dense protective-effect features are
    # engineered from the raw (sparse) allocation column. irs_halflife is the geometric decay (months)
    # of the plain ``decay`` feature. irs_chemical_column (e.g. "irs_insecticide_used") enables the
    # per-chemical decay channels in the feature bank. See mstl_multistep.irs_features.
    irs_column: str | None = None
    irs_features: list[str] = Field(default_factory=list)
    irs_halflife: float = 4.0
    irs_chemical_column: str | None = None

    # --- location-scale variance head ---
    # The RF correction is a point; "tree" widens the predictive spread by the forest's inter-tree
    # variance (sigma_eff^2 = sigma_ARIMA^2 + scale * v(x)). "none" disables it.
    residual_variance: Literal["none", "tree"] = "tree"
    residual_variance_scale: float = 0.5

    # --- output ---
    n_samples: int = 100
    random_seed: int = 42
    rf: RandomForestConfig = Field(default_factory=RandomForestConfig)

    def lags_by_col(self) -> dict[str, tuple[int, int]]:
        """``covariate_lags`` normalized to ``{name: (min, max)}`` tuples.

        Accepts a 2-element ``[min, max]`` list or a single ``[lag]`` (treated as ``[lag, lag]``).
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

    An empty ``{}`` mapping is fine and yields all defaults (the champion, no covariates).
    """
    with Path(path).open("r") as f:
        return ChapModelConfiguration.model_validate(yaml.safe_load(f) or {})
