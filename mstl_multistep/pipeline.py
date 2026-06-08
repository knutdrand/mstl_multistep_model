"""Entry point: build the published model (MSTL + fixed-order ARIMA + RF-residual)."""

from __future__ import annotations

from mstl_multistep.rf_residual import ArimaBaseRFResidualModel
from mstl_multistep.run_config import RunConfig


def build_chap_model(cfg: RunConfig, feature_columns: list[str]) -> ArimaBaseRFResidualModel:
    """Return the model that CHAP fits and predicts with."""
    return ArimaBaseRFResidualModel(cfg, feature_columns)
