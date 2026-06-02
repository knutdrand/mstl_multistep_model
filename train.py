"""Train the MSTL + multistep trend model."""

import argparse
import pickle

import pandas as pd

from mstl_multistep import ChapModelConfiguration, build_chap_model, load_model_configuration


def train(train_data_path: str, model_path: str, config_path: str | None = None) -> None:
    model_cfg = (
        load_model_configuration(config_path) if config_path else ChapModelConfiguration()
    )
    data = pd.read_csv(train_data_path)

    model = build_chap_model(
        cfg=model_cfg.user_option_values,
        feature_columns=model_cfg.additional_continuous_covariates,
    )
    model.fit(data)

    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an MSTL + multistep trend model")
    parser.add_argument("train_data", help="Path to training data CSV file")
    parser.add_argument("model", help="Path to save the trained model")
    parser.add_argument(
        "--config",
        default=None,
        help="Optional path to a CHAP model_configuration_for_run.yaml",
    )
    args = parser.parse_args()
    train(args.train_data, args.model, args.config)
