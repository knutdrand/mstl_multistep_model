"""Generate predictions from a trained MSTL + multistep trend model."""

import argparse
import pickle

import pandas as pd

from mstl_multistep import ChapModelConfiguration, MSTLMultistepModel, load_model_configuration


def predict(
    model_path: str,
    historic_data_path: str,
    future_data_path: str,
    out_file_path: str,
    config_path: str | None = None,
) -> None:
    # The config travels in the pickle, but CHAP passes it again; loading it
    # here keeps parity with the standalone simple_multistep_model scripts.
    _ = load_model_configuration(config_path) if config_path else ChapModelConfiguration()

    with open(model_path, "rb") as f:
        model: MSTLMultistepModel = pickle.load(f)

    historic = pd.read_csv(historic_data_path)
    future = pd.read_csv(future_data_path)

    predictions = model.predict(historic, future)
    predictions.to_csv(out_file_path, index=False)
    print(f"Predictions saved to {out_file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate MSTL + multistep predictions")
    parser.add_argument("model", help="Path to trained model file")
    parser.add_argument("historic_data", help="Path to historic data CSV file")
    parser.add_argument("future_data", help="Path to future data CSV file")
    parser.add_argument("out_file", help="Path to save predictions CSV file")
    parser.add_argument(
        "--config",
        default=None,
        help="Optional path to a CHAP model_configuration_for_run.yaml",
    )
    args = parser.parse_args()
    predict(args.model, args.historic_data, args.future_data, args.out_file, args.config)
