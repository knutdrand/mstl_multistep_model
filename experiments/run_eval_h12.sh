#!/usr/bin/env bash
# Long-horizon benchmark harness: 12-month-ahead, 14 splits, stride 1.
# Separate from the frozen 12x3x1 harness; results are NOT comparable to it.
# Usage: experiments/run_eval_h12.sh <config.yaml> <out_stem>
set -euo pipefail
CONFIG="${1:?config}"; STEM="${2:?stem}"
MODEL="/Users/knutdr/Sources/mstl_multistep_model"
DATASET="/Users/knutdr/Data/CH/chap_data_level5_irs_allocated_monthly.csv"
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-file://$HOME/chap-evaluations/mlruns}"
export MLFLOW_ALLOW_FILE_STORE=true
mkdir -p output
chap eval --model-name "$MODEL" --dataset-csv "$DATASET" \
  --output-file "output/${STEM}.nc" --model-configuration-yaml "$CONFIG" \
  --backtest-params.n-periods 12 --backtest-params.n-splits 14 --backtest-params.stride 1 \
  --run-config.track 2>&1 | tee "output/eval_${STEM}.log"
