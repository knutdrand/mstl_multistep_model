#!/usr/bin/env bash
# Frozen evaluation harness for the spray-dataset improvement session.
# DO NOT change the model, dataset, or backtest params between experiments —
# only the config (arg 1) and the output stem (arg 2) vary, so every run is
# byte-for-byte comparable.
#
# Usage: experiments/run_eval.sh <config.yaml> <out_stem>
#   -> writes output/<out_stem>.nc and logs to output/eval_<out_stem>.log
set -euo pipefail

CONFIG="${1:?config yaml required}"
STEM="${2:?output stem required}"

MODEL="/Users/knutdr/Sources/mstl_multistep_model"
DATASET="/Users/knutdr/Data/CH/chap_data_level5_irs_allocated_monthly.csv"
OUT="output/${STEM}.nc"
LOG="output/eval_${STEM}.log"

# MLflow tracking (global ~/.claude/CLAUDE.md sets MLFLOW_TRACKING_URI in ~/.zshrc;
# set a sensible default here too so the script is self-contained).
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-file://$HOME/chap-evaluations/mlruns}"
export MLFLOW_ALLOW_FILE_STORE=true

mkdir -p output
chap eval \
  --model-name "$MODEL" \
  --dataset-csv "$DATASET" \
  --output-file "$OUT" \
  --model-configuration-yaml "$CONFIG" \
  --backtest-params.n-periods 3 \
  --backtest-params.n-splits 12 \
  --backtest-params.stride 1 \
  --run-config.track 2>&1 | tee "$LOG"
