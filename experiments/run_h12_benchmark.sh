#!/usr/bin/env bash
set -euo pipefail
cd /Users/knutdr/Sources/mstl_multistep_model
declare -a M=(
  "config_irs_baseline.yaml h12_baseline"
  "config_irs_features.yaml h12_irs_features"
  "config_irs_lags.yaml h12_percov_lags"
  "config_tgtlag3.yaml h12_tgtlag3"
)
for row in "${M[@]}"; do set -- $row
  echo "===== EVAL $2 ($(date +%H:%M)) ====="
  ./experiments/run_eval_h12.sh "$1" "$2"
done
echo "===== METRICS ($(date +%H:%M)) ====="
export MLFLOW_TRACKING_URI="file://$HOME/chap-evaluations/mlruns"; export MLFLOW_ALLOW_FILE_STORE=true
chap export-metrics \
  --input-files output/h12_baseline.nc --input-files output/h12_irs_features.nc \
  --input-files output/h12_percov_lags.nc --input-files output/h12_tgtlag3.nc \
  --metric-ids crps_log1p --metric-ids crps \
  --output-file experiments/comparison_h12.csv
echo "===== DONE ($(date +%H:%M)) ====="; cat experiments/comparison_h12.csv
