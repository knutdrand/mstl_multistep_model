#!/usr/bin/env bash
set -euo pipefail
cd /Users/knutdr/Sources/mstl_multistep_model
for v in seasonal2 seasonal3; do
  echo "===== EVAL $v ($(date +%H:%M)) ====="
  ./experiments/run_eval.sh config_$v.yaml irs_$v
done
echo "===== METRICS ($(date +%H:%M)) ====="
export MLFLOW_TRACKING_URI="file://$HOME/chap-evaluations/mlruns"; export MLFLOW_ALLOW_FILE_STORE=true
chap export-metrics --input-files output/irs_tgtlag3.nc --input-files output/irs_seasonal2.nc \
  --input-files output/irs_seasonal3.nc --metric-ids crps_log1p --metric-ids crps \
  --output-file experiments/comparison_seasonal.csv
echo "===== DONE ($(date +%H:%M)) ====="; cat experiments/comparison_seasonal.csv
