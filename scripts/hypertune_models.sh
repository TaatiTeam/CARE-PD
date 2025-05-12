#!/bin/bash

# ========== Setup ==========
timestamp=$(date +%Y%m%d-%H%M%S)
mkdir -p reports/hypertune

# ========== Configs per Model ==========
declare -A CONFIGS
CONFIGS=(
  ["potr"]="BMCLab.json"
  ["poseformerv2_back"]="BMCLab_backright.json"
  ["poseformerv2_side"]="BMCLab_sideright.json"
  ["motionclip"]="BMCLab.json"
  ["motionbert_back"]="BMCLab_backright.json"
  ["motionbert_side"]="BMCLab_sideright.json"
  ["motionagformer_back"]="BMCLab_backright.json"
  ["motionagformer_side"]="BMCLab_sideright.json"
  ["momask"]="BMCLab.json"
  ["mixste_back"]="BMCLab_backright.json"
  ["mixste_side"]="BMCLab_sideright.json"
)

# ========== Run Loop ==========
for model_key in "${!CONFIGS[@]}"; do
  config_file="${CONFIGS[$model_key]}"
  base_model=${model_key%%_*}  # e.g., motionbert from motionbert_back or mixste
  config_name=$(basename "$config_file" .json)
  logfile="./reports/hypertune/${timestamp}-${base_model}-${config_name}.out"

  echo "🔹 Running hypertune: model=$base_model config=$config_file trials=50"
  python run.py \
    --backbone "$base_model" \
    --ntrials 50 \
    --this_run_num 0 \
    --hypertune 1 \
    --tune_fresh 1 \
    --config="$config_file" &> "$logfile"
done

echo "✅ All hypertuning jobs finished (or queued)."