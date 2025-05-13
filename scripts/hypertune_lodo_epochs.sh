#!/bin/bash

timestamp=$(date +%Y%m%d-%H%M%S)
mkdir -p reports/hypertune_lodo

# ========== Configs for one-view models ==========
declare -A ONE_VIEW_CONFIGS
ONE_VIEW_CONFIGS=(
  ["potr_BMCLABS"]="BMCLABS.json"
  ["potr_T-SDU-PD"]="T-SDU-PD.json"
  ["potr_PDGAM"]="PDGAM.json"
  ["potr_3DGAIT"]="3DGAIT.json"
  ["motionclip_T-SDU-PD"]="T-SDU-PD.json"
  ["motionclip_PDGAM"]="PDGAM.json"
  ["motionclip_3DGAIT"]="3DGAIT.json"
  ["momask_T-SDU-PD"]="T-SDU-PD.json"
  ["momask_PDGAM"]="PDGAM.json"
  ["momask_3DGAIT"]="3DGAIT.json"
)

# ========== Configs for two-view models ==========
TWO_VIEW_CONFIGS=(
  "motionbert_BMCLABS_backright"
  "motionbert_BMCLABS_sideright"
  "motionbert_T-SDU-PD_backright"
  "motionbert_T-SDU-PD_sideright"
  "motionbert_PDGAM_backright"
  "motionbert_PDGAM_sideright"
  "motionbert_3DGAIT_backright"
  "motionbert_3DGAIT_sideright"

  "mixste_BMCLABS_backright"
  "mixste_BMCLABS_sideright"
  "mixste_T-SDU-PD_backright"
  "mixste_T-SDU-PD_sideright"
  "mixste_PDGAM_backright"
  "mixste_PDGAM_sideright"
  "mixste_3DGAIT_backright"
  "mixste_3DGAIT_sideright"

  "motionagformer_BMCLABS_backright"
  "motionagformer_BMCLABS_sideright"
  "motionagformer_T-SDU-PD_backright"
  "motionagformer_T-SDU-PD_sideright"
  "motionagformer_PDGAM_backright"
  "motionagformer_PDGAM_sideright"
  "motionagformer_3DGAIT_backright"
  "motionagformer_3DGAIT_sideright"

  "poseformerv2_BMCLABS_backright"
  "poseformerv2_BMCLABS_sideright"
  "poseformerv2_T-SDU-PD_backright"
  "poseformerv2_T-SDU-PD_sideright"
  "poseformerv2_PDGAM_backright"
  "poseformerv2_PDGAM_sideright"
  "poseformerv2_3DGAIT_backright"
  "poseformerv2_3DGAIT_sideright"
)

# ========== Run one-view models ==========
for model_key in "${!ONE_VIEW_CONFIGS[@]}"; do
  config_file="${ONE_VIEW_CONFIGS[$model_key]}"
  base_model=${model_key%%_*}
  config_name=$(basename "$config_file" .json)
  logfile="./reports/hypertune_lodo/${timestamp}-${base_model}-${config_name}_LODO.out"

  echo "🔁 Epoch-tune LODO (1-view): model=$base_model config=$config_file"
  python run.py \
    --backbone "$base_model" \
    --config "$config_file" \
    --ntrials 5 \
    --this_run_num 0 \
    --hypertune 1 \
    --tune_fresh 1 \
    --force_LODO 1 \
    --exp_name_rigid LODO &> "$logfile"
done

# ========== Run two-view models ==========
for model_config in "${TWO_VIEW_CONFIGS[@]}"; do
  base_model=${model_config%%_*}
  rest=${model_config#*_}
  config_file="${rest}.json"
  config_name=$(basename "$config_file" .json)
  logfile="./reports/hypertune_lodo/${timestamp}-${base_model}-${config_name}_LODO.out"

  echo "🔁 Epoch-tune LODO (2-view): model=$base_model config=$config_file"
  python run.py \
    --backbone "$base_model" \
    --config "$config_file" \
    --ntrials 5 \
    --this_run_num 0 \
    --hypertune 1 \
    --tune_fresh 1 \
    --force_LODO 1 \
    --exp_name_rigid LODO &> "$logfile"
done

echo "✅ All LODO hypertuning runs complete."
