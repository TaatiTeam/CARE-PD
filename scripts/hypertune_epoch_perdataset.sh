#!/bin/bash

# ========== Setup ==========
timestamp=$(date +%Y%m%d-%H%M%S)
mkdir -p reports/hypertune_epochs

# ========== Configs for Epoch-Only Tuning ==========
declare -A CONFIGS
CONFIGS=(
  ["potr_T-SDU-PD"]="T-SDU-PD.json"
  ["potr_PDGAM"]="PDGAM.json"
  ["potr_3DGAIT"]="3DGAIT.json"

  ["poseformerv2_back_T-SDU-PD"]="T-SDU-PD_backright.json"
  ["poseformerv2_side_T-SDU-PD"]="T-SDU-PD_sideright.json"
  ["poseformerv2_back_PDGAM"]="PDGAM_backright.json"
  ["poseformerv2_side_PDGAM"]="PDGAM_sideright.json"
  ["poseformerv2_back_3DGAIT"]="3DGAIT_backright.json"
  ["poseformerv2_side_3DGAIT"]="3DGAIT_sideright.json"

  ["motionbert_back_T-SDU-PD"]="T-SDU-PD_backright.json"
  ["motionbert_side_T-SDU-PD"]="T-SDU-PD_sideright.json"
  ["motionbert_back_PDGAM"]="PDGAM_backright.json"
  ["motionbert_side_PDGAM"]="PDGAM_sideright.json"
  ["motionbert_back_3DGAIT"]="3DGAIT_backright.json"
  ["motionbert_side_3DGAIT"]="3DGAIT_sideright.json"

  ["motionagformer_back_T-SDU-PD"]="T-SDU-PD_backright.json"
  ["motionagformer_side_T-SDU-PD"]="T-SDU-PD_sideright.json"
  ["motionagformer_back_PDGAM"]="PDGAM_backright.json"
  ["motionagformer_side_PDGAM"]="PDGAM_sideright.json"
  ["motionagformer_back_3DGAIT"]="3DGAIT_backright.json"
  ["motionagformer_side_3DGAIT"]="3DGAIT_sideright.json"

  ["motionclip_T-SDU-PD"]="T-SDU-PD.json"
  ["motionclip_PDGAM"]="PDGAM.json"
  ["motionclip_3DGAIT"]="3DGAIT.json"

  ["momask_T-SDU-PD"]="T-SDU-PD.json"
  ["momask_PDGAM"]="PDGAM.json"
  ["momask_3DGAIT"]="3DGAIT.json"

  ["mixste_back_T-SDU-PD"]="T-SDU-PD_backright.json"
  ["mixste_side_T-SDU-PD"]="T-SDU-PD_sideright.json"
  ["mixste_back_PDGAM"]="PDGAM_backright.json"
  ["mixste_side_PDGAM"]="PDGAM_sideright.json"
  ["mixste_back_3DGAIT"]="3DGAIT_backright.json"
  ["mixste_side_3DGAIT"]="3DGAIT_sideright.json"
)

# ========== Run Loop ==========
for model_key in "${!CONFIGS[@]}"; do
  config_file="${CONFIGS[$model_key]}"
  base_model=${model_key%%_*}
  config_name=$(basename "$config_file" .json)
  logfile="./reports/hypertune_epochs/${timestamp}-${base_model}-${config_name}.out"

  echo "🔹 Epoch-only tuning: model=$base_model config=$config_file"
  python run.py \
    --backbone "$base_model" \
    --ntrials 5 \
    --this_run_num 0 \
    --hypertune 1 \
    --tune_fresh 1 \
    --config="$config_file" &> "$logfile"
done

echo "✅ Epoch-only tuning complete for all datasets."
