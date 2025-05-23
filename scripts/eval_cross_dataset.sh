#!/bin/bash

timestamp=$(date +%Y%m%d-%H%M%S)
mkdir -p reports/cross_eval

# ========== One-view models ==========
declare -A SINGLE_VIEW_CONFIGS
SINGLE_VIEW_CONFIGS=(
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

for model_key in "${!SINGLE_VIEW_CONFIGS[@]}"; do
  config_file="${SINGLE_VIEW_CONFIGS[$model_key]}"
  base_model=${model_key%%_*}
  config_name=$(basename "$config_file" .json)
  logfile="./reports/cross_eval/${timestamp}-${base_model}-${config_name}_cross.out"

  echo "🔄 Cross-dataset eval (1-view): model=$base_model config=$config_file"
  python run.py \
    --backbone "$base_model" \
    --hypertune 0 \
    --cross_dataset_test 1 \
    --this_run_num 0 \
    --config="$config_file" &> "$logfile"
done

# ========== Two-view models ==========
TWO_VIEW_MODELS=(
  "motionagformer_BMCLABS"
  "motionagformer_T-SDU-PD"
  "motionagformer_PDGAM"
  "motionagformer_3DGAIT"
  "mixste_BMCLABS"
  "mixste_T-SDU-PD"
  "mixste_PDGAM"
  "mixste_3DGAIT"
  "motionbert_BMCLABS"
  "motionbert_T-SDU-PD"
  "motionbert_PDGAM"
  "motionbert_3DGAIT"
  "poseformerv2_BMCLABS"
  "poseformerv2_T-SDU-PD"
  "poseformerv2_PDGAM"
  "poseformerv2_3DGAIT"
)

for model_dataset in "${TWO_VIEW_MODELS[@]}"; do
  base_model=${model_dataset%%_*}
  dataset_name=${model_dataset#*_}

  run_dir="Hypertune"

  logfile="./reports/cross_eval/${timestamp}-${base_model}-${dataset_name}_cross_combined.out"

  echo "🔄 Cross-dataset eval (2-view): model=$base_model dataset=$dataset_name"
  python run.py \
    --backbone "$base_model" \
    --hypertune 0 \
    --cross_dataset_test 1 \
    --combine_views_preds 1 \
    --views_path \
      "${run_dir}/${base_model}_${dataset_name}_backright/0" \
      "${run_dir}/${base_model}_${dataset_name}_sideright/0" &> "$logfile"
done

echo "✅ Cross-dataset evaluations complete."
