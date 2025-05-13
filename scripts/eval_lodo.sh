#!/bin/bash

timestamp=$(date +%Y%m%d-%H%M%S)
mkdir -p reports/lodo_eval

# ========== One-view models ==========
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

for model_key in "${!ONE_VIEW_CONFIGS[@]}"; do
  config_file="${ONE_VIEW_CONFIGS[$model_key]}"
  base_model=${model_key%%_*}
  config_name=$(basename "$config_file" .json)
  logfile="./reports/lodo_eval/${timestamp}-${base_model}-${config_name}_LODO_cross.out"

  echo "🌐 LODO eval (1-view): model=$base_model config=$config_file"
  python eval_encoder_hypertune.py \
    --backbone "$base_model" \
    --config "$config_file" \
    --this_run_num 0 \
    --hypertune 0 \
    --cross_dataset_test 1 \
    --force_LODO 1 \
    --exp_name_rigid LODO &> "$logfile"
done

# ========== Two-view models ==========
TWO_VIEW_MODELS=(
  "mixste_BMCLABS"
  "mixste_T-SDU-PD"
  "mixste_PDGAM"
  "mixste_3DGAIT"
  "motionagformer_BMCLABS"
  "motionagformer_T-SDU-PD"
  "motionagformer_PDGAM"
  "motionagformer_3DGAIT"
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
  logfile="./reports/lodo_eval/${timestamp}-${base_model}-${dataset_name}_LODO_cross_combined.out"

  echo "🌐 LODO eval (2-view): model=$base_model dataset=$dataset_name"
  python eval_encoder_hypertune.py \
    --backbone "$base_model" \
    --hypertune 0 \
    --cross_dataset_test 1 \
    --force_LODO 1 \
    --exp_name_rigid LODO \
    --combine_views_preds 1 \
    --views_path \
      "LODO/${base_model}_${dataset_name}_backright_LODO/0" \
      "LODO/${base_model}_${dataset_name}_sideright_LODO/0" &> "$logfile"
done

echo "✅ LODO cross-dataset evaluations complete."
