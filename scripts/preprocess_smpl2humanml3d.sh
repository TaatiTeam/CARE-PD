#!/bin/bash

echo "🚀 Converting SMPL to HumanML3D format for all datasets..."

# DATASETS=('BMCLab' 'T-SDU-PD' 'PD-GaM' '3DGait' 'DNE' 'E-LC' 'KUL-DT-T' 'T-LTC' 'T-SDU')
DATASETS=('E-LC' 'KUL-DT-T' 'T-LTC' 'T-SDU')

for db in "${DATASETS[@]}"
do
    echo "🔄 Processing dataset: $db"
    python data/preprocessing/smpl2humanml3d.py -db "$db"
done

echo "✅ All datasets processed."
