import numpy as np
import sys
import os
from os.path import join as pjoin
import pandas as pd
import re
import pickle
import glob
from pathlib import Path

joints_num = 22

matches = glob.glob(os.path.join('..', '..', 'assets', 'datasets', 'HumanML3D*'))
if not matches:
    raise FileNotFoundError("No HumanML3D folder found under ../../assets/datasets/")
data_dir = matches[0]
save_dir = data_dir

dataset_list = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]

split = "eval"

data_list = []
for dataset in dataset_list:

    dataset_dir = os.path.join(data_dir, dataset)

    # --- Load and filter split annotations ---
    UPDRS_list = ['3DGait', 'BMCLab', 'PD-GaM', 'T-SDU-PD']
    fold_dir = "../../assets/datasets/folds"
    if dataset in UPDRS_list:
        fold_path = os.path.join(fold_dir, "UPDRS_Datasets")
    else:
        fold_path = os.path.join(fold_dir, "Other_Datasets")

    if dataset == "PD-GaM":
        fold_path = os.path.join(fold_path, "PD-GaM_authors_fixed.pkl")
    elif dataset == "T-SDU-PD":
        fold_path = os.path.join(fold_path, "T-SDU-PD_PD_fixed.pkl")
    else:
        fold_path = os.path.join(fold_path, dataset + "_fixed.pkl")

    with open(fold_path, 'rb') as f:
        fold_dict = pickle.load(f)
    walkIDs = fold_dict[1][split]

    npz_path = os.path.join(dataset_dir, "HumanML3D_collected.npz")
    data = np.load(npz_path, allow_pickle=True)

    i = 1
    for key in data.files:
        base = key.split('__', 1)[0]
        motion = data[key]
        
        if base in walkIDs:
            data_list.append(motion)

data = np.concatenate(data_list, axis=0)
print(data.shape)
Mean = data.mean(axis=0)
Std = data.std(axis=0)
Std[0:1] = Std[0:1].mean() / 1.0
Std[1:3] = Std[1:3].mean() / 1.0
Std[3:4] = Std[3:4].mean() / 1.0
Std[4: 4+(joints_num - 1) * 3] = Std[4: 4+(joints_num - 1) * 3].mean() / 1.0
Std[4+(joints_num - 1) * 3: 4+(joints_num - 1) * 9] = Std[4+(joints_num - 1) * 3: 4+(joints_num - 1) * 9].mean() / 1.0
Std[4+(joints_num - 1) * 9: 4+(joints_num - 1) * 9 + joints_num*3] = Std[4+(joints_num - 1) * 9: 4+(joints_num - 1) * 9 + joints_num*3].mean() / 1.0
Std[4 + (joints_num - 1) * 9 + joints_num * 3: ] = Std[4 + (joints_num - 1) * 9 + joints_num * 3: ].mean() / 1.0
assert 8 + (joints_num - 1) * 9 + joints_num * 3 == Std.shape[-1]
np.save(pjoin(save_dir, 'Mean.npy'), Mean)
np.save(pjoin(save_dir, 'Std.npy'), Std)