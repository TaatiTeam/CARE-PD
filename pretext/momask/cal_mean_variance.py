import numpy as np
import sys
import os
from os.path import join as pjoin
import pandas as pd
import re
import pickle
import glob


joints_num = 22
save_dir = "../../assets/datasets/HumanML3D"
data_dir = "../../assets/datasets/HumanML3D"
dataset_list = ['DNE', '3DGait', 'BMCLab', 'PD-GAM']
split = "train"

UPDRS_list = ['3DGait', 'BMCLab', 'PD-GAM']
fold_dir = "../../assets/datasets/folds"

data_list = []
for dataset in dataset_list:

    dataset_dir = os.path.join(data_dir, dataset)
    # --- Load and filter split annotations ---
    if dataset in UPDRS_list:
        fold_path = os.path.join(fold_dir, "UPDRS_Datasets")
    else:
        fold_path = os.path.join(fold_dir, "Other_Datasets")

    if dataset == "PD-GAM":
        fold_path = os.path.join(fold_path, "PD-GaM_authors_fixed.pkl")
    else:
        fold_path = os.path.join(fold_path, dataset + "_fixed.pkl")
    
    print(fold_path)
    with open(fold_path, 'rb') as f:
        fold_dict = pickle.load(f)

    walkIDs = fold_dict[1][split]

    npz_path = os.path.join(dataset_dir, "HumanML3D_collected.npz")
    data = np.load(npz_path, allow_pickle=True)

    i = 1
    for key in data.files:
        motion = data[key]
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