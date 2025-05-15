import numpy as np
import sys
import os
from os.path import join as pjoin
import pandas as pd
import re

joints_num = 22
save_dir = "./dataset/healthy/"
data_dir = "./dataset/healthy/"
dataset_list = ['Healthy_data']
split = "train"

data_list = []
for dataset in dataset_list:
    dataset_dir = os.path.join(data_dir, dataset)
    # # --- Load and filter split annotations ---
    # split_csv = os.path.join(data_dir, "metadata")
    # csv_name = dataset + "_restructured_metadata.csv"
    # split_csv = os.path.join(split_csv, csv_name)
    # df = pd.read_csv(split_csv)
    # df_split = df[df['split'] == split]
    # walkIDs = df_split['walkID'].tolist()
    npz_path = os.path.join(dataset_dir, "HumanML3D/HumanML3D_collected.npz")
    data = np.load(npz_path, allow_pickle=True)
    
    # data_dict = {re.sub(r'_down\d*', '', key): data[key] for key in data.files if re.sub(r'_down\d*', '', key) in walkIDs}
    i = 1
    for key in data.files:
        # base = re.sub(r'_down.*$', '', key)
        motion = data[key]

        # print(motion.shape)
        # from utils.motion_process import recover_from_ric
        # import torch
        # joint = recover_from_ric(torch.from_numpy(motion).float(), 22).numpy()
        # print(joint.shape)
        # from utils.plot_script import plot_3d_motion
        # from utils.paramUtil import t2m_kinematic_chain
        # kinematic_chain = t2m_kinematic_chain
        # save_path = f"./visualization_healthy/sample_healthy_{i}.mp4"
        # plot_3d_motion(save_path, kinematic_chain, joint, title="test_healthy", fps=20)
        # i +=1
        
        # if (base in walkIDs):
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