import os
import torch
import joblib
import numpy as np
from pathlib import Path
from tqdm import tqdm
from smplx.lbs import vertices2joints
from smplx.body_models import SMPL
from types import SimpleNamespace
import argparse
import sys
import pandas as pd
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from const.const import _DEVICE, SUPPORTED_DATASETS, DATASET_ORIGINAL_FPS
from const import path

from data.preprocessing.transforms.quaternion import *
from data.preprocessing.transforms.paramUtil import t2m_raw_offsets, t2m_kinematic_chain
from data.preprocessing.transforms.skeleton import Skeleton
from data.preprocessing.create_redundant_representation import process_file, recover_from_ric
from data.preprocessing.trajectory_correction_amass import transform_seq_so_it_has_no_slope_AMASS
from data.preprocessing.human_body_prior.body_model.body_model import BodyModel



def generate_smpl_to_pose(bm, sequence, down_sample_rate, down):
    frame_number = sequence['pose'].shape[0]
    
    if sequence['beta'].shape[0] != frame_number:
        sequence['beta'] = np.tile(sequence['beta'], (frame_number, 1))
    
    pose_world    = sequence['pose'].reshape(-1, 24, 3)  # (num_frames, 24, 3)
    betas         = sequence['beta']  # (num_frames, 10)
    world_trans   = sequence['trans']  # (num_frames, 3)
    

    # Extract global orientation (index 0) and body pose (indices 1-23)
    global_orient = torch.tensor(pose_world[:, 0:1, :], dtype=torch.float32)  # (num_frames, 1, 3)
    body_pose     = torch.tensor(pose_world[:, 1:24, :], dtype=torch.float32)  # (num_frames, 23, 3)
    betas         = torch.tensor(betas, dtype=torch.float32)  # (num_frames, 10)
    
    if down_sample_rate > 1:
        global_orient = global_orient[down::down_sample_rate,...]  # (num_frames, 24, 3)  # start from down and they select every down_sample_rate
        body_pose     = body_pose[down::down_sample_rate,...]  # (num_frames, 10)
        betas         = betas[down::down_sample_rate,...] 
        world_trans         = world_trans[down::down_sample_rate,...]
        frame_number  = body_pose.shape[0]

    # Ensure everything is on the same device
    global_orient = global_orient.reshape(frame_number, -1).to(_DEVICE)
    body_pose = body_pose.reshape(frame_number, -1).to(_DEVICE)
    betas = betas.reshape(frame_number, -1).to(_DEVICE)
    world_trans = torch.tensor(world_trans, dtype=torch.float32).to(_DEVICE)  # Ensure on same device

    # Zero values for face, hands, and expression
    zero_pose = torch.zeros((frame_number, 3), dtype=torch.float32).to(_DEVICE)
    zero_hand_pose = torch.zeros((frame_number, 15, 3), dtype=torch.float32).to(_DEVICE)
    zero_expression = torch.zeros((frame_number, 10), dtype=torch.float32).to(_DEVICE)
    
    body_parms = {
            'root_orient': global_orient,
            'pose_body': body_pose,
            'trans': world_trans,
            'betas': betas,
            'pose_hand': None
        }
    
    with torch.no_grad():
        body = bm(**body_parms)
        
    pose_seq_np = body.Jtr.detach().cpu().numpy()
    # pose_seq_np_n = np.dot(pose_seq_np, trans_matrix)

    data = pose_seq_np[:, :22] # remove last two joints as models are using AMASS with 22 joints instead of 24
    
    ###################################
    # at this point pd pose is aready at Z+  
    # so no need to data[..., 0] *= -1
    ###################################

    return data 

def MirrorReflection(sequence):
    """
    Does horizontal flipping for each frame of the sequence.
    """

    left = [1, 4, 7, 10, 13, 16, 18, 20]
    right = [2, 5, 8, 11, 14, 17, 19, 21]

    mirrored_sequence = sequence.copy()
    mirrored_sequence[..., 0] *= -1
    mirrored_sequence[:, left + right, :] = mirrored_sequence[:, right + left, :]

    return mirrored_sequence
     
def create_humanml3d_rep_from_smpl(cfg, bm):

    all_smpls = joblib.load(cfg.DATA_DIR)
    
    all_sample_count = 0
    out_data = {}
    for subject_id in tqdm(all_smpls):
        for walk_id in all_smpls[subject_id]:
            all_sample_count += 1
            smpl_data = all_smpls[subject_id][walk_id]
            
            if smpl_data['pose'].shape[0] < 30:
                    # print(f"Skipping {walk_name} with {smpl_data['pose'].shape[0]} frames")
                    continue
            
            if 'Trimmed' in walk_id:
                continue

            down_sample_rate = max(1, int(cfg.fps / cfg.exfps))
            
            for down in range(down_sample_rate):
                if down_sample_rate == 1: 
                    walk_name = str(subject_id) + '__' + str(walk_id)   
                else:
                    walk_name = str(subject_id) + '__' + str(walk_id) + f'_down{down}'  

                pose = generate_smpl_to_pose(bm, smpl_data, down_sample_rate, down)
                
                if cfg.slope_correction:
                    pose = transform_seq_so_it_has_no_slope_AMASS(pose)

                # ------ Create Redundant HumanML3D data representation ------
                try:
                    data, ground_positions, positions, l_velocity = process_file(pose, cfg.feet_thre, cfg.tgt_offsets, cfg.face_joint_indx, cfg.fid_l, cfg.fid_r, cfg.l_idx1, cfg.l_idx2, cfg.n_raw_offsets, cfg.kinematic_chain)   
                    # rec_ric_data = recover_from_ric(torch.from_numpy(data).unsqueeze(0).float(), cfg.joints_num)
                    out_data[walk_name] = data
                    
                    mirrored_pose = MirrorReflection(pose)
                    mirrored_data, ground_positions, positions, l_velocity = process_file(mirrored_pose, cfg.feet_thre, cfg.tgt_offsets, cfg.face_joint_indx, cfg.fid_l, cfg.fid_r, cfg.l_idx1, cfg.l_idx2, cfg.n_raw_offsets, cfg.kinematic_chain)
                    out_data[walk_name+'_M'] = mirrored_data
                    
                    
                except Exception as e:
                    print(walk_name)
                    print(e)
    print(f"Total number of samples: {len(out_data)}")                
    np.savez(cfg.OUT_PATH / f'HumanML3D_collected.npz', **out_data)
     
            
 # SUPPORTED_DATASETS = ['BMCLab', 'T-SDU-PD', 'PD-GaM', '3DGAIT', 'DNE', 'E-LC', 'KUL-DT-T', 'T-LTC', 'T-SDU']    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-skl', '--skeleton_format', default='h36m')
    parser.add_argument('-db', '--dataset', default='3DGait')
    args = parser.parse_args()
    print(args) 

    assert args.dataset in SUPPORTED_DATASETS, f"Dataset is not supported!"
    
    cfg = SimpleNamespace()
    cfg.BASE_DIR = Path(path.PROJECT_ROOT)
    cfg.MODEL_PATH = Path('./data/preprocessing/common/body_models/smpl/SMPL_NEUTRAL.pkl')
    cfg.DATA_DIR = cfg.BASE_DIR / 'assets' / 'datasets' / (args.dataset + '.pkl')
    cfg.OUT_PATH = Path(path.PROJECT_ROOT) / 'assets' / 'datasets'  / 'HumanML3D' / args.dataset
    
    print(cfg.DATA_DIR)
    print(cfg.OUT_PATH)
    
    DB_Slope_Correction = ['T-LTC', 'T-SDU', 'T-SDU-PD']
    cfg.db = args.dataset
    cfg.slope_correction = args.dataset in DB_Slope_Correction
    cfg.exfps = 30
    cfg.fps = DATASET_ORIGINAL_FPS[args.dataset]
    
    num_betas = 10 # number of body parameters
    neutral_bm_path = './data/preprocessing/common/body_models/smpl/SMPL_NEUTRAL.pkl'
    neutral_bm = BodyModel(bm_fname=neutral_bm_path, num_betas=num_betas).to(_DEVICE)
    
    cfg.joints_num = 22
    example_id = "000021"
    # Lower legs
    cfg.l_idx1, cfg.l_idx2 = 5, 8
    # Right/Left foot
    cfg.fid_r, cfg.fid_l = [8, 11], [7, 10]
    # Face direction, r_hip, l_hip, sdr_r, sdr_l
    cfg.face_joint_indx = [2, 1, 17, 16]
    # l_hip, r_hip
    cfg.r_hip, cfg.l_hip = 2, 1
    cfg.n_raw_offsets = torch.from_numpy(t2m_raw_offsets)
    cfg.kinematic_chain = t2m_kinematic_chain
    # Get offsets of target skeleton
    example_data = np.load(os.path.join('./data/preprocessing/transforms/', example_id + '.npy'))
    example_data = example_data.reshape(len(example_data), -1, 3)
    example_data = torch.from_numpy(example_data)
    tgt_skel = Skeleton(cfg.n_raw_offsets, cfg.kinematic_chain, 'cpu')
    cfg.tgt_offsets = tgt_skel.get_offsets_joints(example_data[0])
    cfg.feet_thre = 0.002

        
    cfg.H = 1000
    cfg.W = 1000

    os.makedirs(cfg.OUT_PATH, exist_ok=True)
    create_humanml3d_rep_from_smpl(cfg, neutral_bm)

    
