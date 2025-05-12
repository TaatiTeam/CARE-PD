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
from data.preprocessing.preprocessing_utils import get_6D_rep_from_24x3_pose
from data.preprocessing.trajectory_correction import transform_seq_so_it_has_no_slope_h36m
from data.preprocessing.smpl2h36m import qbetween_np, qrot_np
from scipy.spatial.transform import Rotation as R
from data.preprocessing.human_body_prior.body_model.body_model import BodyModel

def apply_rot_to_pose_world(pose_world, rot_mats):
    """
    Rotate SMPL pose axis-angle representation using global rotation matrices.
    
    Args:
        pose_world: (N, 24, 3) - axis-angle pose parameters
        rot_mats:   (N, 3, 3)  - rotation matrices from slope correction

    Returns:
        pose_world_rotated: (N, 24, 3) - rotated axis-angle pose
    """
    rotated_pose = []

    for i in range(len(pose_world)):
        frame_pose = pose_world[i]  # (24, 3)
        rotated_frame = []

        for joint_aa in frame_pose:
            joint_rot = R.from_rotvec(joint_aa)
            global_rot = R.from_matrix(rot_mats[i])
            combined_rot = global_rot * joint_rot  # apply global slope correction
            rotated_frame.append(combined_rot.as_rotvec())

        rotated_pose.append(rotated_frame)

    return np.array(rotated_pose)  # (N, 24, 3)

def generate_smpl_in_world(smpl_model, sequence, down_sample_rate, down):
    frame_number = sequence['pose'].shape[0]

    if sequence['beta'].shape[0] != frame_number:
        sequence['beta'] = np.tile(sequence['beta'], (frame_number, 1))
    
    pose_world    = sequence['pose'].reshape(-1, 24, 3)  # (num_frames, 24, 3)
    betas         = sequence['beta']  # (num_frames, 10)
    world_trans   = sequence['trans']  # (num_frames, 3)
    
    pose_world    = pose_world[down::down_sample_rate,...]  # (num_frames, 24, 3)  # start from down and they select every down_sample_rate
    pose_world_out = pose_world.copy()
    betas         = betas[down::down_sample_rate,...]  # (num_frames, 10)
    world_trans   = world_trans[down::down_sample_rate,...] 
    
    frame_number = pose_world.shape[0]
    
    # Extract global orientation (index 0) and body pose (indices 1-23)
    global_orient = torch.tensor(pose_world[:, 0:1, :], dtype=torch.float32)  # (num_frames, 1, 3)
    body_pose     = torch.tensor(pose_world[:, 1:24, :], dtype=torch.float32)  # (num_frames, 23, 3)
    betas         = torch.tensor(betas, dtype=torch.float32)  # (num_frames, 10)

    # Ensure everything is on the same device
    global_orient = global_orient.reshape(frame_number, -1).to(_DEVICE)
    body_pose = body_pose.reshape(frame_number, -1).to(_DEVICE)
    betas = betas.reshape(frame_number, -1).to(_DEVICE)
    world_trans = torch.tensor(world_trans, dtype=torch.float32).to(_DEVICE)  # Ensure on same device

    # Zero values for face, hands, and expression
    zero_pose = torch.zeros((frame_number, 3), dtype=torch.float32).to(_DEVICE)
    zero_hand_pose = torch.zeros((frame_number, 15, 3), dtype=torch.float32).to(_DEVICE)
    zero_expression = torch.zeros((frame_number, 10), dtype=torch.float32).to(_DEVICE)

    # Generate SMPL output
    out = smpl_model(betas=betas, body_pose=body_pose, global_orient=global_orient, 
                     jaw_pose=zero_pose, leye_pose=zero_pose, reye_pose=zero_pose,
                     left_hand_pose=zero_hand_pose, right_hand_pose=zero_hand_pose,
                     expression=zero_expression)

    # Apply global translation (world_trans) to the output vertices
    out.vertices += world_trans[:, None, :]  # Broadcasting (num_frames, 1, 3) to (num_frames, num_vertices, 3)

    return out, pose_world_out


def compute_6D_motionclip_representation_from_pkl_SMPL_params(cfg):
    if cfg.slope_correction:
        ext = '_slopeCorrected'
    else:
        ext = ''
    cfg.OUT_PATH_f = cfg.OUT_PATH / f'6D_SMPL_30f_or_longer{ext}.npz'
    
    h36m_regressor = torch.tensor(np.load(cfg.H36M_J_REG), dtype=torch.float32).to(_DEVICE)
    smpl_model = SMPL(model_path=cfg.MODEL_PATH, num_betas=10).to(_DEVICE)
    
    all_smpls = joblib.load(cfg.DATA_DIR)
    print(f'Number of walks: {len(all_smpls)}')
    
    result = dict()
    for subject_id in tqdm(all_smpls):
        for walk_id in all_smpls[subject_id]:
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

                out_world, pose_world = generate_smpl_in_world(smpl_model, smpl_data, down_sample_rate, down) # (num_frames, 24, 3) axis-angle per frame

                vertices_world = out_world.vertices#.cpu().detach().numpy()  # (n_frames, n_vertices, 3)
                pose_world_aligned = pose_world
                if cfg.slope_correction:
                    h36m_joints_world = vertices2joints(h36m_regressor, vertices_world).cpu().detach().numpy()
                    _, rot_mats = transform_seq_so_it_has_no_slope_h36m(h36m_joints_world, n_frames_est_mov_dir=15, window_size=90, polynomial = 4, return_rot_matrices=True)
                    
                    pose_world_aligned = apply_rot_to_pose_world(pose_world_aligned, rot_mats)

                pose6d = get_6D_rep_from_24x3_pose(torch.tensor(pose_world_aligned))
                # print(f"pose6d: {pose6d.shape}") # shape (T, 25, 6)
                
                if pose6d.shape[0] >= 30:
                    result[walk_name] = pose6d
                else:
                    print(f"Discarding {walk_name} because it is less than 30 frames {pose6d.shape[0]}")
            
    np.savez(cfg.OUT_PATH_f, **result)
    print(f"Saved {cfg.OUT_PATH_f} with {len(result)} sequences")
     
            
 # SUPPORTED_DATASETS = ['BMCLab', 'T-SDU-PD', 'PD-GaM', '3DGait', 'DNE', 'E-LC', 'KUL-DT-T', 'T-LTC', 'T-SDU']    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-skl', '--skeleton_format', default='h36m')
    parser.add_argument('-db', '--dataset', default='3DGait')
    args = parser.parse_args()
    print(args) 

    assert args.dataset in SUPPORTED_DATASETS, f"Dataset is not supported!"
    
    cfg = SimpleNamespace()
    cfg.BASE_DIR = Path(path.PROJECT_ROOT)
    cfg.H36M_J_REG = Path('./data/preprocessing/common/J_regressor_h36m_correct.npy')
    cfg.MODEL_PATH = Path('./data/preprocessing/common/body_models/smpl/SMPL_NEUTRAL.pkl')
    cfg.DATA_DIR = cfg.BASE_DIR / 'assets' / 'datasets' / (args.dataset + '.pkl')
    cfg.OUT_PATH = Path(path.PROJECT_ROOT) / 'assets' / 'datasets'  / '6D_SMPL' / args.dataset
    
    print(cfg.DATA_DIR)
    print(cfg.OUT_PATH)
    
    DB_Slope_Correction = ['T-LTC', 'T-SDU', 'T-SDU-PD']
    cfg.db = args.dataset
    cfg.slope_correction = args.dataset in DB_Slope_Correction
    cfg.exfps = 30
    cfg.fps = DATASET_ORIGINAL_FPS[args.dataset]
    

    os.makedirs(cfg.OUT_PATH, exist_ok=True)
    print('Performing SMPL to 2D conversion...')
    compute_6D_motionclip_representation_from_pkl_SMPL_params(cfg)

    
